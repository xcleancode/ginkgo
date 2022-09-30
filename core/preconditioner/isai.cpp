/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/preconditioner/isai.hpp>


#include <functional>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/base/utils.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace isai {
namespace {


GKO_REGISTER_OPERATION(generate_tri_inverse, isai::generate_tri_inverse);
GKO_REGISTER_OPERATION(generate_general_inverse,
                       isai::generate_general_inverse);
GKO_REGISTER_OPERATION(generate_excess_system, isai::generate_excess_system);
GKO_REGISTER_OPERATION(scale_excess_solution, isai::scale_excess_solution);
GKO_REGISTER_OPERATION(scatter_excess_solution, isai::scatter_excess_solution);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);


}  // anonymous namespace
}  // namespace isai


template <isai_type IsaiType, typename ValueType, typename IndexType>
void Isai<IsaiType, ValueType, IndexType>::generate_inverse(
    std::shared_ptr<const LinOp> input, bool skip_sorting, int power,
    IndexType excess_limit)
{
    using Dense = matrix::Dense<ValueType>;
    using LowerTrs = solver::LowerTrs<ValueType, IndexType>;
    using UpperTrs = solver::UpperTrs<ValueType, IndexType>;
    using Gmres = solver::Gmres<ValueType>;
    using Bj = preconditioner::Jacobi<ValueType, IndexType>;
    GKO_ASSERT_IS_SQUARE_MATRIX(input);
    auto exec = this->get_executor();
    auto is_lower = IsaiType == isai_type::lower;
    auto is_general = IsaiType == isai_type::general;
    auto is_spd = IsaiType == isai_type::spd;
    auto to_invert =
        gko::share(convert_to_with_sorting<Csr>(exec, input, skip_sorting));
    auto num_rows = to_invert->get_size()[0];
    std::shared_ptr<Csr> inverted;
    if (!is_spd) {
        inverted = gko::matrix::extend_sparsity(exec, to_invert, power);
    } else {
        // Extract lower triangular part: compute non-zeros
        array<IndexType> inverted_row_ptrs{exec, num_rows + 1};
        exec->run(isai::make_initialize_row_ptrs_l(
            to_invert.get(), inverted_row_ptrs.get_data()));

        // Get nnz from device memory
        auto inverted_nnz = static_cast<size_type>(
            exec->copy_val_to_host(inverted_row_ptrs.get_data() + num_rows));

        // Init arrays
        array<IndexType> inverted_col_idxs{exec, inverted_nnz};
        array<ValueType> inverted_vals{exec, inverted_nnz};
        auto inverted_base = share(Csr::create(
            exec, dim<2>{num_rows, num_rows}, std::move(inverted_vals),
            std::move(inverted_col_idxs), std::move(inverted_row_ptrs)));

        // Extract lower factor: columns and values
        exec->run(isai::make_initialize_l(to_invert.get(), inverted_base.get(),
                                          false));

        std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>>
            inverted_base_const =
                inverted_base;  // workaround to deal with argument type
                                // deduction failure
        inverted = power == 1 ? std::move(inverted_base)
                              : gko::matrix::extend_sparsity(
                                    exec, inverted_base_const, power);
    }

    // This stores the beginning of the RHS for the sparse block associated with
    // each row of inverted_l
    array<IndexType> excess_block_ptrs{exec, num_rows + 1};
    // This stores the beginning of the non-zeros belonging to each row in the
    // system of excess blocks
    array<IndexType> excess_row_ptrs_full{exec, num_rows + 1};

    if (is_general || is_spd) {
        exec->run(isai::make_generate_general_inverse(
            lend(to_invert), lend(inverted), excess_block_ptrs.get_data(),
            excess_row_ptrs_full.get_data(), is_spd));
    } else {
        exec->run(isai::make_generate_tri_inverse(
            lend(to_invert), lend(inverted), excess_block_ptrs.get_data(),
            excess_row_ptrs_full.get_data(), is_lower));
    }

    auto host_excess_block_ptrs_array =
        array<IndexType>(exec->get_master(), excess_block_ptrs);
    auto host_excess_block_ptrs = host_excess_block_ptrs_array.get_const_data();
    auto host_excess_row_ptrs_full_array =
        array<IndexType>(exec->get_master(), excess_row_ptrs_full);
    auto host_excess_row_ptrs_full =
        host_excess_row_ptrs_full_array.get_const_data();
    auto total_excess_dim = host_excess_block_ptrs[num_rows];
    auto excess_lim = excess_limit == 0 ? total_excess_dim : excess_limit;
    // if we had long rows:
    if (total_excess_dim > 0) {
        bool done = false;
        size_type block = 0;
        while (block < num_rows) {
            // build the excess sparse triangular system
            size_type excess_dim;
            size_type excess_start = block;
            const auto block_offset = host_excess_block_ptrs[block];
            const auto nnz_offset = host_excess_row_ptrs_full[block];
            for (excess_dim = 0; excess_dim < excess_lim && block < num_rows;
                 excess_dim = host_excess_block_ptrs[block] - block_offset) {
                block++;
            }
            if (excess_dim == 0) {
                break;
            }
            auto excess_nnz = host_excess_row_ptrs_full[block] - nnz_offset;
            auto excess_system =
                Csr::create(exec, dim<2>(excess_dim, excess_dim), excess_nnz);
            excess_system->set_strategy(
                std::make_shared<typename Csr::classical>());
            auto excess_rhs = Dense::create(exec, dim<2>(excess_dim, 1));
            auto excess_solution = Dense::create(exec, dim<2>(excess_dim, 1));
            exec->run(isai::make_generate_excess_system(
                lend(to_invert), lend(inverted),
                excess_block_ptrs.get_const_data(),
                excess_row_ptrs_full.get_const_data(), lend(excess_system),
                lend(excess_rhs), excess_start, block));
            // solve it after transposing
            auto system_copy = gko::clone(exec->get_master(), excess_system);
            auto rhs_copy = gko::clone(exec->get_master(), excess_rhs);
            std::shared_ptr<LinOpFactory> excess_solver_factory;
            if (parameters_.excess_solver_factory) {
                excess_solver_factory = parameters_.excess_solver_factory;
                excess_solution->copy_from(excess_rhs.get());
            } else if (is_general || is_spd) {
                excess_solver_factory =
                    Gmres::build()
                        .with_preconditioner(
                            Bj::build().with_max_block_size(32u).on(exec))
                        .with_criteria(
                            gko::stop::Iteration::build()
                                .with_max_iters(excess_dim)
                                .on(exec),
                            gko::stop::ResidualNorm<ValueType>::build()
                                .with_baseline(gko::stop::mode::rhs_norm)
                                .with_reduction_factor(
                                    remove_complex<ValueType>{1e-6})
                                .on(exec))
                        .on(exec);
                excess_solution->copy_from(excess_rhs.get());
            } else if (is_lower) {
                excess_solver_factory = UpperTrs::build().on(exec);
            } else {
                excess_solver_factory = LowerTrs::build().on(exec);
            }
            excess_solver_factory->generate(share(excess_system->transpose()))
                ->apply(lend(excess_rhs), lend(excess_solution));
            if (is_spd) {
                exec->run(isai::make_scale_excess_solution(
                    excess_block_ptrs.get_const_data(), lend(excess_solution),
                    excess_start, block));
            }
            // and copy the results back to the original ISAI
            exec->run(isai::make_scatter_excess_solution(
                excess_block_ptrs.get_const_data(), lend(excess_solution),
                lend(inverted), excess_start, block));
        }
    }

    approximate_inverse_ = std::move(inverted);
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
Isai<IsaiType, ValueType, IndexType>&
Isai<IsaiType, ValueType, IndexType>::operator=(const Isai& other)
{
    if (&other != this) {
        EnableLinOp<Isai>::operator=(other);
        auto exec = this->get_executor();
        approximate_inverse_ = other.approximate_inverse_;
        parameters_ = other.parameters_;
        if (approximate_inverse_ &&
            other.approximate_inverse_->get_executor() != exec) {
            approximate_inverse_ = gko::clone(exec, approximate_inverse_);
        }
    }
    return *this;
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
Isai<IsaiType, ValueType, IndexType>&
Isai<IsaiType, ValueType, IndexType>::operator=(Isai&& other)
{
    if (&other != this) {
        EnableLinOp<Isai>::operator=(std::move(other));
        auto exec = this->get_executor();
        approximate_inverse_ = std::move(other.approximate_inverse_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (approximate_inverse_ &&
            other.approximate_inverse_->get_executor() != exec) {
            approximate_inverse_ = gko::clone(exec, approximate_inverse_);
        }
    }
    return *this;
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
Isai<IsaiType, ValueType, IndexType>::Isai(const Isai& other)
    : Isai{other.get_executor()}
{
    *this = other;
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
Isai<IsaiType, ValueType, IndexType>::Isai(Isai&& other)
    : Isai{other.get_executor()}
{
    *this = std::move(other);
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Isai<IsaiType, ValueType, IndexType>::transpose() const
{
    auto is_spd = IsaiType == isai_type::spd;
    if (is_spd) {
        return this->clone();
    }

    std::unique_ptr<transposed_type> transp{
        new transposed_type{this->get_executor()}};
    transp->set_size(gko::transpose(this->get_size()));
    transp->approximate_inverse_ =
        share(as<Csr>(this->get_approximate_inverse())->transpose());

    return std::move(transp);
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Isai<IsaiType, ValueType, IndexType>::conj_transpose()
    const
{
    auto is_spd = IsaiType == isai_type::spd;
    if (is_spd) {
        return this->clone();
    }

    std::unique_ptr<transposed_type> transp{
        new transposed_type{this->get_executor()}};
    transp->set_size(gko::transpose(this->get_size()));
    transp->approximate_inverse_ =
        share(as<Csr>(this->get_approximate_inverse())->conj_transpose());

    return std::move(transp);
}


#define GKO_DECLARE_LOWER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::lower, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_ISAI);

#define GKO_DECLARE_UPPER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::upper, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_ISAI);

#define GKO_DECLARE_GENERAL_ISAI(ValueType, IndexType) \
    class Isai<isai_type::general, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GENERAL_ISAI);

#define GKO_DECLARE_SPD_ISAI(ValueType, IndexType) \
    class Isai<isai_type::spd, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPD_ISAI);


}  // namespace preconditioner
}  // namespace gko
