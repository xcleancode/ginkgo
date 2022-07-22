/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_REORDER_RCM_HPP_
#define GKO_PUBLIC_CORE_REORDER_RCM_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>


namespace gko {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


enum class starting_strategy { minimum_degree, pseudo_peripheral };


/**
 * Rcm is a reordering algorithm minimizing the bandwidth of a matrix. Such a
 * reordering typically also significantly reduces fill-in, though usually not
 * as effective as more complex algorithms, specifically AMD and nested
 * dissection schemes. The advantage of this algorithm is its low runtime.
 *
 * @note  This class is derived from polymorphic object but is not a LinOp as it
 * does not make sense for this class to implement the apply methods. The
 * objective of this class is to generate a reordering/permutation vector (in
 * the form of the Permutation matrix), which can be used to apply to reorder a
 * matrix as required.
 *
 * There are two "starting strategies" currently available: minimum degree and
 * pseudo-peripheral. These strategies control how a starting vertex for a
 * connected component is choosen, which is then renumbered as first vertex in
 * the component, starting the algorithm from there.
 * In general, the bandwidths obtained by choosing a pseudo-peripheral vertex
 * are slightly smaller than those obtained from choosing a vertex of minimum
 * degree. On the other hand, this strategy is much more expensive, relatively.
 * The algorithm for finding a pseudo-peripheral vertex as
 * described in "Computer Solution of Sparse Linear Systems" (George, Liu, Ng,
 * Oak Ridge National Laboratory, 1994) is implemented here.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup reorder
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Mc64 : public EnablePolymorphicObject<Mc64<ValueType, IndexType>,
                                            ReorderingBase>,
             public EnablePolymorphicAssignment<Mc64<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Mc64, ReorderingBase>;

public:
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using PermutationMatrix = matrix::Permutation<IndexType>;
    using ScalingMatrix = matrix::Diagonal<ValueType>;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = gko::matrix::Csr<double, int>;
    using index_array = Array<IndexType>;

    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const index_array> get_permutation() const
    {
        return permutation_;
    }

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const index_array> get_inverse_permutation() const
    {
        return inv_permutation_;
    }

    std::shared_ptr<const ScalingMatrix> get_row_scaling() const
    {
        return row_scaling_;
    }

    std::shared_ptr<const ScalingMatrix> get_col_scaling() const
    {
        return col_scaling_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(construct_inverse_permutation, false);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(Mc64, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Mc64(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(std::move(exec))
    {}

    explicit Mc64(const Factory* factory, const ReorderingBaseArgs& args)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        auto exec = this->get_executor();
        const auto is_gpu_executor = exec != exec->get_master();
        auto cpu_exec = is_gpu_executor ? exec->get_master() : exec;
        auto system_matrix = args.system_matrix;

        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

        this->generate(exec, system_matrix);
    }

    void generate(std::shared_ptr<const Executor>& exec,
                  std::shared_ptr<LinOp> system_matrix);

private:
    std::shared_ptr<index_array> permutation_;
    std::shared_ptr<index_array> inv_permutation_;
    std::shared_ptr<ScalingMatrix> row_scaling_;
    std::shared_ptr<ScalingMatrix> col_scaling_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_RCM_HPP_
