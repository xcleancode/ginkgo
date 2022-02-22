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

#include <ginkgo/core/solver/minres.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>


#include "core/solver/minres_kernels.hpp"


namespace gko {
namespace solver {
namespace minres {
namespace {


GKO_REGISTER_OPERATION(initialize, minres::initialize);
GKO_REGISTER_OPERATION(step_1, minres::step_1);


}  // anonymous namespace
}  // namespace minres


template <typename ValueType>
std::unique_ptr<LinOp> Minres<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Minres<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Minres<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


/**
 * This Minres implementation is based on Anne Grennbaum's 'Iterative Methods
 * for Solving Linear Systems' (DOI: 10.1137/1.9781611970937) Ch. 2 and Ch. 8.
 * Most variable names are taken from that reference, with the exception that
 * the vector `w` and `w_tilde` from the reference are called `z` and `z_tilde`
 * here. By reusing already allocated memory the number of necessary vectors is
 * reduced to The operations are reordered such that the number of kernel
 * launches can be minimized. Since the dot operations might require global
 * reductions, they are not grouped together with other steps.
 * The algorithm uses a recursion to compute an approximate residual norm. The
 * residual is neither computed exactly, nor approximately, since that would
 * require additional operations.
 */
template <typename ValueType>
void Minres<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);  // z = w_k+1
    auto p = Vector::create_with_config_of(dense_b);  // p = p_k-1
    auto q = Vector::create_with_config_of(dense_b);  // q = q_k+1
    auto v = Vector::create_with_config_of(dense_b);  // v = v_k

    auto z_tilde =
        Vector::create_with_config_of(dense_b);  // z_tilde = w_tilde_k+1

    auto p_prev = Vector::create_with_config_of(p.get());  // p_prev = p_k-2
    auto q_prev = Vector::create_with_config_of(q.get());  // q_prev = q_k

    auto alpha = Vector::create(
        exec, dim<2>{1, dense_b->get_size()[1]});  // alpha = T(k, k)
    auto beta = Vector::create_with_config_of(
        alpha.get());  // beta = T(k + 1, k) = T(k, k + 1)
    auto gamma =
        Vector::create_with_config_of(alpha.get());  // gamma = T(k - 1, k)
    auto delta =
        Vector::create_with_config_of(alpha.get());  // delta = T(k - 2, k)
    auto eta_next = Vector::create_with_config_of(alpha.get());
    auto eta = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::absolute_type::create(
        exec, dim<2>{1, dense_b->get_size()[1]});  // using ||z||, could also be
                                                   // beta or ||r||?

    auto cos_prev = Vector::create_with_config_of(alpha.get());
    auto cos = Vector::create_with_config_of(alpha.get());
    auto sin_prev = Vector::create_with_config_of(alpha.get());
    auto sin = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // r = dense_b
    r = clone(dense_b);
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());

    // z = M^-1 * r
    // beta = <r, z>
    // tau = ||z||_2
    get_preconditioner()->apply(r.get(), z.get());
    r->compute_conj_dot(z.get(), beta.get());
    z->compute_norm2(tau.get());

    // beta = sqrt(beta)
    // eta = eta_next = beta
    // delta = gamma = cos_prev = sin_prev = cos = sin = 0
    // q = r / beta
    // z = z / beta
    // p = p_prev = q_prev = v = 0
    exec->run(minres::make_initialize(
        r.get(), z.get(), p.get(), p_prev.get(), q.get(), q_prev.get(), v.get(),
        beta.get(), gamma.get(), delta.get(), cos_prev.get(), cos.get(),
        sin_prev.get(), sin.get(), eta_next.get(), eta.get(), &stop_status));

    int iter = -1;
    /* Memory movement summary:
     * 27n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x axpy            3n
     * 1x step 1 (axpys)  16n
     */
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, nullptr, dense_x, tau.get(), nullptr);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(nullptr)
                .residual_norm(tau.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // Lanzcos (partial) update
        //
        // v = A * z - beta * q_prev
        // alpha = <v, z>
        // v = v - alpha * q
        // z_tilde = M * v
        // beta = <v, z_tilde>
        //
        // p and z get updated in step_1
        system_matrix_->apply(one_op.get(), z.get(), neg_one_op.get(), v.get());
        v->compute_conj_dot(z.get(), alpha.get());
        v->sub_scaled(alpha.get(), q.get());
        get_preconditioner()->apply(v.get(), z_tilde.get());
        v->compute_conj_dot(z_tilde.get(), beta.get());

        // step 1:
        // finish Lanzcos part 1
        // beta = sqrt(beta)
        // q_prev = v
        // q_tmp = q
        // q = v / beta
        // v = q_tmp * beta
        //
        // apply two previous givens rot to new column
        // delta = sin_prev * gamma  // 0 if iter = 0, 1
        // tmp_d = gamma
        // tmp_a = alpha
        // gamma = cos_prev * cos * tmp_d + sin * tmp_a  // 0 if iter = 0
        // alpha = -conj(sin) * cos_prev * tmp_d + cos * tmp_a
        //
        // compute new givens rot
        // sin_prev = sin
        // cos_prev = cos
        // cos, sin = givens_rot(alpha, beta)
        //
        // apply new givens rot to T and eta
        // tau = abs(sin) * tau
        // eta = eta_next
        // eta_next = -conj(sin) * eta
        // alpha = cos * alpha + sin * beta
        //
        // update search direction and solution
        // swap(p, p_prev)
        // p = (z - gamma * p_prev - delta * p) / alpha
        // x = x + cos * eta * p
        //
        // finish Lanzcos part 2
        // z = z_tilde / beta
        // gamma = beta
        exec->run(minres::make_step_1(
            dense_x, p.get(), p_prev.get(), z.get(), z_tilde.get(), q.get(),
            q_prev.get(), v.get(), alpha.get(), beta.get(), gamma.get(),
            delta.get(), cos_prev.get(), cos.get(), sin_prev.get(), sin.get(),
            eta.get(), eta_next.get(), tau.get(), &stop_status));
    }
}


template <typename ValueType>
void Minres<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                   const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_MINRES(_type) class Minres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES);


}  // namespace solver
}  // namespace gko
