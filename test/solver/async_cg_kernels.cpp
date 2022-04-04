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

#include "core/solver/cg_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Cg : public ::testing::Test {
protected:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;
    using Mtx = gko::matrix::Dense<value_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;

    Cg() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::unique_ptr<CsrMtx> gen_mtx(gko::size_type num_rows,
                                    gko::size_type num_cols)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<CsrMtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = CsrMtx::create(ref, gko::dim<2>{num_rows, num_cols});
        result->copy_from(tmp_mtx.get());

        return result;
    }

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx.get());

        return result;
    }

    void initialize_data()
    {
        gko::size_type m = 597;
        gko::size_type n = 43;
        // all vectors need the same stride as b, except x
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        z = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        q = gen_mtx(m, n, n + 2);
        x = gen_mtx(m, n, n + 3);
        beta = gen_mtx(1, n, n);
        prev_rho = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
        // check correct handling for zero values
        beta->at(2) = 0.0;
        prev_rho->at(2) = 0.0;
        stop_status =
            std::make_unique<gko::Array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_z = gko::clone(exec, z);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_x = gko::clone(exec, x);
        d_beta = gko::clone(exec, beta);
        d_prev_rho = gko::clone(exec, prev_rho);
        d_rho = gko::clone(exec, rho);
        d_stop_status = std::make_unique<gko::Array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Cg, AsyncCgInitializeIsEquivalentToRef)
{
    initialize_data();

    auto hand1 = gko::kernels::reference::cg::initialize(
        ref, ref->get_handle_at(0), b.get(), r.get(), z.get(), p.get(), q.get(),
        prev_rho.get(), rho.get(), stop_status.get());
    auto hand2 = gko::kernels::EXEC_NAMESPACE::cg::initialize(
        exec, exec->get_handle_at(0), d_b.get(), d_r.get(), d_z.get(),
        d_p.get(), d_q.get(), d_prev_rho.get(), d_rho.get(),
        d_stop_status.get());

    hand1->wait();
    hand2->wait();

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Cg, AsyncCgStep1IsEquivalentToRef)
{
    initialize_data();

    auto hand1 = gko::kernels::reference::cg::step_1(
        ref, ref->get_handle_at(0), p.get(), z.get(), rho.get(), prev_rho.get(),
        stop_status.get());
    auto hand2 = gko::kernels::EXEC_NAMESPACE::cg::step_1(
        exec, exec->get_handle_at(0), d_p.get(), d_z.get(), d_rho.get(),
        d_prev_rho.get(), d_stop_status.get());

    hand1->wait();
    hand2->wait();

    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
}


TEST_F(Cg, AsyncCgStep2IsEquivalentToRef)
{
    initialize_data();

    auto hand1 = gko::kernels::reference::cg::step_2(
        ref, ref->get_default_exec_stream(), x.get(), r.get(), p.get(), q.get(),
        beta.get(), rho.get(), stop_status.get());
    auto hand2 = gko::kernels::EXEC_NAMESPACE::cg::step_2(
        exec, exec->get_default_exec_stream(), d_x.get(), d_r.get(), d_p.get(),
        d_q.get(), d_beta.get(), d_rho.get(), d_stop_status.get());

    hand1->wait();
    hand2->wait();

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
}


TEST_F(Cg, AsyncApplyIsEquivalentToRef)
{
    exec->set_default_exec_stream(exec->get_handle_at(0));
    auto dense_mtx = gen_mtx(20, 20, 20);
    gko::test::make_hpd(dense_mtx.get());
    auto mtx = share(CsrMtx::create(ref));
    mtx->copy_from(dense_mtx.get());
    auto x = gen_mtx(20, 1, 1);
    auto b = gen_mtx(20, 1, 1);
    auto d_mtx = gko::clone(exec, mtx);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto d_mtx2 = gko::clone(exec, mtx);
    auto d_x2 = gko::clone(exec, x);
    auto d_b2 = gko::clone(exec, b);
    auto d_mtx3 = gko::clone(exec, mtx);
    auto d_x3 = gko::clone(exec, x);
    auto d_b3 = gko::clone(exec, b);
    auto cg_factory =
        gko::solver::Cg<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(75u).on(ref),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(ref))
            .on(ref);
    auto d_cg_factory =
        gko::solver::Cg<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(75u).on(exec)
                // ,
                // gko::stop::ResidualNorm<value_type>::build()
                //     .with_reduction_factor(::r<value_type>::value)
                // .on(exec)
                )
            .on(exec);
    auto solver = cg_factory->generate(mtx);
    auto d_solver = d_cg_factory->generate(std::move(d_mtx));
    auto d_solver2 = d_cg_factory->generate(std::move(d_mtx2));
    auto d_solver3 = d_cg_factory->generate(std::move(d_mtx3));

    solver->apply(b.get(), x.get());
    auto hand2 = d_solver->apply(d_b.get(), d_x.get(), exec->get_handle_at(1));
    auto hand3 =
        d_solver2->apply(d_b2.get(), d_x2.get(), exec->get_handle_at(2));
    auto hand4 =
        d_solver3->apply(d_b3.get(), d_x3.get(), exec->get_handle_at(3));

    hand2->wait();
    hand3->wait();
    hand4->wait();
    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 1000);
    GKO_ASSERT_MTX_NEAR(d_x2, x, ::r<value_type>::value * 1000);
    GKO_ASSERT_MTX_NEAR(d_x3, x, ::r<value_type>::value * 1000);
}


}  // namespace
