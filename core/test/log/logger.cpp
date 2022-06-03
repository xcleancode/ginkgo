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

#include <ginkgo/core/log/logger.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/log/record.hpp>
#include <ginkgo/core/log/stream.hpp>


namespace {


constexpr int num_iters = 10;


struct DummyLoggedClass : gko::log::EnableLogging<DummyLoggedClass> {
    int get_num_loggers() { return loggers_.size(); }

    void apply()
    {
        this->log<gko::log::Logger::iteration_complete>(
            nullptr, num_iters, nullptr, nullptr, nullptr);
    }
};


TEST(DummyLogged, CanAddLogger)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c;

    c.add_logger(gko::log::Convergence<>::create(
        exec, gko::log::Logger::all_events_mask));

    ASSERT_EQ(c.get_num_loggers(), 1);
}


TEST(DummyLogged, CanAddMultipleLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c;

    c.add_logger(gko::log::Convergence<>::create(
        exec, gko::log::Logger::all_events_mask));
    c.add_logger(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask, std::cout));

    ASSERT_EQ(c.get_num_loggers(), 2);
}


TEST(DummyLogged, CanAccessLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c;

    auto logger1 = gko::share(
        gko::log::Record::create(exec, gko::log::Logger::all_events_mask));
    auto logger2 = gko::share(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask, std::cout));

    c.add_logger(logger1);
    c.add_logger(logger2);

    ASSERT_EQ(c.get_loggers()[0], logger1);
    ASSERT_EQ(c.get_loggers()[1], logger2);
    ASSERT_EQ(c.get_num_loggers(), 2);
}


TEST(DummyLogged, CanClearLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c;
    c.add_logger(
        gko::log::Record::create(exec, gko::log::Logger::all_events_mask));
    c.add_logger(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask, std::cout));

    c.clear_loggers();

    ASSERT_EQ(c.get_num_loggers(), 0);
}


TEST(DummyLogged, CanRemoveLogger)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c;
    auto r = gko::share(gko::log::Convergence<>::create(
        exec, gko::log::Logger::all_events_mask));
    c.add_logger(r);
    c.add_logger(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask, std::cout));

    c.remove_logger(gko::lend(r));

    ASSERT_EQ(c.get_num_loggers(), 1);
}

struct DummyLogger : gko::log::Logger {
    using Logger = gko::log::Logger;

    explicit DummyLogger(
        std::shared_ptr<const gko::Executor> exec,
        const mask_type& enabled_events = Logger::all_events_mask)
        : Logger(std::move(exec), enabled_events)
    {}

    void on_iteration_complete(
        const gko::LinOp* solver, const gko::size_type& num_iterations,
        const gko::LinOp* residual, const gko::LinOp* solution = nullptr,
        const gko::LinOp* residual_norm = nullptr) const override
    {
        this->num_iterations_ = num_iterations;
    }

    mutable gko::size_type num_iterations_{};
};


TEST(DummyLogged, CanLogEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::shared_ptr<DummyLogger>(
        new DummyLogger(exec, gko::log::Logger::iteration_complete_mask));
    DummyLoggedClass c;
    c.add_logger(l);

    c.apply();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


struct DummyMpiLogger : gko::log::Logger {
    using Logger = gko::log::Logger;

    mutable int blocking_count = 0;
    mutable int non_blocking_count = 0;

    explicit DummyMpiLogger(
        std::shared_ptr<const gko::Executor> exec,
        const mask_type& enabled_events = Logger::mpi_events_mask)
        : Logger(std::move(exec), enabled_events)
    {}


    void on_mpi_point_to_point_communication_started(
        bool is_blocking, const char* name, const MPI_Comm* comm,
        const gko::uintptr& loc, int size, MPI_Datatype type, int source_rank,
        int destination_rank, int tag, const MPI_Request* req) const override
    {
        if (is_blocking) {
            blocking_count++;
        } else {
            non_blocking_count++;
        }
    }
};


TEST(DummyMpiLogger, CanLogBlockingMpiEvents)
{
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(gko::ReferenceExecutor::create());

    l->template on<Logger::blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);

    ASSERT_EQ(l->blocking_count, 1);
}


TEST(DummyMpiLogger, CanLogNonBlockingMpiEvents)
{
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(gko::ReferenceExecutor::create());

    l->template on<
        Logger::non_blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);

    ASSERT_EQ(l->non_blocking_count, 1);
}


TEST(DummyMpiLogger, CanExclusivlyLogBlockingMpiEvents)
{
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(
        gko::ReferenceExecutor::create(),
        Logger::mpi_events_mask & ~Logger::mpi_non_blocking_communication_mask);

    l->template on<Logger::blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);
    l->template on<
        Logger::non_blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);

    ASSERT_EQ(l->blocking_count, 1);
    ASSERT_EQ(l->non_blocking_count, 0);
}


TEST(DummyMpiLogger, CanExclusivlyLogNonBlockingMpiEvents)
{
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(
        gko::ReferenceExecutor::create(),
        Logger::mpi_events_mask & ~Logger::mpi_blocking_communication_mask);

    l->template on<Logger::blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);
    l->template on<
        Logger::non_blocking_mpi_point_to_point_communication_started>(
        "", nullptr, 0, 0, MPI_DATATYPE_NULL, 0, 0, 0, nullptr);

    ASSERT_EQ(l->blocking_count, 0);
    ASSERT_EQ(l->non_blocking_count, 1);
}
}  // namespace
