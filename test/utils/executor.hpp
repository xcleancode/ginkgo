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

#ifndef GKO_TEST_UTILS_EXECUTOR_HPP_
#define GKO_TEST_UTILS_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <stdexcept>


#include <gtest/gtest.h>


#ifdef GKO_COMPILING_CUDA

#include "cuda/base/device.hpp"

class CudaEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::cuda::reset_device(0); }
};

testing::Environment* cuda_env =
    testing::AddGlobalTestEnvironment(new CudaEnvironment);

#endif


#ifdef GKO_COMPILING_HIP

#include "hip/base/device.hpp"

class HipEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::hip::reset_device(0); }
};

testing::Environment* hip_env =
    testing::AddGlobalTestEnvironment(new HipEnvironment);

#endif


#if GINKGO_COMMON_SINGLE_MODE
#define SKIP_IF_SINGLE_MODE GTEST_SKIP() << "Skip due to single mode"
#else
#define SKIP_IF_SINGLE_MODE                                                  \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor>,
                          std::shared_ptr<gko::ReferenceExecutor>& exec)
{
    exec = gko::ReferenceExecutor::create();
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor>,
                          std::shared_ptr<gko::OmpExecutor>& exec)
{
    exec = gko::OmpExecutor::create();
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::CudaExecutor>& exec,
                          CUstream_st* stream = nullptr)
{
    {
        if (gko::CudaExecutor::get_num_devices() == 0) {
            throw std::runtime_error{"No suitable CUDA devices"};
        }
        exec = gko::CudaExecutor::create(0, ref, false,
                                         gko::default_cuda_alloc_mode, stream);
    }
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::HipExecutor>& exec,
                          GKO_HIP_STREAM_STRUCT* stream = nullptr)
{
    if (gko::HipExecutor::get_num_devices() == 0) {
        throw std::runtime_error{"No suitable HIP devices"};
    }
    exec = gko::HipExecutor::create(0, ref, false, gko::default_hip_alloc_mode,
                                    stream);
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::DpcppExecutor>& exec)
{
    if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
        exec = gko::DpcppExecutor::create(0, ref, "gpu");
    } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
        exec = gko::DpcppExecutor::create(0, ref, "cpu");
    } else {
        throw std::runtime_error{"No suitable DPC++ devices"};
    }
}


class CommonTestFixture : public ::testing::Test {
public:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;

    CommonTestFixture() : ref{gko::ReferenceExecutor::create()}
    {
#if defined(GKO_TEST_NONDEFAULT_STREAM) && \
    (defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP))
        init_executor(ref, exec, stream.get());
#else
        init_executor(ref, exec);
#endif
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

#ifdef GKO_TEST_NONDEFAULT_STREAM
#ifdef GKO_COMPILING_CUDA
    gko::cuda_stream stream;
#endif
#ifdef GKO_COMPILING_HIP
    gko::hip_stream stream;
#endif
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
};


#endif  // GKO_TEST_UTILS_EXECUTOR_HPP_
