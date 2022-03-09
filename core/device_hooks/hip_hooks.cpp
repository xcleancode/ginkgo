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

#include <memory>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory_space.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_hip_version() noexcept
{
    // We just return 1.1.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 1, 0, "not compiled"};
}


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode, int num_additional_handles)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master), device_reset, alloc_mode,
                        num_additional_handles));
}


HipMemorySpace::HipMemorySpace(int device_id) : device_id_(device_id) {}

HipAsyncHandle::HipAsyncHandle(create_type c_type) {}

void HipAsyncHandle::HipAsyncHandle::get_result() GKO_NOT_IMPLEMENTED;

void HipAsyncHandle::HipAsyncHandle::wait() GKO_NOT_IMPLEMENTED;

void HipAsyncHandle::HipAsyncHandle::wait_for(
    const std::chrono::duration<int>& time) GKO_NOT_IMPLEMENTED;

void HipAsyncHandle::HipAsyncHandle::wait_until(
    const std::chrono::time_point<std::chrono::steady_clock>& time)
    GKO_NOT_IMPLEMENTED;


void HipExecutor::populate_exec_info(const MachineTopology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<MemorySpace> memory_space,
    std::shared_ptr<Executor> master, bool device_reset,
    int num_additional_handles)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, memory_space, std::move(master),
                        device_reset, num_additional_handles));
}

std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const HipMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);

std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const HipMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


void HipMemorySpace::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* HipMemorySpace::raw_alloc(size_type num_bytes) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const HostMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const ReferenceMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const CudaMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const CudaUVMSpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const DpcppMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const HipMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::synchronize() const GKO_NOT_COMPILED(hip);


void HipMemorySpace::synchronize() const GKO_NOT_COMPILED(hip);


void HipExecutor::run(const Operation& op) const
{
    op.run(
        std::static_pointer_cast<const HipExecutor>(this->shared_from_this()));
}


std::shared_ptr<AsyncHandle> HipExecutor::run(
    const AsyncOperation& op, std::shared_ptr<AsyncHandle> handle) const
{
    return op.run(
        std::static_pointer_cast<const HipExecutor>(this->shared_from_this()),
        handle);
}


std::string HipError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipblasError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HiprandError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipsparseError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipfftError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


int HipExecutor::get_num_devices() { return 0; }


int HipMemorySpace::get_num_devices() { return 0; }


void HipExecutor::set_gpu_property() {}


void HipExecutor::init_handles() {}


}  // namespace gko


#define GKO_HOOK_MODULE hip
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
