/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_KERNELS_KERNEL_LAUNCH_HPP_
#error "This file can only be used from inside ginkgo/kernels/kernel_launch.hpp"
#endif


namespace gko {
namespace kernels {
namespace reference {


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const ReferenceExecutor> exec,
                KernelFunction fn, size_type size, KernelArgs&&... args)
{
    for (int64 i = 0; i < static_cast<int64>(size); i++) {
        fn(i, map_to_device(args)...);
    }
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const ReferenceExecutor> exec,
                KernelFunction fn, dim<2> size, KernelArgs&&... args)
{
    for (int64 row = 0; row < static_cast<int64>(size[0]); row++) {
        for (int64 col = 0; col < static_cast<int64>(size[1]); col++) {
            fn(row, col, map_to_device(args)...);
        }
    }
}


}  // namespace reference
}  // namespace kernels
}  // namespace gko
