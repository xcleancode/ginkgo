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

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


constexpr int default_block_size = 256;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/components/reduction.hpp.inc"
#include "common/cuda_hip/log/batch_logger.hpp.inc"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp.inc"
// TODO: remove batch dense include
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


int get_shared_memory_per_sm(std::shared_ptr<const CudaExecutor> exec)
{
    // cudaDeviceProp prop;
    // // find the initial amount of shared memory
    // auto err = cudaGetDeviceProperties(&prop, exec->get_device_id());
    // assert(err == cudaSuccess);
    // const size_t shmem_per_blk = prop.sharedMemPerBlock;
    int shmem_per_sm = 0;
    cudaDeviceGetAttribute(&shmem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           exec->get_device_id());
    return shmem_per_sm;
}


template <typename T>
using BatchBicgstabOptions =
    gko::kernels::batch_bicgstab::BatchBicgstabOptions<T>;

#define BATCH_BICGSTAB_KERNEL_LAUNCH(_stoppertype, _prectype)           \
    apply_kernel<stop::_stoppertype<ValueType>>                         \
        <<<nbatch, default_block_size, shared_size>>>(                  \
            shared_gap, sconf, opts.max_its, opts.residual_tol, logger, \
            _prectype(), a, b.values, x.values, workspace.get_data())

template <typename PrecType, typename BatchMatrixType, typename LogType,
          typename ValueType>
static void apply_impl(
    std::shared_ptr<const CudaExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> opts, LogType logger,
    const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const int shared_gap = ((a.num_rows - 1) / 8 + 1) * 8;
    static_assert(default_block_size >= 2 * config::warp_size,
                  "Need at least two warps!");

    const size_t shmem_per_sm = get_shared_memory_per_sm(exec);

    const size_t prec_size =
        PrecType::dynamic_work_size(shared_gap, a.num_nnz) * sizeof(ValueType);
    const auto sconf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PrecType,
                                                             ValueType>(
            shmem_per_sm, shared_gap, a.num_nnz, b.num_rhs);
    const size_t shared_size = sconf.n_shared * shared_gap * sizeof(ValueType) +
                               (sconf.prec_shared ? prec_size : 0);
    auto workspace = gko::Array<ValueType>(
        exec, sconf.gmem_stride_bytes * nbatch / sizeof(ValueType));

    printf(" Shared vectors = %d, global vectors = %d.\n", sconf.n_shared,
           sconf.n_global);
    if (sconf.prec_shared) {
        printf(" Preconditioner in shared\n");
    }
    printf(" Global size for each batch entry = %d.\n",
           sconf.gmem_stride_bytes);

    if (opts.tol_type == gko::stop::batch::ToleranceType::absolute) {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleAbsResidual, PrecType);
    } else {
        BATCH_BICGSTAB_KERNEL_LAUNCH(SimpleRelResidual, PrecType);
    }
    GKO_CUDA_LAST_IF_ERROR_THROW;
}


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchBicgstabOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;

    batch_log::SimpleFinalLogger<remove_complex<ValueType>> logger(
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<cu_value_type> x_b =
        get_batch_struct(x);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
        auto m_b = get_batch_struct(amat);
        auto b_b = get_batch_struct(b);
        if (opts.preconditioner == gko::preconditioner::batch::type::none) {
            apply_impl<BatchIdentity<cu_value_type>>(exec, opts, logger, m_b,
                                                     b_b, x_b);
        } else if (opts.preconditioner ==
                   gko::preconditioner::batch::type::jacobi) {
            apply_impl<BatchJacobi<cu_value_type>>(exec, opts, logger, m_b, b_b,
                                                   x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
