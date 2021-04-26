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

#include <hip/hip_runtime.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>

#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


constexpr int default_block_size = 128;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {

#include "common/components/uninitialized_array.hpp.inc"


#include "common/log/batch_logger.hpp.inc"
#include "common/matrix/batch_csr_kernels.hpp.inc"
#include "common/matrix/batch_dense_kernels.hpp.inc"
#include "common/preconditioner/batch_identity.hpp.inc"
#include "common/preconditioner/batch_jacobi.hpp.inc"
#include "common/stop/batch_criteria.hpp.inc"


#include "common/solver/batch_bicgstab_kernels.hpp.inc"


template <typename T>
using BatchBicgstabOptions =
    gko::kernels::batch_bicgstab::BatchBicgstabOptions<T>;

template <typename BatchMatrixType, typename LogType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const HipExecutor> exec,
    const BatchBicgstabOptions<remove_complex<ValueType>> opts, LogType logger,
    const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;

    if (opts.preconditioner == gko::preconditioner::batch::jacobi_str) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                apply_kernel<BatchJacobi<ValueType>,
                             stop::AbsAndRelResidualMaxIter<ValueType>>),
            dim3(nbatch), dim3(default_block_size), 0, 0, opts.max_its,
            opts.abs_residual_tol, opts.rel_residual_tol, opts.tol_type, logger,
            a, left, right, b, x);
    } else if (opts.preconditioner == gko::preconditioner::batch::none_str) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                apply_kernel<BatchIdentity<ValueType>,
                             stop::AbsAndRelResidualMaxIter<ValueType>>),
            dim3(nbatch), dim3(default_block_size), 0, 0, opts.max_its,
            opts.abs_residual_tol, opts.rel_residual_tol, opts.tol_type, logger,
            a, left, right, b, x);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const HipExecutor> exec,
           const BatchBicgstabOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           log::BatchLogData<ValueType> &logdata)
{
    using hip_value_type = hip_type<ValueType>;

    // For now, FinalLogger is the only one available
    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        static_cast<int>(b->get_size().at(0)[1]), opts.max_its,
        logdata.res_norms->get_values(), logdata.iter_counts.get_data());


    const gko::batch_dense::UniformBatch<const hip_value_type> left_sb =
        maybe_null_batch_struct(left_scale);
    const gko::batch_dense::UniformBatch<const hip_value_type> right_sb =
        maybe_null_batch_struct(right_scale);
    const auto to_scale = left_sb.values || right_sb.values;
    if (to_scale) {
        if (!left_sb.values || !right_sb.values) {
            // one-sided scaling not implemented
            GKO_NOT_IMPLEMENTED;
        }
    }


    const gko::batch_dense::UniformBatch<hip_value_type> x_b =
        get_batch_struct(x);

    if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        const gko::batch_csr::UniformBatch<hip_value_type> m_b =
            get_batch_struct(const_cast<matrix::BatchCsr<ValueType> *>(amat));

        const gko::batch_dense::UniformBatch<hip_value_type> b_b =
            get_batch_struct(const_cast<matrix::BatchDense<ValueType> *>(b));

        apply_impl(exec, opts, logger, m_b, left_sb, right_sb, b_b, x_b);
    } else {
        GKO_NOT_SUPPORTED(a);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace hip
}  // namespace kernels
}  // namespace gko
