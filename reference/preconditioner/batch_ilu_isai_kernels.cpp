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

#include "core/preconditioner/batch_ilu_isai_kernels.hpp"


#include <algorithm>
#include <cassert>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
//#include "reference/preconditioner/batch_ilu_isai.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_ilu_isai {


template <typename ValueType, typename IndexType>
void apply_ilu_isai(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                    const matrix::BatchCsr<ValueType, IndexType>* const l_inv,
                    const matrix::BatchCsr<ValueType, IndexType>* const u_inv,
                    const preconditioner::batch_ilu_isai_apply apply_type,
                    const matrix::BatchDense<ValueType>* const r,
                    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_ilu_isai): change the code imported from
// preconditioner/batch_ilu if needed
//    const auto nbatch = sys_matrix->get_num_batch_entries();
//    const auto num_rows = static_cast<int>(sys_matrix->get_size().at(0)[0]);
//
//    const batch_csr::UniformBatch<const ValueType> factored_mat_batch =
//        gko::kernels::host::get_batch_struct(factored_matrix);
//    const auto rub = gko::kernels::host::get_batch_struct(r);
//    const auto zub = gko::kernels::host::get_batch_struct(z);
//
//    using prec_type = gko::kernels::host::batch_ilu_isai<ValueType>;
//    prec_type prec(factored_mat_batch, diag_locs);
//
//    const auto work_arr_size = prec_type::dynamic_work_size(
//        num_rows,
//        static_cast<int>(sys_matrix->get_num_stored_elements() / nbatch));
//    std::vector<ValueType> work(work_arr_size);
//
//    for (size_type batch_id = 0; batch_id < nbatch; batch_id++) {
//        const auto r_b = gko::batch::batch_entry(rub, batch_id);
//        const auto z_b = gko::batch::batch_entry(zub, batch_id);
//        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
//                      work.data());
//        prec.apply(r_b, z_b);
//    }
//}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_ISAI_APPLY_KERNEL);


}  // namespace batch_ilu_isai
}  // namespace reference
}  // namespace kernels
}  // namespace gko
