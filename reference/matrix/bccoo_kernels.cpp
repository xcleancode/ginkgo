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

#include "core/matrix/bccoo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/unaligned_access.hpp"
#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


void get_default_block_size(std::shared_ptr<const DefaultExecutor> exec,
                            size_type& block_size)
{
    block_size = 10;
    //	block_size = 1;
    block_size = 2;
    //	block_size = 3;
    //	block_size = 4;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
//        c->at(i) = zero<ValueType>();
//    }
//    spmv2(exec, a, b, c);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto beta_val = beta->at(0, 0);
//    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
//        c->at(i) *= beta_val;
//    }
//    advanced_spmv2(exec, alpha, a, b, c);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b,
           matrix::Dense<ValueType>* c)  // GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto bccoo_val = a->get_const_values();
//    auto bccoo_col = a->get_const_col_idxs();
//    auto bccoo_row = a->get_const_row_idxs();
//    auto num_cols = b->get_size()[1];
//    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
//        for (size_type j = 0; j < num_cols; j++) {
//            c->at(bccoo_row[i], j) += bccoo_val[i] * b->at(bccoo_col[i], j);
//        }
//    }
//}
{
    auto* rows_data = a->get_const_rows();
    auto* offsets_data = a->get_const_offsets();
    auto* chunk_data = a->get_const_chunk();
    auto num_stored_elements = a->get_num_stored_elements();
    auto block_size = a->get_block_size();
    auto num_cols = b->get_size()[1];
    //    auto exec = this->get_executor();
    //    auto exec_master = exec->get_master();

    //    std::unique_ptr<const LinOp> op{};
    //    const Bccoo *tmp{};
    //    if (this->get_executor()->get_master() != this->get_executor()) {
    //    if (exec_master != exec) {
    //        //        op = this->clone(this->get_executor()->get_master());
    //        op = this->clone(exec_master);
    //        tmp = static_cast<const Bccoo *>(op.get());
    //    } else {
    //        tmp = this;
    //    }

    // Computation of chunk
    size_type k = 0, blk = 0, col = 0, row = 0, shf = 0;
    ValueType val;
    for (size_type i = 0; i < num_stored_elements; i++) {
        if (k == 0) {
            row = rows_data[blk];
            col = 0;
            shf = offsets_data[blk];
        }
        //				uint8 ind =
        // a->get_value_chunk<gko::uint8>(chunk_data, shf);
        // uint8_t ind = a->get_value_chunk<uint8_t>(chunk_data, shf);
        // uint8_t ind = a->get_value_chunk(chunk_data, shf);
        // uint8 ind =
        // a->get_value_chunk<uint8>(chunk_data, shf);
        // uint8 ind = * (chunk_data
        //+ shf);
        uint8 ind = (chunk_data[shf]);
        while (ind == 0xFF) {
            row++;
            shf++;
            col = 0;
            ind = chunk_data[shf];
        }
        if (ind < 0xFD) {
            col += ind;
            shf++;
        } else if (ind == 0xFD) {
            shf++;
            col += get_value_chunk<uint16>(chunk_data, shf);
            //            col += *(uint16 *)(chunk_data + shf);
            shf += 2;
        } else {
            shf++;
            //            col += get_value_chunk<uint32>(chunk_data, shf);
            col += *(uint32*)(chunk_data + shf);
            shf += 4;
        }
        //				val =
        // get_value_chunk<ValueType>(chunk_data, shf);
        val = *(ValueType*)(chunk_data + shf);
        //        data.nonzeros.emplace_back(row, col, val);
        for (size_type j = 0; j < num_cols; j++) {
            c->at(row, j) += val * b->at(col, j);
        }
        shf += sizeof(ValueType);
        if (++k == block_size) {
            k = 0;
            blk++;
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto bccoo_val = a->get_const_values();
//    auto bccoo_col = a->get_const_col_idxs();
//    auto bccoo_row = a->get_const_row_idxs();
//    auto alpha_val = alpha->at(0, 0);
//    auto num_cols = b->get_size()[1];
//    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
//        for (size_type j = 0; j < num_cols; j++) {
//            c->at(bccoo_row[i], j) +=
//                alpha_val * bccoo_val[i] * b->at(bccoo_col[i], j);
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const ReferenceExecutor> exec,
                              const IndexType* idxs, size_type num_nonzeros,
                              IndexType* ptrs,
                              size_type length) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    convert_idxs_to_ptrs(idxs, num_nonzeros, ptrs, length);
//}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto num_rows = result->get_size()[0];
//
//    auto row_ptrs = result->get_row_ptrs();
//    const auto nnz = result->get_num_stored_elements();
//
//    const auto source_row_idxs = source->get_const_row_idxs();
//
//    convert_row_idxs_to_ptrs(exec, source_row_idxs, nnz, row_ptrs,
//                             num_rows + 1);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto bccoo_val = source->get_const_values();
//    auto bccoo_col = source->get_const_col_idxs();
//    auto bccoo_row = source->get_const_row_idxs();
//    auto num_rows = result->get_size()[0];
//    auto num_cols = result->get_size()[1];
//    for (size_type row = 0; row < num_rows; row++) {
//        for (size_type col = 0; col < num_cols; col++) {
//            result->at(row, col) = zero<ValueType>();
//        }
//    }
//    for (size_type i = 0; i < source->get_num_stored_elements(); i++) {
//        result->at(bccoo_row[i], bccoo_col[i]) += bccoo_val[i];
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    const auto row_idxs = orig->get_const_row_idxs();
//    const auto col_idxs = orig->get_const_col_idxs();
//    const auto values = orig->get_const_values();
//    const auto diag_size = diag->get_size()[0];
//    const auto nnz = orig->get_num_stored_elements();
//    auto diag_values = diag->get_values();
//
//    for (size_type idx = 0; idx < nnz; idx++) {
//        if (row_idxs[idx] == col_idxs[idx]) {
//            diag_values[row_idxs[idx]] = values[idx];
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


}  // namespace bccoo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
