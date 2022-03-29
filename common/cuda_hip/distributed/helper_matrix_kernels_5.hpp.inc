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

using gko::distributed::comm_index_type;


template <typename GlobalIndexType>
void sort_offdiag_cols_and_rid_by_pid_and_idx(
    comm_index_type* offdiag_col_part_ids,
    GlobalIndexType* offdiag_global_col_idxs, size_type* offdiag_col_range_ids,
    size_type num_offdiag_elements)
{
    auto combined_it = thrust::make_zip_iterator(
        thrust::make_tuple(offdiag_col_part_ids, offdiag_global_col_idxs));
    thrust::stable_sort_by_key(thrust::device, combined_it,
                               combined_it + num_offdiag_elements,
                               offdiag_col_range_ids);
}

#define GKO_DECLARE_SORT_OFFDIAG_COLS_AND_RID_BY_PID_AND_IDX(_type)            \
    void sort_offdiag_cols_and_rid_by_pid_and_idx(                             \
        comm_index_type* offdiag_col_part_ids, _type* offdiag_global_col_idxs, \
        size_type* offdiag_col_range_ids, size_type num_offdiag_elements)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SORT_OFFDIAG_COLS_AND_RID_BY_PID_AND_IDX);


template <typename LocalIndexType, typename GlobalIndexType>
void sort_offdiag_cols_by_idx(comm_index_type* offdiag_col_part_ids,
                              GlobalIndexType* offdiag_global_col_idxs,
                              LocalIndexType* permutation,
                              size_type num_offdiag_cols)
{
    thrust::sequence(thrust::device, permutation,
                     permutation + num_offdiag_cols);
    thrust::sort_by_key(offdiag_global_col_idxs,
                        offdiag_global_col_idxs + num_offdiag_cols,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            offdiag_col_part_ids, permutation)));
}

#define GKO_DECLARE_SORT_OFFDIAG_COLS_BY_IDX(_local_index_t, _global_index_t) \
    void sort_offdiag_cols_by_idx(comm_index_type* offdiag_col_part_ids,      \
                                  _global_index_t* offdiag_global_col_idxs,   \
                                  _local_index_t* permutation,                \
                                  size_type num_offdiag_cols)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SORT_OFFDIAG_COLS_BY_IDX);
