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


template <typename LocalIndexType, typename GlobalIndexType>
void map_global_cols_to_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const GlobalIndexType* input_col_idxs, size_type num_input,
    size_type num_offdiag_elements, size_type num_offdiag_cols,
    comm_index_type local_part, const size_type* row_range_ids,
    const size_type* col_range_ids, const comm_index_type* row_part_ids,
    const comm_index_type* col_part_ids, const LocalIndexType* permutation,
    GlobalIndexType* offdiag_global_col_idxs, LocalIndexType* offdiag_col_idxs)
{
    auto map_to_unique_col =
        [permutation] __host__ __device__(const size_type lower_bound) {
            return permutation[lower_bound];
        };

    auto range_id_begin = thrust::make_zip_iterator(
        thrust::make_tuple(row_range_ids, col_range_ids));
    Array<GlobalIndexType> tmp_offdiag_cols{exec, num_offdiag_elements};
    thrust::copy_if(
        thrust::device, input_col_idxs, input_col_idxs + num_input,
        range_id_begin, tmp_offdiag_cols.get_data(),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part != local_part;
        });
    Array<size_type> lower_bounds{exec, num_offdiag_elements};
    thrust::lower_bound(thrust::device, offdiag_global_col_idxs,
                        offdiag_global_col_idxs + num_offdiag_cols,
                        tmp_offdiag_cols.get_data(),
                        tmp_offdiag_cols.get_data() + num_offdiag_elements,
                        lower_bounds.get_data());
    thrust::transform(
        thrust::device, lower_bounds.get_data(),
        lower_bounds.get_data() + num_offdiag_elements, offdiag_col_idxs,
        [map_to_unique_col] __host__ __device__(const GlobalIndexType col) {
            return map_to_unique_col(col);
        });
}

#define GKO_DECLARE_MAP_GLOBAL_COLS_TO_LOCAL(_local_index_t, _global_index_t) \
    void map_global_cols_to_local(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const _global_index_t* input_col_idxs, size_type num_input,           \
        size_type num_offdiag_elements, size_type num_offdiag_cols,           \
        comm_index_type local_part, const size_type* row_range_ids,           \
        const size_type* col_range_ids, const comm_index_type* row_part_ids,  \
        const comm_index_type* col_part_ids,                                  \
        const _local_index_t* permutation,                                    \
        _global_index_t* offdiag_global_col_idxs,                             \
        _local_index_t* offdiag_col_idxs)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAP_GLOBAL_COLS_TO_LOCAL);
