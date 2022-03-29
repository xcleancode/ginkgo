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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void copy_local_inner(
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    comm_index_type local_part, const size_type* row_range_ids,
    const size_type* col_range_ids, const comm_index_type* row_part_ids,
    const GlobalIndexType* row_range_bounds,
    const LocalIndexType* row_range_starting_indices,
    const comm_index_type* col_part_ids,
    const GlobalIndexType* col_range_bounds,
    const LocalIndexType* col_range_starting_indices,
    LocalIndexType* out_row_idxs, LocalIndexType* out_col_idxs,
    ValueType* out_values)
{
    auto map_to_local_row =
        [row_range_bounds, row_range_starting_indices] __host__ __device__(
            const GlobalIndexType row, const size_type range_id) {
            return static_cast<LocalIndexType>(row -
                                               row_range_bounds[range_id]) +
                   row_range_starting_indices[range_id];
        };
    auto map_to_local_col =
        [col_range_bounds, col_range_starting_indices] __host__ __device__(
            const GlobalIndexType col, const size_type range_id) {
            return static_cast<LocalIndexType>(col -
                                               col_range_bounds[range_id]) +
                   col_range_starting_indices[range_id];
        };

    using input_type = thrust::tuple<GlobalIndexType, GlobalIndexType,
                                     ValueType, size_type, size_type>;
    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values(), row_range_ids, col_range_ids));
    auto range_id_begin = thrust::make_zip_iterator(
        thrust::make_tuple(row_range_ids, col_range_ids));
    auto local_diag_it = thrust::make_transform_iterator(
        input_begin, [map_to_local_row, map_to_local_col] __host__ __device__(
                         const input_type& output) {
            auto local_row = map_to_local_row(thrust::get<0>(output),
                                              thrust::get<3>(output));
            auto local_col = map_to_local_col(thrust::get<1>(output),
                                              thrust::get<4>(output));
            return thrust::make_tuple(local_row, local_col,
                                      thrust::get<2>(output));
        });
    thrust::copy_if(
        thrust::device, local_diag_it, local_diag_it + input.get_num_elems(),
        range_id_begin,
        thrust::make_zip_iterator(
            thrust::make_tuple(out_row_idxs, out_col_idxs, out_values)),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part == local_part;
        });
}

#define GKO_DECLARE_COPY_LOCAL_INNER(_value_t, _local_index_t,               \
                                     _global_index_t)                        \
    void copy_local_inner(                                                   \
        const device_matrix_data<_value_t, _global_index_t>& input,          \
        comm_index_type local_part, const size_type* row_range_ids,          \
        const size_type* col_range_ids, const comm_index_type* row_part_ids, \
        const _global_index_t* row_range_bounds,                             \
        const _local_index_t* row_range_starting_indices,                    \
        const comm_index_type* col_part_ids,                                 \
        const _global_index_t* col_range_bounds,                             \
        const _local_index_t* col_range_starting_indices,                    \
        _local_index_t* out_row_idxs, _local_index_t* out_col_idxs,          \
        _value_t* out_values)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_COPY_LOCAL_INNER);
