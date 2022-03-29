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
void compute_gather_idxs_and_sizes(
    const GlobalIndexType* col_range_bounds,
    const LocalIndexType* col_range_starting_indices,
    const GlobalIndexType* local_to_global_ghost,
    const comm_index_type* offdiag_col_part_ids,
    const size_type* offdiag_col_range_ids, size_type num_offdiag_cols,
    size_type num_parts, LocalIndexType* local_gather_idxs,
    comm_index_type* recv_sizes)
{
    auto map_to_local_col =
        [col_range_bounds, col_range_starting_indices] __host__ __device__(
            const GlobalIndexType col, const size_type range_id) {
            return static_cast<LocalIndexType>(col -
                                               col_range_bounds[range_id]) +
                   col_range_starting_indices[range_id];
        };

    auto transform_it = thrust::make_zip_iterator(
        thrust::make_tuple(local_to_global_ghost, offdiag_col_range_ids));
    thrust::transform(
        thrust::device, transform_it, transform_it + num_offdiag_cols,
        local_gather_idxs,
        [map_to_local_col] __host__ __device__(
            const thrust::tuple<GlobalIndexType, size_type>& tuple) {
            return map_to_local_col(thrust::get<0>(tuple),
                                    thrust::get<1>(tuple));
        });
    thrust::fill_n(thrust::device, recv_sizes, num_parts, 0);
    thrust::for_each_n(thrust::device, offdiag_col_part_ids, num_offdiag_cols,
                       [recv_sizes] __device__(const size_type part) {
                           atomic_add(recv_sizes + part, 1);
                       });
}

#define GKO_DECLARE_COMPUTE_GATHER_IDXS_AND_SIZES(_local_index_t,           \
                                                  _global_index_t)          \
    void compute_gather_idxs_and_sizes(                                     \
        const _global_index_t* col_range_bounds,                            \
        const _local_index_t* col_range_starting_indices,                   \
        const _global_index_t* local_to_global_ghost,                       \
        const comm_index_type* offdiag_col_part_ids,                        \
        const size_type* offdiag_col_range_ids, size_type num_offdiag_cols, \
        size_type num_parts, _local_index_t* local_gather_idxs,             \
        comm_index_type* recv_sizes)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_GATHER_IDXS_AND_SIZES);
