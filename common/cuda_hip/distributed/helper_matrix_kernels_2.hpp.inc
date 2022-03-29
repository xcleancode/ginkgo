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


thrust::tuple<size_type, size_type> compute_local_sizes(
    const size_type* row_range_ids, const size_type* col_range_ids,
    size_type num_input, comm_index_type local_part,
    const comm_index_type* row_part_ids, const comm_index_type* col_part_ids)
{
    auto range_id_begin = thrust::make_zip_iterator(
        thrust::make_tuple(row_range_ids, col_range_ids));
    return thrust::transform_reduce(
        thrust::device, range_id_begin, range_id_begin + num_input,
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            bool is_inner_entry =
                row_part == local_part && col_part == local_part;
            bool is_ghost_entry =
                row_part == local_part && col_part != local_part;
            return thrust::make_tuple(
                is_inner_entry ? size_type{1} : size_type{0},
                is_ghost_entry ? size_type{1} : size_type{0});
        },
        thrust::make_tuple(size_type{}, size_type{}),
        [] __host__ __device__(const thrust::tuple<size_type, size_type>& a,
                               const thrust::tuple<size_type, size_type>& b) {
            return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                      thrust::get<1>(a) + thrust::get<1>(b));
        });
}
