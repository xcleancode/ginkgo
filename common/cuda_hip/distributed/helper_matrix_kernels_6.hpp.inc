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
GlobalIndexType* make_offdiag_cols_unique(
    comm_index_type* offdiag_col_part_ids,
    GlobalIndexType* offdiag_global_col_idxs, size_type* offdiag_col_range_ids,
    size_type num_offdiag_elements)
{
    return thrust::get<0>(thrust::unique_by_key(
        thrust::device, offdiag_global_col_idxs,
        offdiag_global_col_idxs + num_offdiag_elements,
        thrust::make_zip_iterator(
            thrust::make_tuple(offdiag_col_part_ids, offdiag_col_range_ids))));
}

#define GKO_DECLARE_MAKE_OFFDIAG_COLS_UNIQUE(_type)                            \
    _type* make_offdiag_cols_unique(                                           \
        comm_index_type* offdiag_col_part_ids, _type* offdiag_global_col_idxs, \
        size_type* offdiag_col_range_ids, size_type num_offdiag_elements)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MAKE_OFFDIAG_COLS_UNIQUE);
