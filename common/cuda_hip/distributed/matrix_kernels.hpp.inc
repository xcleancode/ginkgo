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


template <typename GlobalIndexType>
void compute_range_ids(const GlobalIndexType* range_bounds,
                       size_type num_ranges, const GlobalIndexType* input,
                       size_type num_input, size_type* result);

thrust::tuple<size_type, size_type> compute_local_sizes(
    const size_type* row_range_ids, const size_type* col_range_ids,
    size_type num_input, comm_index_type local_part,
    const comm_index_type* row_part_ids, const comm_index_type* col_part_ids);

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
    ValueType* out_values);

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void copy_local_ghost(
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    comm_index_type local_part, const size_type* row_range_ids,
    const size_type* col_range_ids, const comm_index_type* row_part_ids,
    const GlobalIndexType* row_range_bounds,
    const LocalIndexType* row_range_starting_indices,
    const comm_index_type* col_part_ids,
    const GlobalIndexType* col_range_bounds,
    const LocalIndexType* col_range_starting_indices,
    LocalIndexType* out_row_idxs, GlobalIndexType* out_col_idxs,
    ValueType* out_values, comm_index_type* out_col_part_ids,
    size_type* out_col_range_ids);

template <typename GlobalIndexType>
void sort_offdiag_cols_and_rid_by_pid_and_idx(
    comm_index_type* offdiag_col_part_ids,
    GlobalIndexType* offdiag_global_col_idxs, size_type* offdiag_col_range_ids,
    size_type num_offdiag_elements);

template <typename LocalIndexType, typename GlobalIndexType>
void sort_offdiag_cols_by_idx(comm_index_type* offdiag_col_part_ids,
                              GlobalIndexType* offdiag_global_col_idxs,
                              LocalIndexType* permutation,
                              size_type num_offdiag_cols);

template <typename GlobalIndexType>
GlobalIndexType* make_offdiag_cols_unique(
    comm_index_type* offdiag_col_part_ids,
    GlobalIndexType* offdiag_global_col_idxs, size_type* offdiag_col_range_ids,
    size_type num_offdiag_elements);


template <typename LocalIndexType, typename GlobalIndexType>
void map_global_cols_to_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const GlobalIndexType* input_col_idxs, size_type num_input,
    size_type num_offdiag_elements, size_type num_offdiag_cols,
    comm_index_type local_part, const size_type* row_range_ids,
    const size_type* col_range_ids, const comm_index_type* row_part_ids,
    const comm_index_type* col_part_ids, const LocalIndexType* permutation,
    GlobalIndexType* offdiag_global_col_idxs, LocalIndexType* offdiag_col_idxs);

template <typename LocalIndexType, typename GlobalIndexType>
void compute_gather_idxs_and_sizes(
    const GlobalIndexType* col_range_bounds,
    const LocalIndexType* col_range_starting_indices,
    const GlobalIndexType* local_to_global_ghost,
    const comm_index_type* offdiag_col_part_ids,
    const size_type* offdiag_col_range_ids, size_type num_offdiag_cols,
    size_type num_parts, LocalIndexType* local_gather_idxs,
    comm_index_type* recv_sizes);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part,
    device_matrix_data<ValueType, LocalIndexType>& diag_data,
    device_matrix_data<ValueType, LocalIndexType>& offdiag_data,
    Array<LocalIndexType>& local_gather_idxs, comm_index_type* recv_sizes,
    Array<GlobalIndexType>& local_to_global_ghost)
{
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = static_cast<size_type>(row_partition->get_num_parts());
    const auto* row_range_bounds = row_partition->get_range_bounds();
    const auto* col_range_bounds = col_partition->get_range_bounds();
    const auto* row_range_starting_indices =
        row_partition->get_range_starting_indices();
    const auto* col_range_starting_indices =
        col_partition->get_range_starting_indices();
    const auto num_row_ranges = row_partition->get_num_ranges();
    const auto num_col_ranges = col_partition->get_num_ranges();
    const auto num_input_elements = input.get_num_elems();

    Array<size_type> row_range_id{exec, num_input_elements};
    compute_range_ids(row_range_bounds, num_row_ranges,
                      input.get_const_row_idxs(), num_input_elements,
                      row_range_id.get_data());
    Array<size_type> col_range_id{exec, input.get_num_elems()};
    compute_range_ids(col_range_bounds, num_col_ranges,
                      input.get_const_col_idxs(), num_input_elements,
                      col_range_id.get_data());

    // count number of diag<0> and offdiag<1> elements
    auto num_elements_pair = compute_local_sizes(
        row_range_id.get_const_data(), col_range_id.get_const_data(),
        num_input_elements, local_part, row_part_ids, col_part_ids);
    auto num_diag_elements = thrust::get<0>(num_elements_pair);
    auto num_offdiag_elements = thrust::get<1>(num_elements_pair);

    // copy and transform diag entries into arrays
    Array<LocalIndexType> diag_row_idxs{exec, num_diag_elements};
    Array<LocalIndexType> diag_col_idxs{exec, num_diag_elements};
    Array<ValueType> diag_values{exec, num_diag_elements};
    copy_local_inner(input, local_part, row_range_id.get_data(),
                     col_range_id.get_data(), row_part_ids, row_range_bounds,
                     row_range_starting_indices, col_part_ids, col_range_bounds,
                     col_range_starting_indices, diag_row_idxs.get_data(),
                     diag_col_idxs.get_data(), diag_values.get_data());
    // copy and transform offdiag entries into arrays, but keep global column
    // indices, also stores the column part id for each offdiag entry in an
    // array
    Array<LocalIndexType> offdiag_row_idxs{exec, num_offdiag_elements};
    Array<GlobalIndexType> offdiag_global_col_idxs{exec, num_offdiag_elements};
    Array<ValueType> offdiag_values{exec, num_offdiag_elements};
    Array<comm_index_type> offdiag_col_part_ids{exec, num_offdiag_elements};
    Array<size_type> offdiag_col_range_ids{exec, num_offdiag_elements};
    copy_local_ghost(input, local_part, row_range_id.get_data(),
                     col_range_id.get_data(), row_part_ids, row_range_bounds,
                     row_range_starting_indices, col_part_ids, col_range_bounds,
                     col_range_starting_indices, offdiag_row_idxs.get_data(),
                     offdiag_global_col_idxs.get_data(),
                     offdiag_values.get_data(), offdiag_col_part_ids.get_data(),
                     offdiag_col_range_ids.get_data());

    // 1. stable sort global columns, part-id and range-id according to
    // their part-id and global columns
    sort_offdiag_cols_and_rid_by_pid_and_idx(
        offdiag_col_part_ids.get_data(), offdiag_global_col_idxs.get_data(),
        offdiag_col_range_ids.get_data(), num_offdiag_elements);

    // 2. remove duplicate columns, now the new column i has global index
    // offdiag_global_col_idxs[i]
    auto offdiag_global_col_idxs_begin = offdiag_global_col_idxs.get_data();
    auto offdiag_global_col_idxs_end = make_offdiag_cols_unique(
        offdiag_col_part_ids.get_data(), offdiag_global_col_idxs.get_data(),
        offdiag_col_range_ids.get_data(), num_offdiag_elements);
    auto num_offdiag_cols = static_cast<size_type>(thrust::distance(
        offdiag_global_col_idxs_begin, offdiag_global_col_idxs_end));

    // 2.5 copy unique_columns to array
    local_to_global_ghost.resize_and_reset(num_offdiag_cols);
    exec->copy(num_offdiag_cols, offdiag_global_col_idxs_begin,
               local_to_global_ghost.get_data());
    auto local_to_global_ghost_data = local_to_global_ghost.get_const_data();

    // 3. create mapping from unique_columns
    // since we don't have hash tables on GPUs I'm first sorting the offdiag
    // global column indices and their new local index again by the global
    // column index. Then I'm using binary searches to find the new local column
    // index.
    Array<LocalIndexType> permutation{exec, num_offdiag_cols};
    sort_offdiag_cols_by_idx<LocalIndexType>(
        offdiag_col_part_ids.get_data(), offdiag_global_col_idxs.get_data(),
        permutation.get_data(), num_offdiag_cols);

    // 4. map column index of offdiag entries to new columns
    // on cuda devices calling `thrust::lower_bound` from device code doesn't
    // work it always returns the begin iterator because of that I have to
    // precompute
    Array<LocalIndexType> offdiag_col_idxs{exec, num_offdiag_elements};
    map_global_cols_to_local(
        exec, input.get_const_col_idxs(), num_input_elements,
        num_offdiag_elements, num_offdiag_cols, local_part,
        row_range_id.get_const_data(), col_range_id.get_const_data(),
        row_part_ids, col_part_ids, permutation.get_const_data(),
        offdiag_global_col_idxs.get_data(), offdiag_col_idxs.get_data());

    // compute gather idxs and recv_offsets
    local_gather_idxs.resize_and_reset(num_offdiag_cols);
    compute_gather_idxs_and_sizes(col_range_bounds, col_range_starting_indices,
                                  local_to_global_ghost.get_const_data(),
                                  offdiag_col_part_ids.get_const_data(),
                                  offdiag_col_range_ids.get_const_data(),
                                  num_offdiag_cols, num_parts,
                                  local_gather_idxs.get_data(), recv_sizes);

    const auto num_diag_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_diag_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    diag_data = std::move(device_matrix_data<ValueType, LocalIndexType>(
        exec, gko::dim<2>{num_diag_rows, num_diag_cols},
        std::move(diag_row_idxs), std::move(diag_col_idxs),
        std::move(diag_values)));

    offdiag_data = std::move(device_matrix_data<ValueType, LocalIndexType>(
        exec, gko::dim<2>{num_diag_rows, num_offdiag_cols},
        std::move(offdiag_row_idxs), std::move(offdiag_col_idxs),
        std::move(offdiag_values)));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_DIAG_OFFDIAG);
