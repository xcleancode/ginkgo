/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_l_u_factors(
    size_type num_elements, const IndexType* __restrict__ row_idxs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values,
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_col_idxs, ValueType* __restrict__ l_values,
    const IndexType* __restrict__ u_row_ptrs,
    const IndexType* __restrict__ u_col_idxs, ValueType* __restrict__ u_values)
{
    const auto elem_id = thread::get_thread_id_flat<IndexType>();
    if (elem_id < num_elements) {
        const auto row = row_idxs[elem_id];
        const auto col = col_idxs[elem_id];
        const auto val = values[elem_id];
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[col];
        ValueType sum{val};
        ValueType last_operation{};
        while (l_idx < l_row_ptrs[row + 1] && u_idx < u_row_ptrs[col + 1]) {
            const auto l_col = l_col_idxs[l_idx];
            const auto u_col = u_col_idxs[u_idx];
            last_operation = zero<ValueType>();
            if (l_col == u_col) {
                last_operation = l_values[l_idx] * u_values[u_idx];
                sum -= last_operation;
            }
            l_idx += (l_col <= u_col);
            u_idx += (u_col <= l_col);
        }
        sum += last_operation;  // undo the last operation
        if (row > col) {
            auto to_write = sum / u_values[u_row_ptrs[col + 1] - 1];
            if (is_finite(to_write)) {
                l_values[l_idx - 1] = to_write;
            }
        } else {
            auto to_write = sum;
            if (is_finite(to_write)) {
                u_values[u_idx - 1] = to_write;
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         size_type iterations,
                         const matrix::Coo<ValueType, IndexType>* system_matrix,
                         matrix::Csr<ValueType, IndexType>* l_factor,
                         matrix::Csr<ValueType, IndexType>* u_factor)
{
    iterations = (iterations == 0) ? 10 : iterations;
    const auto num_elements = system_matrix->get_num_stored_elements();
    const auto block_size = default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_elements, static_cast<size_type>(block_size)));
    if (grid_dim > 0) {
        for (size_type i = 0; i < iterations; ++i) {
            kernel::compute_l_u_factors<<<grid_dim, block_size, 0,
                                          exec->get_stream()>>>(
                num_elements, system_matrix->get_const_row_idxs(),
                system_matrix->get_const_col_idxs(),
                as_device_type(system_matrix->get_const_values()),
                l_factor->get_const_row_ptrs(), l_factor->get_const_col_idxs(),
                as_device_type(l_factor->get_values()),
                u_factor->get_const_row_ptrs(), u_factor->get_const_col_idxs(),
                as_device_type(u_factor->get_values()));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);
