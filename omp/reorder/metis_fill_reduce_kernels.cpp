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

#include "core/reorder/metis_fill_reduce_kernels.hpp"


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#if GKO_HAVE_METIS
#include <metis.h>
#endif


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace metis_fill_reduce {


template <typename IndexType>
void get_permutation(std::shared_ptr<const DefaultExecutor> exec,
                     IndexType num_vertices, const IndexType* row_ptrs,
                     const IndexType* col_idxs, const IndexType* vertex_weights,
                     IndexType* permutation, IndexType* inv_permutation)
#if GKO_HAVE_METIS
{
    IndexType num_vtxs = IndexType(num_vertices);
    idx_t options[METIS_NOPTIONS];
    GKO_ASSERT_NO_METIS_ERRORS(METIS_SetDefaultOptions(options));
    GKO_ASSERT_NO_METIS_ERRORS(
        METIS_NodeND(&num_vtxs, const_cast<IndexType*>(row_ptrs),
                     const_cast<IndexType*>(col_idxs),
                     const_cast<IndexType*>(vertex_weights), options,
                     permutation, inv_permutation));
}
#else
{
    std::iota(permutation, permutation + num_vertices, 0);
}
#endif

GKO_INSTANTIATE_FOR_EACH_METIS_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL);


}  // namespace metis_fill_reduce
}  // namespace omp
}  // namespace kernels
}  // namespace gko
