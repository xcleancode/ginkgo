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

#ifndef GKO_PUBLIC_CORE_MATRIX_CVCSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_CVCSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/base/accessors.hpp"
#include "core/base/extended_float.hpp"

namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType, typename IndexType>
class Ell;


template <typename ValueType>
class Dense;


/**
 * CVCSR stores a matrix in the cvcsrrdinate matrix format.
 *
 * The nonzero elements are stored in an array row-wise (but not neccessarily
 * sorted by column index within a row). Two extra arrays contain the row and
 * column indexes of each nonzero element of the matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup cvcsr
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision,
          typename StorageType = default_precision, typename IndexType = int32>
class Cvcsr
    : public Transposable,
      public EnableLinOp<Cvcsr<ValueType, StorageType, IndexType>>,
      public EnableCreateMethod<Cvcsr<ValueType, StorageType, IndexType>> {
    friend class EnableCreateMethod<Cvcsr>;
    friend class EnablePolymorphicObject<Cvcsr, LinOp>;
    friend class Ell<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Cvcsr<to_complex<ValueType>, to_complex<StorageType>,
                       IndexType>;

public:
    using value_type = ValueType;
    using storage_type = StorageType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Cvcsr>;
    using transposed_type = Cvcsr<ValueType, StorageType, IndexType>;

    using accessor =
        gko::accessor::reduced_row_major<1, value_type, storage_type>;
    using const_accessor =
        gko::accessor::reduced_row_major<1, value_type, const storage_type>;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    range<accessor> get_values() noexcept
    {
        return range<accessor>(
            num_stored_elements_per_row_ * this->get_size()[0],
            values_.get_data());
    }

    /**
     * @copydoc Csr::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const range<const_accessor> get_const_values() const noexcept
    {
        return range<const_accessor>(
            num_stored_elements_per_row_ * this->get_size()[0],
            values_.get_const_data());
    }

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the row indexes of the matrix.
     *
     * @return the row indexes of the matrix.
     */
    // index_type *get_row_ptrs() noexcept { return csr_->get_row_ptrs(); }

    /**
     * @copydoc Csr::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    /*const index_type *get_const_row_ptrs() const noexcept
    {
        return csr_->get_const_row_ptrs();
    }*/

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements_per_row() const noexcept
    {
        return num_stored_elements_per_row_;
    }

    size_type get_stride() const noexcept { return stride_; }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

protected:
    /**
     * Creates an uninitialized CVCSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    Cvcsr(std::shared_ptr<const Executor> exec, const dim<2> &size = dim<2>{},
          size_type num_nonzeros = {})
        : EnableLinOp<Cvcsr>(exec, size)
    {}

    /**
     * Creates a CVCSR matrix from already allocated (and initialized) row
     * index, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowIdxArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray>
    Cvcsr(std::shared_ptr<const Executor> exec, const dim<2> &size,
          ValuesArray &&values, ColIdxsArray &&col_idxs,
          size_type num_stored_elements_per_row, size_type stride)
        : EnableLinOp<Cvcsr>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          num_stored_elements_per_row_{num_stored_elements_per_row},
          stride_{stride}
    {
        GKO_ASSERT_EQ(num_stored_elements_per_row_ * stride_,
                      values_.get_num_elems());
        GKO_ASSERT_EQ(num_stored_elements_per_row_ * stride_,
                      col_idxs_.get_num_elems());
    }

    template <typename InputValueType>
    Cvcsr(std::shared_ptr<const Executor> exec,
          std::shared_ptr<Csr<InputValueType, index_type>> csr)
        : EnableLinOp<Cvcsr>(exec, csr->get_size())
    {
        auto tmp_csr = as<Csr<remove_complex<InputValueType>, index_type>>(csr);
        values_ = Array<storage_type>(exec);
        auto tmp_ell =
            Ell<remove_complex<InputValueType>, index_type>::create(exec);
        tmp_csr->convert_to(tmp_ell.get());
        stride_ = tmp_ell->get_stride();
        auto num_rows = this->get_size()[0];
        num_stored_elements_per_row_ =
            tmp_ell->get_num_stored_elements_per_row();
        auto tmp_vals = Array<remove_complex<InputValueType>>::view(
            exec, num_stored_elements_per_row_ * num_rows,
            tmp_ell->get_values());
        values_ = tmp_vals;
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void apply2_impl(const LinOp *b, LinOp *x) const;

    void apply2_impl(const LinOp *alpha, const LinOp *b, LinOp *x) const;

private:
    std::shared_ptr<Ell<next_precision<value_type>, index_type>> ell_;
    Array<storage_type> values_;
    Array<index_type> col_idxs_;
    size_type num_stored_elements_per_row_;
    size_type stride_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_CVCSR_HPP_
