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

#include "core/components/device_matrix_data_kernels.hpp"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename ValueIndexType>
class DeviceMatrixData : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using nonzero_type = gko::matrix_data_entry<value_type, index_type>;

    DeviceMatrixData() : rand{82754} {}

    void SetUp()
    {
        init_executor(gko::ReferenceExecutor::create(), exec);
        host_data.size = {100, 200};
        std::uniform_int_distribution<index_type> row_distr(
            0, host_data.size[0] - 1);
        std::uniform_int_distribution<index_type> col_distr(
            0, host_data.size[1] - 1);
        std::uniform_real_distribution<gko::remove_complex<value_type>>
            val_distr(1.0, 2.0);
        // add random entries
        for (int i = 0; i < 1000; i++) {
            host_data.nonzeros.emplace_back(
                row_distr(rand), col_distr(rand),
                gko::test::detail::get_rand_value<value_type>(val_distr, rand));
        }
        // add random numerical zeros
        for (int i = 0; i < 1000; i++) {
            host_data.nonzeros.emplace_back(row_distr(rand), col_distr(rand),
                                            gko::zero<value_type>());
        }
        // remove duplicate nonzero locations
        host_data.ensure_row_major_order();
        host_data.nonzeros.erase(
            std::unique(host_data.nonzeros.begin(), host_data.nonzeros.end(),
                        [](nonzero_type nz1, nonzero_type nz2) {
                            return nz1.row == nz2.row &&
                                   nz1.column == nz2.column;
                        }),
            host_data.nonzeros.end());
        sorted_host_data = host_data;
        // shuffle the data again
        std::shuffle(host_data.nonzeros.begin(), host_data.nonzeros.end(),
                     rand);
        nonzero_host_data = host_data;
        nonzero_host_data.remove_zeros();
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::default_random_engine rand;
    gko::matrix_data<value_type, index_type> host_data;
    gko::matrix_data<value_type, index_type> sorted_host_data;
    gko::matrix_data<value_type, index_type> nonzero_host_data;
};

TYPED_TEST_SUITE(DeviceMatrixData, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(DeviceMatrixData, DefaultConstructsCorrectly)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    gko::device_matrix_data<value_type, index_type> local_data{this->exec};

    ASSERT_EQ((gko::dim<2>{0, 0}), local_data.get_size());
    ASSERT_EQ(this->exec, local_data.get_executor());
    ASSERT_EQ(local_data.get_num_elems(), 0);
}


TYPED_TEST(DeviceMatrixData, ConstructsCorrectly)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    gko::device_matrix_data<value_type, index_type> local_data{
        this->exec, gko::dim<2>{4, 3}, 10};

    ASSERT_EQ((gko::dim<2>{4, 3}), local_data.get_size());
    ASSERT_EQ(this->exec, local_data.get_executor());
    ASSERT_EQ(local_data.get_num_elems(), 10);
}


TYPED_TEST(DeviceMatrixData, CreatesFromHost)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto data =
        gko::device_matrix_data<value_type, index_type>::create_from_host(
            this->exec, this->host_data);

    ASSERT_EQ(data.get_size(), this->host_data.size);
    auto host_device_data = gko::device_matrix_data<value_type, index_type>(
        this->exec->get_master(), data);
    for (gko::size_type i = 0; i < data.get_num_elems(); i++) {
        const auto entry = this->host_data.nonzeros[i];
        ASSERT_EQ(host_device_data.get_const_row_idxs()[i], entry.row);
        ASSERT_EQ(host_device_data.get_const_col_idxs()[i], entry.column);
        ASSERT_EQ(host_device_data.get_const_values()[i], entry.value);
    }
}


TYPED_TEST(DeviceMatrixData, EmptiesOut)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto data =
        gko::device_matrix_data<value_type, index_type>::create_from_host(
            this->exec, this->host_data);
    const auto original_ptr1 = data.get_const_row_idxs();
    const auto original_ptr2 = data.get_const_col_idxs();
    const auto original_ptr3 = data.get_const_values();

    auto arrays = data.empty_out();

    ASSERT_EQ(data.get_size(), gko::dim<2>{});
    ASSERT_EQ(data.get_num_elems(), 0);
    ASSERT_NE(data.get_const_row_idxs(), original_ptr1);
    ASSERT_NE(data.get_const_col_idxs(), original_ptr2);
    ASSERT_NE(data.get_const_values(), original_ptr3);
    ASSERT_EQ(arrays.row_idxs.get_const_data(), original_ptr1);
    ASSERT_EQ(arrays.col_idxs.get_const_data(), original_ptr2);
    ASSERT_EQ(arrays.values.get_const_data(), original_ptr3);
    arrays.row_idxs.set_executor(this->exec->get_master());
    arrays.col_idxs.set_executor(this->exec->get_master());
    arrays.values.set_executor(this->exec->get_master());
    for (gko::size_type i = 0; i < arrays.values.get_num_elems(); i++) {
        const auto entry = this->host_data.nonzeros[i];
        ASSERT_EQ(arrays.row_idxs.get_const_data()[i], entry.row);
        ASSERT_EQ(arrays.col_idxs.get_const_data()[i], entry.column);
        ASSERT_EQ(arrays.values.get_const_data()[i], entry.value);
    }
}


TYPED_TEST(DeviceMatrixData, CopiesToHost)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto local_data =
        gko::device_matrix_data<value_type, index_type>::create_from_host(
            this->exec, this->host_data)
            .copy_to_host();

    ASSERT_EQ(local_data.size, this->host_data.size);
    ASSERT_EQ(local_data.nonzeros, this->host_data.nonzeros);
}


TYPED_TEST(DeviceMatrixData, SortsRowMajor)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using device_matrix_data = gko::device_matrix_data<value_type, index_type>;
    auto device_data =
        device_matrix_data::create_from_host(this->exec, this->host_data);
    auto device_sorted_data = device_matrix_data::create_from_host(
        this->exec, this->sorted_host_data);

    device_data.sort_row_major();

    ASSERT_EQ(device_data.copy_to_host().nonzeros,
              device_sorted_data.copy_to_host().nonzeros);
}


TYPED_TEST(DeviceMatrixData, RemovesZeros)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using device_matrix_data = gko::device_matrix_data<value_type, index_type>;
    auto device_data =
        device_matrix_data::create_from_host(this->exec, this->host_data);
    auto device_nonzero_data = device_matrix_data::create_from_host(
        this->exec, this->nonzero_host_data);

    device_data.remove_zeros();

    ASSERT_EQ(device_data.copy_to_host().nonzeros,
              device_nonzero_data.copy_to_host().nonzeros);
}


TYPED_TEST(DeviceMatrixData, DoesntRemoveZerosIfThereAreNone)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using device_matrix_data = gko::device_matrix_data<value_type, index_type>;
    auto device_nonzero_data = device_matrix_data::create_from_host(
        this->exec, this->nonzero_host_data);
    auto original = device_nonzero_data;
    auto original_ptr1 = device_nonzero_data.get_const_row_idxs();
    auto original_ptr2 = device_nonzero_data.get_const_col_idxs();
    auto original_ptr3 = device_nonzero_data.get_const_values();

    device_nonzero_data.remove_zeros();

    // no reallocation
    ASSERT_EQ(device_nonzero_data.get_const_row_idxs(), original_ptr1);
    ASSERT_EQ(device_nonzero_data.get_const_col_idxs(), original_ptr2);
    ASSERT_EQ(device_nonzero_data.get_const_values(), original_ptr3);
    // unmodified data
    ASSERT_EQ(device_nonzero_data.copy_to_host().nonzeros,
              original.copy_to_host().nonzeros);
}


}  // namespace
