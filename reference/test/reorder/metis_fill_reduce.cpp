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

#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {

#if GKO_HAVE_METIS


template <typename ValueIndexType>
class MetisFillReduce : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using reorder_type = gko::reorder::MetisFillReduce<v_type, i_type>;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    MetisFillReduce()
        : exec(gko::ReferenceExecutor::create()),
          metis_fill_reduce_factory(reorder_type::build().on(exec)),
          // clang-format off
          id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 0.0}, 
              {0.0, 1.0, 0.0}, 
              {0.0, 0.0, 1.0}}, exec)),
          not_id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 1.0}, 
              {0.0, 1.0, 0.0}, 
              {1.0, 0.0, 1.0}}, exec)),
          // clang-format on
          reorder_op(metis_fill_reduce_factory->generate(id3_mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> id3_mtx;
    std::shared_ptr<CsrMtx> not_id3_mtx;
    std::unique_ptr<typename reorder_type::Factory> metis_fill_reduce_factory;
    std::unique_ptr<reorder_type> reorder_op;
};

TYPED_TEST_SUITE(MetisFillReduce, gko::test::ValueMetisIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(MetisFillReduce, CanBeCleared)
{
    this->reorder_op->clear();

    auto reorder_op_perm = this->reorder_op->get_permutation();

    ASSERT_EQ(reorder_op_perm, nullptr);
}


TYPED_TEST(MetisFillReduce, CanBeCopied)
{
    auto metis_fill_reduce =
        this->metis_fill_reduce_factory->generate(this->id3_mtx);
    auto metis_fill_reduce_copy =
        this->metis_fill_reduce_factory->generate(this->not_id3_mtx);

    metis_fill_reduce_copy->copy_from(metis_fill_reduce.get());

    EXPECT_EQ(
        metis_fill_reduce_copy->get_permutation()->get_const_permutation()[0],
        1);
    EXPECT_EQ(
        metis_fill_reduce_copy->get_permutation()->get_const_permutation()[1],
        2);
    EXPECT_EQ(
        metis_fill_reduce_copy->get_permutation()->get_const_permutation()[2],
        0);
}


TYPED_TEST(MetisFillReduce, CanBeMoved)
{
    auto metis_fill_reduce =
        this->metis_fill_reduce_factory->generate(this->id3_mtx);
    auto metis_fill_reduce_move =
        this->metis_fill_reduce_factory->generate(this->not_id3_mtx);

    metis_fill_reduce_move->move_from(metis_fill_reduce.get());

    EXPECT_EQ(
        metis_fill_reduce_move->get_permutation()->get_const_permutation()[0],
        1);
    EXPECT_EQ(
        metis_fill_reduce_move->get_permutation()->get_const_permutation()[1],
        2);
    EXPECT_EQ(
        metis_fill_reduce_move->get_permutation()->get_const_permutation()[2],
        0);
}


TYPED_TEST(MetisFillReduce, CanBeCloned)
{
    auto metis_fill_reduce =
        this->metis_fill_reduce_factory->generate(this->id3_mtx);

    auto metis_fill_reduce_clone = metis_fill_reduce->clone();

    EXPECT_EQ(
        metis_fill_reduce_clone->get_permutation()->get_const_permutation()[0],
        1);
    EXPECT_EQ(
        metis_fill_reduce_clone->get_permutation()->get_const_permutation()[1],
        2);
    EXPECT_EQ(
        metis_fill_reduce_clone->get_permutation()->get_const_permutation()[2],
        0);
}

#endif

}  // namespace
