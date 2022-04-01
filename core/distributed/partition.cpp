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

#include <ginkgo/core/distributed/partition.hpp>


#include "core/distributed/partition_kernels.hpp"
#include "ginkgo/core/base/math.hpp"


namespace gko {
namespace distributed {
namespace distributed_partition {


GKO_REGISTER_OPERATION(count_ranges, distributed_partition::count_ranges);
GKO_REGISTER_OPERATION(build_from_mapping,
                       distributed_partition::build_from_mapping);
GKO_REGISTER_OPERATION(build_from_contiguous,
                       distributed_partition::build_from_contiguous);
GKO_REGISTER_OPERATION(build_ranges_from_global_size,
                       distributed_partition::build_ranges_from_global_size);
GKO_REGISTER_OPERATION(build_starting_indices,
                       distributed_partition::build_starting_indices);
GKO_REGISTER_OPERATION(has_ordered_parts,
                       distributed_partition::has_ordered_parts);


}  // namespace distributed_partition


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec)
    : exec_{std::move(exec)},
      num_parts_{0},
      num_empty_parts_{0},
      size_{0},
      offsets_{exec_, {0}},
      starting_indices_{exec_, 0},
      part_sizes_{exec_, size_type{0}},
      part_ids_{exec_, 0}
{}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec, const Array<comm_index_type>& mapping,
    comm_index_type num_parts)
    : exec_{std::move(exec)},
      num_parts_{num_parts},
      num_empty_parts_{0},
      size_{-1},
      offsets_{exec_},
      starting_indices_{exec_},
      part_sizes_{exec_, static_cast<size_type>(num_parts)},
      part_ids_{exec_}
{
    auto local_mapping = make_temporary_clone(exec_, &mapping);
    size_type num_ranges{};
    exec_->run(distributed_partition::make_count_ranges(*local_mapping.get(),
                                                        num_ranges));
    offsets_.resize_and_reset(num_ranges + 1);
    offsets_.fill(zero<GlobalIndexType>());
    starting_indices_.resize_and_reset(num_ranges);
    part_ids_.resize_and_reset(num_ranges);
    exec_->run(distributed_partition::make_build_from_mapping(
        *local_mapping.get(), offsets_.get_data(), part_ids_.get_data()));
    finalize_construction();
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec, const Array<GlobalIndexType>& ranges)
    : exec_{std::move(exec)},
      num_parts_{static_cast<comm_index_type>(ranges.get_num_elems() - 1)},
      num_empty_parts_{0},
      size_{-1},
      offsets_{exec_, ranges.get_num_elems()},
      starting_indices_{exec_, ranges.get_num_elems() - 1},
      part_sizes_{exec_, ranges.get_num_elems() - 1},
      part_ids_{exec_, ranges.get_num_elems() - 1}
{
    offsets_.fill(zero<GlobalIndexType>());
    exec_->run(distributed_partition::make_build_from_contiguous(
        *make_temporary_clone(exec_, &ranges).get(), offsets_.get_data(),
        part_ids_.get_data()));
    finalize_construction();
}


template <typename GlobalIndexType>
Array<GlobalIndexType> build_ranges_from_global_size(
    std::shared_ptr<const Executor> exec, const comm_index_type num_parts,
    const GlobalIndexType global_size)
{
    Array<GlobalIndexType> ranges(exec, num_parts + 1);
    exec->run(distributed_partition::make_build_ranges_from_global_size(
        num_parts, global_size, ranges));
    return ranges;
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec, comm_index_type num_parts,
    GlobalIndexType global_size)
    : partition(exec,
                build_ranges_from_global_size(exec, num_parts, global_size))
{}


template <typename LocalIndexType, typename GlobalIndexType>
void partition<LocalIndexType, GlobalIndexType>::finalize_construction()
{
    exec_->run(distributed_partition::make_build_starting_indices(
        offsets_.get_const_data(), part_ids_.get_const_data(), get_num_ranges(),
        get_num_parts(), num_empty_parts_, starting_indices_.get_data(),
        part_sizes_.get_data()));
    size_ =
        exec_->copy_val_to_host(offsets_.get_const_data() + get_num_ranges());
}


template <typename LocalIndexType, typename GlobalIndexType>
bool partition<LocalIndexType, GlobalIndexType>::has_connected_parts()
{
    return this->get_num_parts() - this->get_num_empty_parts() ==
           this->get_num_ranges();
}


template <typename LocalIndexType, typename GlobalIndexType>
bool partition<LocalIndexType, GlobalIndexType>::has_ordered_parts()
{
    if (this->has_connected_parts()) {
        bool has_ordered_parts;
        exec_->run(distributed_partition::make_has_ordered_parts(
            this, &has_ordered_parts));
        return has_ordered_parts;
    } else {
        return false;
    }
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec, const partition& other)
    : partition(std::move(exec))
{
    *this = other;
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(
    std::shared_ptr<const Executor> exec, partition&& other)
    : partition(std::move(exec))
{
    *this = std::move(other);
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(const partition& other)
    : partition(other.get_executor(), other)
{}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>::partition(partition&& other)
    : partition(other.get_executor(), std::move(other))
{}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>&
partition<LocalIndexType, GlobalIndexType>::operator=(const partition& other)
{
    if (this != &other) {
        num_parts_ = other.num_parts_;
        num_empty_parts_ = other.num_empty_parts_;
        size_ = other.size_;
        offsets_ = other.offsets_;
        starting_indices_ = other.starting_indices_;
        part_sizes_ = other.part_sizes_;
        part_ids_ = other.part_ids_;
    }
    return *this;
}


template <typename LocalIndexType, typename GlobalIndexType>
partition<LocalIndexType, GlobalIndexType>&
partition<LocalIndexType, GlobalIndexType>::operator=(partition&& other)
{
    if (this != &other) {
        num_parts_ = std::exchange(other.num_parts_, 0);
        num_empty_parts_ = std::exchange(other.num_empty_parts_, 0);
        size_ = std::exchange(other.size_, 0);
        offsets_ = std::move(other.offsets_);
        starting_indices_ = std::move(other.starting_indices_);
        part_sizes_ = std::move(part_sizes_);
        part_ids_ = std::move(other.part_ids_);
    }
    return *this;
}


#define GKO_DECLARE_PARTITION(_local, _global) class partition<_local, _global>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_PARTITION);


}  // namespace distributed
}  // namespace gko
