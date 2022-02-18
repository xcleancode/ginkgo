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

#ifndef GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
#define GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_


#include <utility>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {
namespace syn {


#define GKO_ENABLE_IMPLEMENTATION_SELECTION(_name, _callable)                \
    template <typename Predicate, int... IntArgs, typename... TArgs,         \
              typename... InferredArgs>                                      \
    inline void _name(std::integer_sequence<int>, Predicate,                 \
                      std::integer_sequence<int, IntArgs...>,                \
                      ::gko::syn::type_list<TArgs...>, InferredArgs...)      \
        GKO_KERNEL_NOT_FOUND;                                                \
                                                                             \
    template <int K, int... Rest, typename Predicate, int... IntArgs,        \
              typename... TArgs, typename... InferredArgs>                   \
    inline void _name(                                                       \
        std::integer_sequence<int, K, Rest...>, Predicate is_eligible,       \
        std::integer_sequence<int, IntArgs...> int_args,                     \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs... args)     \
    {                                                                        \
        if (is_eligible(K)) {                                                \
            _callable<IntArgs..., TArgs...>(                                 \
                std::integer_sequence<int, K>(),                             \
                std::forward<InferredArgs>(args)...);                        \
        } else {                                                             \
            _name(std::integer_sequence<int, Rest...>(), is_eligible,        \
                  int_args, type_args, std::forward<InferredArgs>(args)...); \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(_name, _callable)          \
    template <typename Predicate, bool... BoolArgs, int... IntArgs,           \
              gko::size_type... SizeTArgs, typename... TArgs,                 \
              typename... InferredArgs>                                       \
    inline void _name(std::integer_sequence<int>, Predicate,                  \
                      std::integer_sequence<bool, BoolArgs...>,               \
                      std::integer_sequence<int, IntArgs...>,                 \
                      std::integer_sequence<gko::size_type, SizeTArgs...>,    \
                      ::gko::syn::type_list<TArgs...>, InferredArgs...)       \
        GKO_KERNEL_NOT_FOUND;                                                 \
                                                                              \
    template <int K, int... Rest, typename Predicate, bool... BoolArgs,       \
              int... IntArgs, gko::size_type... SizeTArgs, typename... TArgs, \
              typename... InferredArgs>                                       \
    inline void _name(                                                        \
        std::integer_sequence<int, K, Rest...>, Predicate is_eligible,        \
        std::integer_sequence<bool, BoolArgs...> bool_args,                   \
        std::integer_sequence<int, IntArgs...> int_args,                      \
        std::integer_sequence<gko::size_type, SizeTArgs...> size_args,        \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs... args)      \
    {                                                                         \
        if (is_eligible(K)) {                                                 \
            _callable<BoolArgs..., IntArgs..., SizeTArgs..., TArgs..., K>(    \
                std::forward<InferredArgs>(args)...);                         \
        } else {                                                              \
            _name(std::integer_sequence<int, Rest...>(), is_eligible,         \
                  bool_args, int_args, size_args, type_args,                  \
                  std::forward<InferredArgs>(args)...);                       \
        }                                                                     \
    }


}  // namespace syn
}  // namespace gko


#endif  // GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
