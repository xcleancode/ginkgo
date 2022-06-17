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

#include <ginkgo/core/log/tau.hpp>


#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/stop/criterion.hpp>


#define PERFSTUBS_USE_TIMERS
#include <perfstubs_api/timer.h>


namespace gko {
namespace log {


void Tau::on_operation_launched(const Executor* exec,
                                const Operation* operation) const
{
    PERFSTUBS_START_STRING(operation->get_name());
}


void Tau::on_operation_completed(const Executor* exec,
                                 const Operation* operation) const
{
    PERFSTUBS_STOP_STRING(operation->get_name());
}


void Tau::on_polymorphic_object_copy_started(const Executor* exec,
                                             const PolymorphicObject* from,
                                             const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


void Tau::on_polymorphic_object_move_started(const Executor* exec,
                                             const PolymorphicObject* from,
                                             const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


void Tau::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                 const LinOp* x) const
{
    std::stringstream ss;
    ss << "apply(" << name_demangling::get_dynamic_type(*A) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                   const LinOp* x) const
{
    std::stringstream ss;
    ss << "apply(" << name_demangling::get_dynamic_type(*A) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


void Tau::on_linop_advanced_apply_started(const LinOp* A, const LinOp* alpha,
                                          const LinOp* b, const LinOp* beta,
                                          const LinOp* x) const
{
    std::stringstream ss;
    ss << "advanced_apply(" << name_demangling::get_dynamic_type(*A) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_linop_advanced_apply_completed(const LinOp* A, const LinOp* alpha,
                                            const LinOp* b, const LinOp* beta,
                                            const LinOp* x) const
{
    std::stringstream ss;
    ss << "advanced_apply(" << name_demangling::get_dynamic_type(*A) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


void Tau::on_linop_factory_generate_started(const LinOpFactory* factory,
                                            const LinOp* input) const
{
    std::stringstream ss;
    ss << "generate(" << name_demangling::get_dynamic_type(*factory) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_linop_factory_generate_completed(const LinOpFactory* factory,
                                              const LinOp* input,
                                              const LinOp* output) const
{
    std::stringstream ss;
    ss << "generate(" << name_demangling::get_dynamic_type(*factory) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


void Tau::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized) const
{
    std::stringstream ss;
    ss << "check(" << name_demangling::get_dynamic_type(*criterion) << ")";
    PERFSTUBS_START_STRING(ss.str().c_str());
}


void Tau::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stoppingId, const bool& setFinalized,
    const array<stopping_status>* status, const bool& oneChanged,
    const bool& converged) const
{
    std::stringstream ss;
    ss << "check(" << name_demangling::get_dynamic_type(*criterion) << ")";
    PERFSTUBS_STOP_STRING(ss.str().c_str());
}


}  // namespace log
}  // namespace gko
