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


#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using hyb_mtx = gko::matrix::Hybrid<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using metis_reorder_t = gko::reorder::MetisFillReduce<ValueType, IndexType>;
    using rcm_reorder_t = gko::reorder::Rcm<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto mtx_fname = argc >= 3 ? argv[2] : "A.mtx";
    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Read data
    auto A = share(
        gko::read<mtx>(std::ifstream("data/" + std::string(mtx_fname)), exec));
    A->set_strategy(std::make_shared<mtx::sparselib>());
    // Create RHS and initial guess as 1
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::clone(exec, host_x);
    auto b = gko::clone(exec, host_x);
    auto metis_reorder = metis_reorder_t::build().on(exec)->generate(A);
    auto rcm_reorder = rcm_reorder_t::build()
                           // .with_construct_inverse_permutation(true)
                           .on(exec)
                           ->generate(A);

    auto rcm_A =
        gko::share(mtx::create(exec, std::make_shared<mtx::sparselib>()));
    auto metis_A =
        gko::share(mtx::create(exec, std::make_shared<mtx::sparselib>()));
    rcm_A->copy_from(A.get());
    metis_A->copy_from(A.get());

    metis_reorder->get_permutation()->apply(A.get(), metis_A.get());
    rcm_reorder->get_permutation()->apply(A.get(), rcm_A.get());
    // auto rcm_hyb_A = gko::share(
    //     hyb_mtx::create(exec, std::make_shared<hyb_mtx::automatic>()));
    // auto metis_hyb_A = gko::share(
    //     hyb_mtx::create(exec, std::make_shared<hyb_mtx::automatic>()));
    // rcm_hyb_A->copy_from(rcm_A.get());
    // metis_hyb_A->copy_from(metis_A.get());

    // std::ofstream o;
    // o.open("metis_mat.mtx");
    // write(o, metis_A.get());
    // o.close();
    // o.open("rcm_mat.mtx");
    // write(o, rcm_A.get());
    // o.close();
    // o.open("base_mat.mtx");
    // write(o, A.get());
    // o.close();

    // Solve system
    // exec->synchronize();
    // auto x0 = x->clone();
    // auto x1 = x->clone();
    // auto x2 = x->clone();
    // A->apply(lend(b), lend(x0));
    // exec->synchronize();
    // rcm_A->apply(lend(b), lend(x1));
    // exec->synchronize();
    // metis_A->apply(lend(b), lend(x2));
    // exec->synchronize();
    // auto ix1 = x->clone();
    // auto ix2 = x->clone();
    // rcm_reorder->get_inverse_permutation()->apply(x1.get(), ix1.get());
    // metis_reorder->get_inverse_permutation()->apply(x2.get(), ix2.get());
    // auto neg_one = gko::initialize<vec>({-1.0}, exec);
    // x1->add_scaled(neg_one.get(), ix1.get());
    // x2->add_scaled(neg_one.get(), ix2.get());

    // auto rcm_err = gko::initialize<real_vec>({0.0}, exec->get_master());
    // auto metis_err = gko::initialize<real_vec>({0.0}, exec->get_master());
    // ix1->compute_norm2(lend(rcm_err));
    // ix2->compute_norm2(lend(metis_err));
    // std::cout << " RCM error:" << std::endl;
    // write(std::cout, rcm_err.get());
    // std::cout << " METIS error:" << std::endl;
    // write(std::cout, metis_err.get());

    std::chrono::nanoseconds base_time(0);
    std::chrono::nanoseconds rcm_time(0);
    std::chrono::nanoseconds metis_time(0);
    exec->synchronize();
    auto x_clone = x->clone();
    for (int i = 0; i < 3; ++i) {
        x->copy_from(x_clone.get());
        A->apply(lend(b), lend(x));
        exec->synchronize();
        x->copy_from(x_clone.get());
        rcm_A->apply(lend(b), lend(x));
        exec->synchronize();
        x->copy_from(x_clone.get());
        metis_A->apply(lend(b), lend(x));
        exec->synchronize();
    }
    for (int i = 0; i < 10; ++i) {
        x->copy_from(x_clone.get());
        auto btic = std::chrono::steady_clock::now();
        A->apply(lend(b), lend(x));
        exec->synchronize();
        auto btoc = std::chrono::steady_clock::now();
        base_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(btoc - btic);
        x->copy_from(x_clone.get());
        auto rtic = std::chrono::steady_clock::now();
        rcm_A->apply(lend(b), lend(x));
        exec->synchronize();
        auto rtoc = std::chrono::steady_clock::now();
        rcm_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(rtoc - rtic);
        x->copy_from(x_clone.get());
        auto mtic = std::chrono::steady_clock::now();
        metis_A->apply(lend(b), lend(x));
        exec->synchronize();
        auto mtoc = std::chrono::steady_clock::now();
        metis_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(mtoc - mtic);
    }

    // Print solver statistics
    std::cout << "Matrix size: " << A->get_size()
              << "\nMatrix nnz count: " << A->get_num_stored_elements()
              << "\n\tBase SpMV execution time [ms]: "
              << static_cast<double>(base_time.count()) / 10000000.0
              << "\n\tRcm SpMV execution time [ms]: "
              << static_cast<double>(rcm_time.count()) / 10000000.0
              << "\n\tMetis SpMV execution time [ms]: "
              << static_cast<double>(metis_time.count()) / 10000000.0
              << std::endl;
}
