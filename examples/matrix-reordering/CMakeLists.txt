cmake_minimum_required(VERSION 3.9)
project(matrix-reordering)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(matrix-reordering matrix-reordering.cpp)
target_link_libraries(matrix-reordering Ginkgo::ginkgo)

configure_file(data/A.mtx data/A.mtx COPYONLY)
