cmake_minimum_required(VERSION 3.10)
project(deformation)

set(pybind_name "deformation")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # for pybind11

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w -std=c++11")

# add_definitions("-DNDEBUG")
include_directories(ext/eigen3)
include_directories(ext/spdlog/include)
add_subdirectory(ext/pybind11)
pybind11_add_module(${pybind_name}
    src/pybind.cpp
    src/log.cpp src/deform.cpp
    src/rotation/utils_rotation.cpp
)
