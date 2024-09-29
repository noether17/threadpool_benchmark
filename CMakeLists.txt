cmake_minimum_required(VERSION 3.29)
project(threadpool_benchmark)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror")

enable_testing()
find_package(GTest 1.14.0 REQUIRED)
include_directories({GTEST_INCLUDE_DIRS})

add_subdirectory(src)
