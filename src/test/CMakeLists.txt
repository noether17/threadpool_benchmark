include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(n_body_tests n_body_tests.cpp)
target_link_libraries(n_body_tests GTest::GTest GTest::Main physics)
add_test(NAME n_body_tests COMMAND n_body_tests)

add_executable(n_body_benchmarks n_body_benchmarks.cpp)
target_link_libraries(n_body_benchmarks
  physics
  benchmark::benchmark
  benchmark::benchmark_main
)

add_custom_target(run_benchmarks
  COMMAND n_body_benchmarks --benchmark_out=data.json
)
add_dependencies(run_benchmarks n_body_benchmarks)
