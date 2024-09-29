include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(n_body_tests n_body_tests.cpp)
target_link_libraries(n_body_tests GTest::GTest GTest::Main physics)
add_test(NAME n_body_tests COMMAND n_body_tests)
