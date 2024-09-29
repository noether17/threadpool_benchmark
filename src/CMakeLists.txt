add_library(physics STATIC physics.cpp)

add_executable(threaded_euler_main threaded_euler_main.cpp)
target_link_libraries(threaded_euler_main physics)

add_executable(thread_pool_main thread_pool_main.cpp)
target_link_libraries(thread_pool_main physics)
