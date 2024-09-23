add_library(physics STATIC physics.cpp)

add_executable(threaded_euler_main threaded_euler_main.cpp)
target_link_libraries(threaded_euler_main physics)
