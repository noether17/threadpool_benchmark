add_library(physics STATIC physics.cpp)

add_executable(main main.cpp)
target_link_libraries(main physics)
