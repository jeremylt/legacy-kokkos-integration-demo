cmake_minimum_required(VERSION 4.0)
project(GameOfLife)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
# Change this to /tags/5.0.0 or something else to check forward compatibility
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.04.zip
)
FetchContent_MakeAvailable(Kokkos)

# Add executable
add_executable(GameOfLife_CPU GameOfLife_CPU.cpp)
add_executable(GameOfLife_Kokkos GameOfLife_Kokkos.cpp)
target_link_libraries(GameOfLife_Kokkos Kokkos::kokkos)
