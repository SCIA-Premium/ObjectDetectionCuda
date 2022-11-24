cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)


FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )


FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(renderer)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer PRIVATE spdlog::spdlog)


if (NOT PROF)
  target_sources(renderer PRIVATE src/render.hpp src/render.cu)
  set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(render_cpu_ref SHARED IMPORTED)
  target_link_libraries(render_cpu_ref INTERFACE tbb)
  set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)
else()
  target_sources(renderer PRIVATE src/render.hpp src/render_prof.cu)
  set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_library(render_cpu_ref SHARED IMPORTED)
  target_link_libraries(render_cpu_ref INTERFACE tbb)
  set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)

  # Uncomment to regenerate the reference
  #add_library(render_cpu_ref SHARED src/render.hpp src/render_cpu.cpp)
  #target_link_libraries(render_cpu_ref PRIVATE tbb)
  #target_link_libraries(render_cpu_ref PRIVATE "-s" "-nostdlib")
  #target_compile_features(render_cpu_ref PRIVATE cxx_std_17)
  #target_compile_options(render_cpu_ref PRIVATE "-fno-rtti" "-fno-cxx-exceptions")
  #target_compile_options(render_cpu_ref PRIVATE "-mavx")
endif()




add_executable(mandel src/mandel.cpp)
target_compile_features(mandel PUBLIC cxx_std_17)
target_link_libraries(mandel PRIVATE render_cpu_ref PNG::PNG CLI11::CLI11 spdlog::spdlog renderer)

add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE render_cpu_ref spdlog::spdlog benchmark::benchmark renderer)



