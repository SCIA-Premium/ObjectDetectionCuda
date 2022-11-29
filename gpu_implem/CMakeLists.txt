cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

include(FetchContent)
FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )

FetchContent_Declare(
    stb_image
    GIT_REPOSITORY https://github.com/nothings/stb
)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(stb_image)

include_directories(${stb_image_SOURCE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wextra -pedantic -Wall -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_library(images)
target_compile_features(images PUBLIC cxx_std_14)
target_link_libraries(images PRIVATE spdlog::spdlog)
target_sources(images PRIVATE src/images.hpp src/images.cu)
set_target_properties(images PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Executables

add_executable(gpu_implem src/main.cpp)
target_compile_features(gpu_implem PUBLIC cxx_std_17)
target_link_libraries(gpu_implem PRIVATE images PNG::PNG CLI11::CLI11 spdlog::spdlog nlohmann_json::nlohmann_json)

# add_executable(bench src/bench.cpp)
# target_compile_features(bench PUBLIC cxx_std_17)
# target_link_libraries(bench PRIVATE render_cpu_ref spdlog::spdlog benchmark::benchmark renderer)



