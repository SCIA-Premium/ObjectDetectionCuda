cmake_minimum_required(VERSION 3.8)
project(CPU_IMPLEM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(
    stb_image
    GIT_REPOSITORY https://github.com/nothings/stb
)
FetchContent_MakeAvailable(stb_image)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

include_directories(${stb_image_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Werror -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SRC_ENGINE)
file(GLOB_RECURSE SRC_ENGINE "src/*.cc")

add_executable(cpu_implem)
target_sources(cpu_implem PRIVATE ${SRC_ENGINE})
target_link_libraries(cpu_implem PRIVATE nlohmann_json::nlohmann_json)