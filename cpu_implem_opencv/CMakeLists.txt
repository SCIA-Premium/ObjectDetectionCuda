cmake_minimum_required(VERSION 3.8)
project(OPENCV_IMPLEM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g" "-fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SRC_ENGINE)
file(GLOB_RECURSE SRC_ENGINE "src/*.cc")

find_package(OpenCV REQUIRED)

add_executable(opencv_implem)
target_sources(opencv_implem PRIVATE ${SRC_ENGINE})
target_link_libraries(opencv_implem PRIVATE ${OpenCV_LIBS})