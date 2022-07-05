cmake_minimum_required(VERSION 3.20)
project(CBmpResize)

set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

aux_source_directory(
        ${PROJECT_SOURCE_DIR}/src SRC_DIR
)

if(MSVC)
    set(CMAKE_CXX_FLAGS "/utf-8")
    set(CMAKE_C_FLAGS "/utf-8")
else()
    set(CMAKE_C_FLAGS -lm)
endif(MSVC)

add_executable(CBmpResize ${SRC_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
