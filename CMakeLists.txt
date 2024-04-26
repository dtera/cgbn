cmake_minimum_required(VERSION 3.28)
project(cgbn CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SRCS src/*.cpp src/*.cc src/*.cu)
file(GLOB_RECURSE CUDA_SRCS samples/sample_01_add/add.cu samples/sample_03_powm/powm_odd.cu
)

include_directories(. include third_party/include)
link_directories(third_party/lib)

add_library(${PROJECT_NAME}_lib ${SRCS})

# Building tests
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

set(EXE_TARGETS)
# Building bench
foreach (cuda_file ${CUDA_SRCS})
    string(REPLACE ".cu" "" cuda_name ${cuda_file})
    string(REGEX REPLACE "(.*)?/" "" cuda_name ${cuda_name})
    add_executable(${PROJECT_NAME}_${cuda_name} ${cuda_file})
    target_link_libraries(${PROJECT_NAME}_${cuda_name} GTest::GTest gmp)
    list(APPEND EXE_TARGETS ${PROJECT_NAME}_${cuda_name})
endforeach ()

add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_lib ${EXE_TARGETS} PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "35;50;72"
)
