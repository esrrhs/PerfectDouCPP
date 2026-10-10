cmake_minimum_required(VERSION 3.12)
project(PerfectDouCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(Threads REQUIRED)

add_library(perfectdou STATIC
  src/ddz/cards.cpp
  src/ddz/moves.cpp
  src/ddz/action_space.cpp
  src/ddz/oracle.cpp
  src/ddz/game.cpp
  src/ddz/features.cpp
  src/nn/net.cpp
  src/nn/gemm_cpu.cpp
  src/nn/kernels.cpp
  src/algo/rollout.cpp
  src/algo/ppo.cpp
  src/algo/eval_douzero.cpp
)
target_include_directories(perfectdou PUBLIC src)
target_link_libraries(perfectdou PUBLIC Threads::Threads)

if(APPLE)
  target_link_libraries(perfectdou PUBLIC "-framework Accelerate")
  enable_language(OBJCXX)
  target_sources(perfectdou PRIVATE src/nn/gemm_mps.mm)
  set_source_files_properties(src/nn/gemm_mps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_compile_definitions(perfectdou PUBLIC PD_HAVE_MPS)
  target_link_libraries(perfectdou PUBLIC
    "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" OR MSVC)
  target_sources(perfectdou PRIVATE src/nn/gemm_avx2.cpp)
  if(MSVC)
    set_source_files_properties(src/nn/gemm_avx2.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/nn/gemm_avx2.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
endif()

if(WIN32)
  target_sources(perfectdou PRIVATE
    src/nn/gemm_d3d.cpp
    src/nn/gemm_cuda_bridge.cpp)
  target_compile_definitions(perfectdou PUBLIC PD_HAVE_D3D)
  target_link_libraries(perfectdou PUBLIC d3d12 dxgi d3dcompiler ws2_32)
  find_path(CUDA_TOOLKIT_INCLUDE cuda.h
    HINTS "$ENV{CUDA_PATH}/include"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/include")
  if(CUDA_TOOLKIT_INCLUDE)
    target_include_directories(perfectdou PRIVATE "${CUDA_TOOLKIT_INCLUDE}")
    target_compile_definitions(perfectdou PRIVATE PD_HAVE_CUBLAS)
    message(STATUS "Optional cuBLAS D3D12 bridge enabled")
  endif()
endif()

add_executable(perfectdou_train src/train.cpp)
target_link_libraries(perfectdou_train PRIVATE perfectdou)
if(WIN32)
  target_link_libraries(perfectdou_train PRIVATE powrprof)
endif()

add_executable(perfectdou_eval src/eval_douzero.cpp)
target_link_libraries(perfectdou_eval PRIVATE perfectdou)

enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(test_rules tests/test_rules.cpp)
target_link_libraries(test_rules PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_rules)

add_executable(test_grad tests/test_grad.cpp)
target_link_libraries(test_grad PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_grad)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_gemm)

add_executable(test_algo tests/test_algo.cpp)
target_link_libraries(test_algo PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_algo)

