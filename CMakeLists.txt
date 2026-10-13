cmake_minimum_required(VERSION 3.12)
project(PerfectDouCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/bigobj /utf-8)
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

find_package(Threads REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(perfectdou STATIC
  src/ddz/cards.cpp
  src/ddz/moves.cpp
  src/ddz/action_space.cpp
  src/ddz/oracle.cpp
  src/ddz/game.cpp
  src/ddz/features.cpp
  src/algo/rollout.cpp
  src/algo/ppo.cpp
  src/algo/eval_douzero.cpp
  src/torch_nn/models.cpp
  src/torch_nn/ppo_learner.cpp
  src/torch_nn/tensor_bridge.cpp
)
target_include_directories(perfectdou PUBLIC src ${TORCH_INCLUDE_DIRS})
target_link_libraries(perfectdou PUBLIC Threads::Threads ${TORCH_LIBRARIES})

if(WIN32)
  target_link_libraries(perfectdou PUBLIC ws2_32)
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

add_executable(test_algo tests/test_algo.cpp)
target_link_libraries(test_algo PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_algo)

add_executable(test_torch_nn tests/test_torch_nn.cpp)
target_link_libraries(test_torch_nn PRIVATE perfectdou GTest::gtest_main)
gtest_discover_tests(test_torch_nn)
