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
  src/algo/rollout.cpp
  src/algo/ppo.cpp
)
target_include_directories(perfectdou PUBLIC src)
target_link_libraries(perfectdou PUBLIC Threads::Threads)

add_executable(perfectdou_train src/train.cpp)
target_link_libraries(perfectdou_train PRIVATE perfectdou)

enable_testing()
add_executable(test_rules tests/test_rules.cpp)
target_link_libraries(test_rules PRIVATE perfectdou)
add_test(NAME rules COMMAND test_rules)

add_executable(test_grad tests/test_grad.cpp)
target_link_libraries(test_grad PRIVATE perfectdou)
add_test(NAME gradients COMMAND test_grad)
