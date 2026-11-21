cmake_minimum_required(VERSION 3.16)
project(cpp_rl VERSION 0.1.0 LANGUAGES CXX)

# Use C++20 for our native RL library and C++17+ compatibility for RLtools
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
elseif(MSVC)
    add_compile_options(/O2 /W3)
endif()

# Shared core RL library (GridWorld + TabularAgent + QLearningAgent + SarsaAgent)
add_library(rl_core STATIC
    src/gridworld.cpp
    src/tabular_agent.cpp
    src/q_learning_agent.cpp
    src/sarsa_agent.cpp
)
target_include_directories(rl_core PUBLIC include)

# 1. RLtools Continuous Control SAC Example (Custom GridWorld1D)
add_executable(rl_main
    src/main.cpp
)
target_include_directories(rl_main PRIVATE
    include
    external/rl-tools/include
)

# 2. Parallel Q-Learning vs. SARSA GUI Arena (Windows GDI)
add_executable(visualize_gridworld
    src/visualize_gridworld.cpp
)
target_link_libraries(visualize_gridworld PRIVATE rl_core)
if(WIN32)
    target_link_libraries(visualize_gridworld PRIVATE gdi32 user32)
endif()

# 3. Unit Testing with CTest
enable_testing()

add_executable(test_gridworld tests/test_gridworld.cpp)
target_link_libraries(test_gridworld PRIVATE rl_core)
add_test(NAME GridWorldTests COMMAND test_gridworld)

add_executable(test_q_learning tests/test_q_learning.cpp)
target_link_libraries(test_q_learning PRIVATE rl_core)
add_test(NAME QLearningTests COMMAND test_q_learning)

add_executable(test_sarsa tests/test_sarsa.cpp)
target_link_libraries(test_sarsa PRIVATE rl_core)
add_test(NAME SarsaTests COMMAND test_sarsa)
