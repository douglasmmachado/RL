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
    src/cartpole.cpp
    src/tabular_agent.cpp
    src/q_learning_agent.cpp
    src/sarsa_agent.cpp
)
target_include_directories(rl_core PUBLIC include)

# 1. RLtools Continuous Control Benchmark & GUI Arena: Off-Policy SAC vs. On-Policy PPO
add_executable(compare_sac_ppo
    src/compare_sac_ppo.cpp
)
target_include_directories(compare_sac_ppo PRIVATE
    include
    external/rl-tools/include
)

add_executable(visualize_sac_ppo
    src/visualize_sac_ppo.cpp
)
target_include_directories(visualize_sac_ppo PRIVATE
    include
    external/rl-tools/include
)
if(WIN32)
    target_link_libraries(visualize_sac_ppo PRIVATE gdi32 user32)
endif()

# 2. Parallel Q-Learning vs. SARSA GUI Arena (Windows GDI)
add_executable(visualize_gridworld
    src/visualize_gridworld.cpp
)
target_link_libraries(visualize_gridworld PRIVATE rl_core)
if(WIN32)
    target_link_libraries(visualize_gridworld PRIVATE gdi32 user32)
endif()

# 3. Animated CartPole Discretized Q-Learning GUI Arena (Windows GDI)
add_executable(visualize_cartpole
    src/visualize_cartpole.cpp
)
target_link_libraries(visualize_cartpole PRIVATE rl_core)
if(WIN32)
    target_link_libraries(visualize_cartpole PRIVATE gdi32 user32)
endif()

# 4. RLtools 3-Way Continuous CartPole Arena: SAC vs. PPO vs. TD3
add_executable(visualize_cartpole_rltools
    src/visualize_cartpole_rltools.cpp
)
target_include_directories(visualize_cartpole_rltools PRIVATE
    include
    external/rl-tools/include
)
if(WIN32)
    target_link_libraries(visualize_cartpole_rltools PRIVATE gdi32 user32)
endif()

# 5. Unified C++ RLtools Gym Suite (5 Environments x 3 Algorithms: SAC / PPO / TD3)
add_executable(visualize_gym_suite
    src/visualize_gym_suite.cpp
)
target_include_directories(visualize_gym_suite PRIVATE
    include
    external/rl-tools/include
)
if(WIN32)
    target_link_libraries(visualize_gym_suite PRIVATE gdi32 user32)
endif()

# 6. Complete Interactive C++ RL Studio (Dear ImGui + ImPlot + DirectX11 + RLtools)
add_library(imgui_implot STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_win32.cpp
    external/imgui/backends/imgui_impl_dx11.cpp
    external/implot/implot.cpp
    external/implot/implot_items.cpp
)
target_include_directories(imgui_implot PUBLIC
    external/imgui
    external/imgui/backends
    external/implot
)

add_executable(rl_studio
    src/rl_studio.cpp
)
target_include_directories(rl_studio PRIVATE
    include
    external/rl-tools/include
)
target_link_libraries(rl_studio PRIVATE rl_core imgui_implot)
if(WIN32)
    target_link_libraries(rl_studio PRIVATE d3d11 d3dcompiler dxgi dwmapi gdi32 user32)
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

add_executable(test_cartpole tests/test_cartpole.cpp)
target_link_libraries(test_cartpole PRIVATE rl_core)
add_test(NAME CartPoleTests COMMAND test_cartpole)
