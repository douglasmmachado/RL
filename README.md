# C++ Reinforcement Learning Sandbox

A clean, hands-on **C++20 Reinforcement Learning** repository built from scratch to explore modern C++ software design (templates, inheritance, `<random>`, CMake/CTest, native Win32 GDI visualization) alongside foundational and continuous-control RL algorithms.

## Features
- **Generic RL Interface (`Environment<StateType, ActionType>`)**: Compile-time parameterized base interface mirroring Gymnasium's `reset()`, `step()`, and `StepResult`.
- **Procedural 2D GridWorld (`GridWorld`)**:
  - Configurable dimensions, impassable walls, multiple traps, and Goal states.
  - Built-in **Breadth-First Search (BFS)** solvability verifier so every randomly generated map is guaranteed solvable.
  - Classic **8x4 Cliff-Walking** layout preset for contrasting off-policy vs. on-policy learning.
- **Modular Tabular RL Agents (`TabularAgent` Hierarchy)**:
  - `TabularAgent`: Base class managing a cache-friendly flattened 1D Q-table (`std::vector<double>`) and `<random>` $\epsilon$-greedy exploration.
  - `QLearningAgent`: Off-policy Bellman optimality update returning per-step TD error.
  - `SarsaAgent`: On-policy State-Action-Reward-State-Action update returning per-step TD error.
- **Parallel Q-Learning vs. SARSA GUI Arena (`visualize_gridworld`)**:
  - Zero-dependency native Windows GDI dashboard displaying both agents learning side by side on identical maps.
  - Real-time Q-value heatmaps, greedy policy arrows (`↑ → ↓ ←`), and **3 dual-curve live plots** (Smoothed Episode Return, Mean TD Error / Bellman Loss, and Mean Max Q-Value).
- **Continuous Control with `RLtools` (`rl_main`)**:
  - Header-only `RLtools` submodule integrated with a custom C++ continuous 1D environment trained via Soft Actor-Critic (SAC).
- **Unit Testing & Doxygen Documentation**:
  - Automated C++ unit tests registered with CMake `CTest` (`test_gridworld`, `test_q_learning`, `test_sarsa`).
  - Full Doxygen comments across all headers and source files (enforced by `AGENTS.md`).

## Repository Structure

```text
├── CMakeLists.txt                  # CMake build system (rl_core static library, apps, CTest)
├── AGENTS.md                       # Coding conventions (Doxygen default & tutor workflow)
├── .gitignore                      # Ignores build artifacts, binaries, and local IDE config
├── include/rl/
│   ├── environment.hpp             # Environment<StateType, ActionType> & StepResult<StateType>
│   ├── gridworld.hpp               # 2D GridWorld (Walls, Traps, Cliff Walking, BFS generator)
│   ├── tabular_agent.hpp           # TabularAgent base class (1D Q-table, epsilon-greedy)
│   ├── q_learning_agent.hpp        # QLearningAgent : public TabularAgent
│   ├── sarsa_agent.hpp             # SarsaAgent : public TabularAgent
│   └── custom_env.hpp              # RLtools-compatible continuous 1D environment
├── src/
│   ├── gridworld.cpp               # GridWorld dynamics, Cliff layout & BFS solvability check
│   ├── tabular_agent.cpp           # Shared TabularAgent implementation
│   ├── q_learning_agent.cpp        # Off-policy Q-Learning update rule
│   ├── sarsa_agent.cpp             # On-policy SARSA update rule
│   ├── visualize_gridworld.cpp     # Parallel Q-Learning vs. SARSA Windows GUI Arena & Plots
│   └── main.cpp                    # RLtools SAC continuous control example
├── tests/
│   ├── test_gridworld.cpp          # GridWorld movement, wall clamping, trap & goal tests
│   ├── test_q_learning.cpp         # Q-Learning Bellman update & 5x5 convergence tests
│   └── test_sarsa.cpp              # SARSA on-policy update & 5x5 convergence tests
└── external/rl-tools/              # RLtools submodule (header-only C++ deep RL library)
```

## Build & Run

### Prerequisites
- C++20 compiler (e.g., MinGW-W64 `g++ 13+` on Windows)
- CMake `3.16+` (optional, for CMake/CTest builds)

### Option 1: Using `CMake` & `CTest` (Recommended)
```powershell
cmake -B build -G "MinGW Makefiles"
cmake --build build
ctest --test-dir build --output-on-failure
.\build\visualize_gridworld.exe
```

### Option 2: Direct `g++` Commands (Windows PowerShell)

**Compile & run the Parallel Q-Learning vs. SARSA GUI Arena:**
```powershell
g++ -std=c++20 -Wall -Wextra -Iinclude src/visualize_gridworld.cpp src/gridworld.cpp src/tabular_agent.cpp src/q_learning_agent.cpp src/sarsa_agent.cpp -lgdi32 -luser32 -o visualize_gridworld.exe
.\visualize_gridworld.exe
```

**Compile & run all Unit Tests:**
```powershell
g++ -std=c++20 -Wall -Wextra -Iinclude tests/test_gridworld.cpp src/gridworld.cpp -o test_gridworld.exe; .\test_gridworld.exe
g++ -std=c++20 -Wall -Wextra -Iinclude tests/test_q_learning.cpp src/gridworld.cpp src/tabular_agent.cpp src/q_learning_agent.cpp -o test_q_learning.exe; .\test_q_learning.exe
g++ -std=c++20 -Wall -Wextra -Iinclude tests/test_sarsa.cpp src/gridworld.cpp src/tabular_agent.cpp src/sarsa_agent.cpp -o test_sarsa.exe; .\test_sarsa.exe
```

**Compile & run the `RLtools` SAC Continuous Control Example:**
```powershell
g++ -std=c++17 -O3 -Iinclude -Iexternal/rl-tools/include src/main.cpp -o rl_main.exe; .\rl_main.exe
```

## GUI Arena Controls (`visualize_gridworld.exe`)

| Key | Action |
| :--- | :--- |
| **`L`** | Toggle **Live Parallel Training** (watch Q-Learning & SARSA explore and learn step-by-step) |
| **`T`** | Fast-train both agents **+100 episodes** and update the 3 live comparison plots |
| **`Space`** | Run a **Greedy Policy Race** (both trained agents navigate side-by-side without exploration) |
| **`K`** | Load the classic **8x4 Cliff-Walking** trap map (highlights Q-Learning vs. SARSA differences) |
| **`M`** | Generate a **New Random Map** (random dimensions, walls, and traps, verified solvable via BFS) |
| **`C`** | Clear both Q-tables and reset all plots |
| **`R`** | Reset the current episode (keeps learned Q-tables) |
| **`Esc`** | Close the window |

## Roadmap
- [x] Stage 1: Generic C++20 `Environment` template & 2D `GridWorld` with unit tests
- [x] Stage 2: Tabular `QLearningAgent` with 1D flattened Q-table & live heatmap visualizer
- [x] Stage 3: On-policy `SarsaAgent`, `TabularAgent` inheritance refactor & Parallel GUI Arena
- [ ] Stage 4: Continuous control environments (`CartPole` / 2D Continuous Navigation) & neural network function approximation
