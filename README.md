# C++ Reinforcement Learning Studio

A complete, native **C++20 Reinforcement Learning & Control Studio** built with **Dear ImGui**, **ImPlot**, **DirectX 11**, and **RLtools** — combining from-scratch tabular RL, model-based Dynamic Programming, and compile-time optimized Deep RL (**SAC**, **PPO**, **TD3**) across **13 interactive environments**.

## Highlights

### 1. Interactive C++ RL Studio (`build/rl_studio.exe`)
- **13 Environments in One Native 60 FPS App:**
  1. **Multi-Armed Bandits** — 10-armed Gaussian testbed comparing **$\epsilon$-Greedy**, **UCB1**, and **Thompson Sampling** (True Mean vs. Q-Estimates, Cumulative Regret, % Optimal Pulls).
  2. **GridWorld Lab** — **Q-Learning vs. SARSA** Robots with flickering Fire Traps, **Cliff-Walking**, **Windy GridWorld (10x7)**, **FrozenLake Ice Slip** slider, **Click-to-Paint Map Editor**, **Hover 4-Action Q-Value Inspector**, and **Exact Model-Based Value Iteration (DP)**!
  3. **Taxi-v3** — 500-state hierarchical pickup/dropoff grid with colored landmarks (`R, G, Y, B`), walls, and animated Q-Learning vs. SARSA taxis.
  4. **CartPole-Continuous** (`SAC / PPO / TD3` + Center/Angle/Effort Reward Shaping).
  5. **Pendulum-v1 Swing-Up** (`RLtools` Built-in).
  6. **Acrobot 2-Link Swing-Up** (`RLtools` Built-in + Anti-Spin Velocity Penalty toggle).
  7. **MountainCar-Continuous** (Dense vs. Sparse Reward Shaping toggle).
  8. **Drone2D Planar Quadrotor Hover** (6D state, dual-rotor thrust + Tilt/Vel/Effort sliders).
  9. **ContinuousGrid2D Trap Arena** (Random multi-trap maps + Dense vs. Sparse toggle).
  10. **LunarLander / Moon Landing** (6D lunar descent, main rocket + RCS thrusters, soft touchdown vs. crash, Fuel Penalty slider).
  11. **Reacher-2DOF Robotic Arm** (8D observation, 2-joint continuous inverse kinematics).
  12. **Ball-and-Beam** (Classic control engineering rolling ball balance).
  13. **Inverted Double CartPole** (Balance two stacked poles simultaneously!).

- **Educational & Diagnostic Tools:**
  - **Live Hyperparameter Lab**: Real-time Adam learning rate (`alpha`), discount (`gamma`), $\epsilon$-decay, slip probability, and per-environment **Reward Shaping Lab** with `(?)` tooltips.
  - **Train-Until-Threshold Benchmark**: Race all algorithms to a target return and compare exact **Rounds**, **Env Steps (Sample Efficiency)**, and **Wall-Clock Time (ms)** in the Benchmark Table.
  - **Zoomable `ImPlot` Analytics**: Toggle X-axis between *Training Rounds*, *Env Steps*, and *Wall-Clock Seconds*, overlay **Snapshot Curves** across runs, inspect **Phase Space Portraits**, and view 25×25 **Neural Policy Heatmaps** (`RdBu`).
  - **Rollout Scrubber**: Play/Pause, Loop, speed control (`0.2x`–`2.5x`), and frame-by-frame inspection.

## Repository Structure

```text
├── CMakeLists.txt                       # CMake build system (rl_core, imgui_implot, rl_studio, CTest)
├── AGENTS.md                            # Coding conventions (Doxygen default)
├── include/rl/
│   ├── environment.hpp                  # Generic Environment<StateType, ActionType> & StepResult
│   ├── gridworld.hpp                    # 2D GridWorld (Walls, Traps, Cliff, Windy, Ice Slip, BFS)
│   ├── tabular_agent.hpp                # TabularAgent base class (1D Q-table, GLIE epsilon-greedy)
│   ├── q_learning_agent.hpp             # Off-policy Q-Learning (inherits TabularAgent)
│   ├── sarsa_agent.hpp                  # On-policy SARSA (inherits TabularAgent)
│   ├── dp_solver.hpp                    # Exact Model-Based Value Iteration solver
│   ├── bandits_lab.hpp                  # 10-Armed Bandits (Eps-Greedy, UCB1, Thompson Sampling)
│   ├── taxi_lab.hpp                     # 500-State Taxi-v3 hierarchical tabular RL lab
│   ├── cartpole.hpp                     # Native C++20 CartPole physics
│   ├── cartpole_discretizer.hpp         # 4D Continuous-to-Discrete bin discretizer (5184 states)
│   ├── reward_shaping.hpp               # Live runtime reward shaping config for RL Studio
│   ├── studio_engines.hpp               # Unified RLtools SAC/PPO/TD3 & Tabular Studio backends
│   ├── custom_env.hpp                   # RLtools ContinuousGrid2D random trap arena
│   ├── rltools_cartpole.hpp             # RLtools CartPole-Continuous
│   ├── rltools_mountaincar.hpp          # RLtools MountainCar-Continuous
│   ├── rltools_drone2d.hpp              # RLtools 2D Planar Quadrotor Hover
│   ├── rltools_lunarlander.hpp          # RLtools Moon Landing (LunarLander)
│   ├── rltools_reacher.hpp              # RLtools 2-DOF Robotic Arm Reacher
│   ├── rltools_ballbeam.hpp             # RLtools Ball-and-Beam Control
│   └── rltools_double_cartpole.hpp      # RLtools Inverted Double CartPole
├── src/
│   ├── rl_studio.cpp                    # Main Dear ImGui + ImPlot + DirectX11 RL Studio app
│   ├── gridworld.cpp                    # GridWorld dynamics, Windy/Cliff layouts & BFS check
│   ├── cartpole.cpp                     # CartPole Euler physics integration
│   ├── tabular_agent.cpp                # Shared TabularAgent implementation
│   ├── q_learning_agent.cpp             # Q-Learning Bellman update
│   ├── sarsa_agent.cpp                  # SARSA on-policy update
│   └── visualize_*.cpp                  # Standalone lightweight Win32 GDI visualizers & benchmarks
├── tests/
│   ├── test_gridworld.cpp               # GridWorld unit tests
│   ├── test_q_learning.cpp              # Q-Learning unit & convergence tests
│   ├── test_sarsa.cpp                   # SARSA on-policy unit tests
│   ├── test_cartpole.cpp                # CartPole Euler physics unit tests
│   └── test_cartpole_discretizer.cpp    # 4D Discretizer unit tests
└── external/
    ├── rl-tools/                        # RLtools header-only C++ deep RL library
    ├── imgui/                           # Dear ImGui UI library
    └── implot/                          # ImPlot interactive plotting library
```

## Build & Run

### Prerequisites
- C++20 compiler (`g++ 13+` via MinGW-W64 on Windows)
- CMake `3.16+` & Ninja (or MinGW Makefiles)

### Clone with Submodules
```powershell
git clone --recursive https://github.com/douglasmmachado/RL.git
cd RL
```

### Build & Launch RL Studio
```powershell
cmake -B build
cmake --build build --target rl_studio -j 4
.\build\rl_studio.exe
```

### Run All Unit Tests (CTest)
```powershell
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```
