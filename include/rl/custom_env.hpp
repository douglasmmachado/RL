#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>

namespace rl_tools::rl::environments::gridworld_1d {

    // 1. Parameters: constants or episode-level configuration for our environment.
    // Imagine a 1D corridor from x = -1.0 to x = +1.0, where the Goal is at target_x = +0.8.
    template <typename T>
    struct DefaultParameters {
        constexpr static T target_x = 0.8;
        constexpr static T max_step = 0.2;   // Maximum distance moved per action in [-1, 1]
        constexpr static T boundary = 1.0;   // Corridor walls at [-1.0, +1.0]
        constexpr static T goal_radius = 0.05;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    // 2. State: holds all variables that change at each step(state, action, next_state)
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 1;
        T x; // Current position of the agent in [-1.0, 1.0]
    };

    // 3. Observation: describes the shape of the vector fed into the Actor/Critic neural networks
    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 1;
    };

} // namespace rl_tools::rl::environments::gridworld_1d

namespace rl_tools::rl::environments {

    // 4. Environment Definition: ties State, Parameters, Observation, and Action dimensions together
    template <typename T_SPEC>
    struct GridWorld1D : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = gridworld_1d::State<T, TI>;
        using Observation = gridworld_1d::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 1;         // 1D continuous velocity action in [-1, +1]
        static constexpr TI EPISODE_STEP_LIMIT = 50; // Max steps per episode
    };

} // namespace rl_tools::rl::environments

// 5. Environment Operations: in RLtools, methods are free template functions in namespace rl_tools!
namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::GridWorld1D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::GridWorld1D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::GridWorld1D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, RNG&) {}

    // Deterministic start state (used during evaluation runs)
    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, typename rl::environments::GridWorld1D<SPEC>::State& state) {
        state.x = -0.5; // Always start at x = -0.5 during evaluation
    }

    // Random start state (used at the start of training episodes)
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, typename rl::environments::GridWorld1D<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.9, (T)0.5, rng);
    }

    // Transition dynamics: takes (state, action) and writes next_state
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, const typename rl::environments::GridWorld1D<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::GridWorld1D<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;

        // Clamp action to [-1, 1] and move agent
        T u = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T new_x = state.x + u * PARAMS::max_step;
        next_state.x = math::clamp(device.math, new_x, -PARAMS::boundary, PARAMS::boundary);

        return (T)1.0; // dt
    }

    // Reward function: penalize distance to target_x and action effort
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, const typename rl::environments::GridWorld1D<SPEC>::State&, const Matrix<ACTION_SPEC>& action, const typename rl::environments::GridWorld1D<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;

        T dist = math::abs(device.math, next_state.x - PARAMS::target_x);
        T u = get(action, 0, 0);
        T r = -dist - (T)0.01 * (u * u);
        if (dist <= PARAMS::goal_radius) {
            r += (T)5.0; // Bonus for reaching goal!
        }
        return r;
    }

    // Observation function: fills the neural network input vector from `state`
    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::GridWorld1D<SPEC>&, const typename rl::environments::GridWorld1D<SPEC>::Parameters&, const typename rl::environments::GridWorld1D<SPEC>::State& state, const rl::environments::gridworld_1d::Observation<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 1);
        set(observation, 0, 0, state.x);
    }

    // Termination condition: episode ends early if the agent reaches the goal
    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::GridWorld1D<SPEC>&, typename rl::environments::GridWorld1D<SPEC>::Parameters&, const typename rl::environments::GridWorld1D<SPEC>::State state, RNG&) {
        using PARAMS = typename SPEC::PARAMETERS;
        return math::abs(device.math, state.x - PARAMS::target_x) <= PARAMS::goal_radius;
    }

} // namespace rl_tools
