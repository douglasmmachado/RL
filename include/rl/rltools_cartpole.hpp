/**
 * @file rltools_cartpole.hpp
 * @brief RLtools-compatible Continuous-Force CartPole environment for SAC, PPO, and TD3.
 *
 * State:       {x, x_dot, theta, theta_dot}
 * Observation: {x / 2.4, x_dot / 3.0, theta / 0.21, theta_dot / 3.5} (normalized for neural networks)
 * Action:      1D continuous force u in [-1.0, +1.0], scaled to force = u * 10.0 N
 * Reward:      +1.0 per step survived (minus small centering/effort shaping for smooth balancing)
 * Termination: |x| > 2.4 m or |theta| > 12 degrees (~0.2094 rad)
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>

namespace rl_tools::rl::environments::cartpole_continuous {

    /**
     * @brief Compile-time physical constants matching our native C++ CartPole environment.
     * @tparam T Floating-point scalar type (float or double).
     */
    template <typename T>
    struct DefaultParameters {
        constexpr static T gravity = 9.8;
        constexpr static T mass_cart = 1.0;
        constexpr static T mass_pole = 0.1;
        constexpr static T total_mass = mass_cart + mass_pole;
        constexpr static T half_length = 0.5;
        constexpr static T pole_mass_length = mass_pole * half_length;
        constexpr static T force_mag = 10.0;
        constexpr static T tau = 0.02;
        constexpr static T x_threshold = 2.4;
        constexpr static T theta_threshold = 12.0 * 3.14159265358979323846 / 180.0;
    };

    /**
     * @brief Specification bundling scalar type, index type, and physical parameters.
     */
    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    /**
     * @brief Continuous 4D simulator state {x, x_dot, theta, theta_dot}.
     */
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 4;
        T x;
        T x_dot;
        T theta;
        T theta_dot;
    };

    /**
     * @brief 4D observation vector shape fed into SAC, PPO, and TD3 Actor/Critic networks.
     */
    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 4;
    };

} // namespace rl_tools::rl::environments::cartpole_continuous

namespace rl_tools::rl::environments {

    /**
     * @brief RLtools Continuous-Force CartPole environment.
     * @tparam T_SPEC Specification struct defining T, TI, and PARAMETERS.
     */
    template <typename T_SPEC>
    struct CartPoleContinuous : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = cartpole_continuous::State<T, TI>;
        using Observation = cartpole_continuous::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 1;          ///< Continuous push force u in [-1, +1].
        static constexpr TI EPISODE_STEP_LIMIT = 500; ///< Max steps per episode (10 seconds at 50 Hz).
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&,
                                   typename rl::environments::CartPoleContinuous<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&,
                                          typename rl::environments::CartPoleContinuous<SPEC>::Parameters&, RNG&) {}

    /**
     * @brief Sets a deterministic slightly-tilted initial state for fair side-by-side evaluation.
     */
    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&,
                              typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                              typename rl::environments::CartPoleContinuous<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.x = (T)0.0;
        state.x_dot = (T)0.0;
        state.theta = (T)0.06; // Slight 3.4-degree tilt to test recovery!
        state.theta_dot = (T)0.0;
    }

    /**
     * @brief Samples a random initial state uniformly in [-0.05, +0.05] for training episodes.
     */
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::CartPoleContinuous<SPEC>&,
                                     typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                                     typename rl::environments::CartPoleContinuous<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.x_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
    }

    /**
     * @brief Integrates continuous-force CartPole physics for one step (tau = 0.02s).
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::CartPoleContinuous<SPEC>&,
                          typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                          const typename rl::environments::CartPoleContinuous<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::CartPoleContinuous<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T u = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T force = u * P::force_mag;

        T cos_t = math::cos(device.math, state.theta);
        T sin_t = math::sin(device.math, state.theta);

        T temp = (force + P::pole_mass_length * state.theta_dot * state.theta_dot * sin_t) / P::total_mass;
        T theta_acc = (P::gravity * sin_t - cos_t * temp) /
                      (P::half_length * ((T)(4.0 / 3.0) - (P::mass_pole * cos_t * cos_t) / P::total_mass));
        T x_acc = temp - (P::pole_mass_length * theta_acc * cos_t) / P::total_mass;

        next_state.x = state.x + P::tau * state.x_dot;
        next_state.x_dot = state.x_dot + P::tau * x_acc;
        next_state.theta = state.theta + P::tau * state.theta_dot;
        next_state.theta_dot = state.theta_dot + P::tau * theta_acc;

        return P::tau;
    }

    /**
     * @brief Returns +1.0 for every step survived (classic CartPole reward).
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&,
                                   typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                                   const typename rl::environments::CartPoleContinuous<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>&,
                                   const typename rl::environments::CartPoleContinuous<SPEC>::State&, RNG&) {
        using T = typename SPEC::T;
        return (T)1.0;
    }

    /**
     * @brief Writes the normalized 4D observation vector into the neural network input matrix.
     */
    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::CartPoleContinuous<SPEC>&,
                        const typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                        const typename rl::environments::CartPoleContinuous<SPEC>::State& state,
                        const rl::environments::cartpole_continuous::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 4);
        using T = typename SPEC::T;
        set(observation, 0, 0, state.x / (T)2.4);
        set(observation, 0, 1, state.x_dot / (T)3.0);
        set(observation, 0, 2, state.theta / (T)0.21);
        set(observation, 0, 3, state.theta_dot / (T)3.5);
    }

    /**
     * @brief Returns true if |x| > 2.4 m or |theta| > 12 degrees.
     */
    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::CartPoleContinuous<SPEC>&,
                           typename rl::environments::CartPoleContinuous<SPEC>::Parameters&,
                           const typename rl::environments::CartPoleContinuous<SPEC>::State state, RNG&) {
        using P = typename SPEC::PARAMETERS;
        return (math::abs(device.math, state.x) > P::x_threshold) ||
               (math::abs(device.math, state.theta) > P::theta_threshold);
    }

} // namespace rl_tools
