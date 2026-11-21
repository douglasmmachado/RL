/**
 * @file rltools_lunarlander.hpp
 * @brief RLtools-compatible LunarLanderContinuous-v2 (Moon Landing) environment for SAC, PPO, and TD3!
 *
 * Physics:
 * - Lunar gravity (-1.62 m/s^2, scaled for crisp 50Hz simulation)
 * - 6D State: {x, y, vx, vy, theta, omega} + landing leg contact detection
 * - 2D Continuous Actions:
 *     action[0] = main engine throttle in [-1, 1] -> [0, 1] thrust along lander body up-axis
 *     action[1] = side orientation thrusters in [-1, 1] -> torque + small lateral push
 * - Goal: Touch down softly on the lunar landing pad (|x| <= 0.25, y <= 0.0, |vx| < 0.35, |vy| < 0.45, |theta| < 0.25)!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>
#include "rl/reward_shaping.hpp"

namespace rl_tools::rl::environments::lunar_lander {

    /**
     * @brief Compile-time lunar physics parameters.
     * @tparam T Floating-point scalar type.
     */
    template <typename T>
    struct DefaultParameters {
        constexpr static T gravity = 3.2;            ///< Scaled lunar gravity [m/s^2]
        constexpr static T main_thrust = 8.5;        ///< Max main engine upward acceleration [m/s^2]
        constexpr static T side_torque = 6.0;        ///< Max angular acceleration from RCS side thrusters [rad/s^2]
        constexpr static T side_push = 1.2;          ///< Lateral push from RCS thrusters [m/s^2]
        constexpr static T dt = 0.04;                ///< Timestep [s]
        constexpr static T pad_half_width = 0.28;    ///< Landing pad x in [-0.28, +0.28]
        constexpr static T x_limit = 1.3;            ///< Horizontal out-of-bounds limit
        constexpr static T y_ceil = 1.6;             ///< Altitude ceiling
        constexpr static T angle_limit = 1.2;        ///< Flip crash angle (~68 deg)
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    /**
     * @brief 6D Lunar Lander state {x, y, vx, vy, theta, omega} + landed flag.
     */
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 6;
        T x, y;
        T vx, vy;
        T theta, omega;
        bool landed;
        bool crashed;
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 6;
    };

} // namespace rl_tools::rl::environments::lunar_lander

namespace rl_tools::rl::environments {

    /**
     * @brief RLtools Continuous Moon Landing (LunarLander) environment.
     */
    template <typename T_SPEC>
    struct LunarLander : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = lunar_lander::State<T, TI>;
        using Observation = lunar_lander::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;          ///< [main_engine, side_thruster] in [-1, 1].
        static constexpr TI EPISODE_STEP_LIMIT = 200;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::LunarLander<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::LunarLander<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::LunarLander<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::LunarLander<SPEC>&,
                                   typename rl::environments::LunarLander<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::LunarLander<SPEC>&,
                                          typename rl::environments::LunarLander<SPEC>::Parameters&, RNG&) {}

    /**
     * @brief Deterministic evaluation drop from high altitude with slight offset & tilt.
     */
    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::LunarLander<SPEC>&,
                              typename rl::environments::LunarLander<SPEC>::Parameters&,
                              typename rl::environments::LunarLander<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.x = (T)-0.45;
        state.y = (T)1.25;
        state.vx = (T)0.25;
        state.vy = (T)-0.15;
        state.theta = (T)0.12;
        state.omega = (T)0.0;
        state.landed = false;
        state.crashed = false;
    }

    /**
     * @brief Samples a randomized lunar descent start state for training.
     */
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::LunarLander<SPEC>&,
                                     typename rl::environments::LunarLander<SPEC>::Parameters&,
                                     typename rl::environments::LunarLander<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.65, (T)0.65, rng);
        state.y = random::uniform_real_distribution(device.random, (T)1.05, (T)1.35, rng);
        state.vx = random::uniform_real_distribution(device.random, (T)-0.35, (T)0.35, rng);
        state.vy = random::uniform_real_distribution(device.random, (T)-0.25, (T)0.05, rng);
        state.theta = random::uniform_real_distribution(device.random, (T)-0.20, (T)0.20, rng);
        state.omega = random::uniform_real_distribution(device.random, (T)-0.20, (T)0.20, rng);
        state.landed = false;
        state.crashed = false;
    }

    /**
     * @brief Integrates lunar descent dynamics with main engine + RCS side thrusters.
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::LunarLander<SPEC>&,
                          typename rl::environments::LunarLander<SPEC>::Parameters&,
                          const typename rl::environments::LunarLander<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::LunarLander<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T u_main_raw = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T u_side = math::clamp(device.math, get(action, 0, 1), (T)-1.0, (T)1.0);
        T main_throttle = (u_main_raw + (T)1.0) * (T)0.5; // [0, 1]

        T sin_t = math::sin(device.math, state.theta);
        T cos_t = math::cos(device.math, state.theta);

        T ax = -main_throttle * P::main_thrust * sin_t + u_side * P::side_push * cos_t;
        T ay = main_throttle * P::main_thrust * cos_t - P::gravity;
        T alpha = u_side * P::side_torque;

        next_state.vx = math::clamp(device.math, state.vx + ax * P::dt, (T)-3.0, (T)3.0);
        next_state.vy = math::clamp(device.math, state.vy + ay * P::dt, (T)-3.0, (T)3.0);
        next_state.x = state.x + next_state.vx * P::dt;
        next_state.y = state.y + next_state.vy * P::dt;
        next_state.omega = math::clamp(device.math, state.omega + alpha * P::dt, (T)-4.0, (T)4.0);
        next_state.theta = state.theta + next_state.omega * P::dt;
        next_state.landed = false;
        next_state.crashed = false;

        // Ground contact check at lunar surface y <= 0.0
        if (next_state.y <= (T)0.0) {
            next_state.y = (T)0.0;
            bool on_pad = math::abs(device.math, next_state.x) <= P::pad_half_width;
            bool soft_touchdown = (math::abs(device.math, next_state.vx) < (T)0.45) &&
                                  (math::abs(device.math, next_state.vy) < (T)0.55) &&
                                  (math::abs(device.math, next_state.theta) < (T)0.30);
            if (on_pad && soft_touchdown) {
                next_state.landed = true;
            } else {
                next_state.crashed = true;
            }
        } else if (math::abs(device.math, next_state.x) > P::x_limit ||
                   next_state.y > P::y_ceil ||
                   math::abs(device.math, next_state.theta) > P::angle_limit) {
            next_state.crashed = true;
        }

        return P::dt;
    }

    /**
     * @brief Potential-based LunarLander reward (just like Gymnasium LunarLander-v2!) + live fuel penalty slider.
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::LunarLander<SPEC>&,
                                   typename rl::environments::LunarLander<SPEC>::Parameters&,
                                   const typename rl::environments::LunarLander<SPEC>::State& state,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::LunarLander<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        const auto& rs = studio::reward_shaping();

        auto potential = [&](const auto& s) {
            T dist = math::sqrt(device.math, s.x * s.x + s.y * s.y);
            T speed = math::sqrt(device.math, s.vx * s.vx + s.vy * s.vy);
            return -(T)60.0 * dist - (T)30.0 * speed - (T)40.0 * math::abs(device.math, s.theta);
        };

        T u_main = (math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0) + (T)1.0) * (T)0.5;
        T u_side = math::abs(device.math, math::clamp(device.math, get(action, 0, 1), (T)-1.0, (T)1.0));

        T r = (potential(next_state) - potential(state))
              - static_cast<T>(rs.lander_fuel_weight) * (u_main + (T)0.2 * u_side);

        if (next_state.landed) {
            r += (T)100.0; // Safe touchdown on lunar pad!
        } else if (next_state.crashed) {
            r -= (T)100.0; // Crashed or flew out of bounds!
        }
        return r;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::LunarLander<SPEC>&,
                        const typename rl::environments::LunarLander<SPEC>::Parameters&,
                        const typename rl::environments::LunarLander<SPEC>::State& state,
                        const rl::environments::lunar_lander::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 6);
        using T = typename SPEC::T;
        set(observation, 0, 0, state.x / (T)1.3);
        set(observation, 0, 1, state.y / (T)1.5);
        set(observation, 0, 2, state.vx / (T)2.0);
        set(observation, 0, 3, state.vy / (T)2.0);
        set(observation, 0, 4, state.theta / (T)1.2);
        set(observation, 0, 5, state.omega / (T)3.0);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE&, const rl::environments::LunarLander<SPEC>&,
                           typename rl::environments::LunarLander<SPEC>::Parameters&,
                           const typename rl::environments::LunarLander<SPEC>::State state, RNG&) {
        return state.landed || state.crashed;
    }

} // namespace rl_tools
