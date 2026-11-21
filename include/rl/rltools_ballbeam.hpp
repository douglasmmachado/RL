/**
 * @file rltools_ballbeam.hpp
 * @brief RLtools-compatible Classic Control Engineering "Ball-and-Beam" environment!
 *
 * A steel ball rolls freely along a tilting beam of length 2.0m (x in [-1.0, +1.0]).
 * The agent applies angular acceleration to tilt the beam angle theta and balance
 * the ball at the center target x = 0.0!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>

namespace rl_tools::rl::environments::ball_beam {

    template <typename T>
    struct DefaultParameters {
        constexpr static T gravity = 9.81;
        constexpr static T dt = 0.02;
        constexpr static T beam_half_len = 1.0;
        constexpr static T max_angle = 0.55;      ///< ~31 deg max beam tilt
        constexpr static T max_omega = 3.0;
        constexpr static T max_alpha = 8.0;       ///< Control angular acceleration [rad/s^2]
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 4;
        T x, x_dot;         ///< Ball position & velocity along the beam
        T theta, theta_dot; ///< Beam tilt angle & angular velocity
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 4;
    };

} // namespace rl_tools::rl::environments::ball_beam

namespace rl_tools::rl::environments {

    template <typename T_SPEC>
    struct BallBeam : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = ball_beam::State<T, TI>;
        using Observation = ball_beam::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 1;
        static constexpr TI EPISODE_STEP_LIMIT = 300;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::BallBeam<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::BallBeam<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::BallBeam<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::BallBeam<SPEC>&,
                                   typename rl::environments::BallBeam<SPEC>::Parameters&) {}
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::BallBeam<SPEC>&,
                                          typename rl::environments::BallBeam<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::BallBeam<SPEC>&,
                              typename rl::environments::BallBeam<SPEC>::Parameters&,
                              typename rl::environments::BallBeam<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.x = (T)-0.55; state.x_dot = (T)0.0;
        state.theta = (T)0.05; state.theta_dot = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::BallBeam<SPEC>&,
                                     typename rl::environments::BallBeam<SPEC>::Parameters&,
                                     typename rl::environments::BallBeam<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.7, (T)0.7, rng);
        state.x_dot = random::uniform_real_distribution(device.random, (T)-0.3, (T)0.3, rng);
        state.theta = random::uniform_real_distribution(device.random, (T)-0.15, (T)0.15, rng);
        state.theta_dot = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::BallBeam<SPEC>&,
                          typename rl::environments::BallBeam<SPEC>::Parameters&,
                          const typename rl::environments::BallBeam<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::BallBeam<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T u = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        // Solid sphere rolling acceleration: (5/7) * g * sin(theta) - (5/7) * x * theta_dot^2
        T ball_acc = (T)(5.0 / 7.0) * P::gravity * math::sin(device.math, state.theta)
                   - (T)(5.0 / 7.0) * state.x * state.theta_dot * state.theta_dot;

        next_state.x_dot = math::clamp(device.math, state.x_dot + ball_acc * P::dt, (T)-4.0, (T)4.0);
        next_state.x = state.x + next_state.x_dot * P::dt;

        next_state.theta_dot = math::clamp(device.math, state.theta_dot + u * P::max_alpha * P::dt, -P::max_omega, P::max_omega);
        next_state.theta = math::clamp(device.math, state.theta + next_state.theta_dot * P::dt, -P::max_angle, P::max_angle);
        if (math::abs(device.math, next_state.theta) >= P::max_angle) next_state.theta_dot = (T)0.0;
        return P::dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::BallBeam<SPEC>&,
                                   typename rl::environments::BallBeam<SPEC>::Parameters&,
                                   const typename rl::environments::BallBeam<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::BallBeam<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;
        T u = get(action, 0, 0);
        T r = (T)1.0 - (T)1.2 * math::abs(device.math, next_state.x)
              - (T)0.15 * math::abs(device.math, next_state.x_dot)
              - (T)0.05 * (u * u);
        if (math::abs(device.math, next_state.x) > P::beam_half_len) r -= (T)15.0;
        return r;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::BallBeam<SPEC>&,
                        const typename rl::environments::BallBeam<SPEC>::Parameters&,
                        const typename rl::environments::BallBeam<SPEC>::State& state,
                        const rl::environments::ball_beam::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 4);
        using T = typename SPEC::T;
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.x_dot / (T)3.0);
        set(observation, 0, 2, state.theta / (T)0.55);
        set(observation, 0, 3, state.theta_dot / (T)3.0);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::BallBeam<SPEC>&,
                           typename rl::environments::BallBeam<SPEC>::Parameters&,
                           const typename rl::environments::BallBeam<SPEC>::State state, RNG&) {
        using P = typename SPEC::PARAMETERS;
        return math::abs(device.math, state.x) > P::beam_half_len;
    }

} // namespace rl_tools
