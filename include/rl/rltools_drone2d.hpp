/**
 * @file rltools_drone2d.hpp
 * @brief RLtools-compatible 2D Planar Quadrotor Drone Hover environment for SAC, PPO, and TD3.
 *
 * State:       {x, y, vx, vy, theta, omega} (position, velocity, pitch angle, angular velocity)
 * Action:      2D continuous motor thrusts (left_rotor, right_rotor) in [-1, +1] mapped to [0, 1.5*mg]
 * Goal:        Hover smoothly at (0.0, 0.0) upright (theta = 0) against gravity!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>
#include "rl/reward_shaping.hpp"

namespace rl_tools::rl::environments::drone_2d {

    /**
     * @brief Physical constants for the 2D planar quadrotor drone.
     */
    template <typename T>
    struct DefaultParameters {
        constexpr static T gravity = 9.81;
        constexpr static T mass = 1.0;
        constexpr static T arm_length = 0.25;
        constexpr static T inertia = 0.05;
        constexpr static T dt = 0.02;
        constexpr static T max_thrust_per_rotor = 9.0; // Hover is ~4.905 N per rotor
        constexpr static T pos_limit = 1.5;
        constexpr static T angle_limit = 1.2; // ~68 degrees flip termination
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    /**
     * @brief 6D planar quadrotor state {x, y, vx, vy, theta, omega}.
     */
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 6;
        T x, y;
        T vx, vy;
        T theta, omega;
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 6;
    };

} // namespace rl_tools::rl::environments::drone_2d

namespace rl_tools::rl::environments {

    /**
     * @brief RLtools 2D Planar Quadrotor Drone Hover environment.
     */
    template <typename T_SPEC>
    struct Drone2D : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = drone_2d::State<T, TI>;
        using Observation = drone_2d::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;          ///< Left and right rotor throttle in [-1, +1].
        static constexpr TI EPISODE_STEP_LIMIT = 250;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::Drone2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::Drone2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::Drone2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::Drone2D<SPEC>&,
                                   typename rl::environments::Drone2D<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::Drone2D<SPEC>&,
                                          typename rl::environments::Drone2D<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::Drone2D<SPEC>&,
                              typename rl::environments::Drone2D<SPEC>::Parameters&,
                              typename rl::environments::Drone2D<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.x = (T)-0.6;
        state.y = (T)-0.4;
        state.vx = (T)0.0;
        state.vy = (T)0.0;
        state.theta = (T)0.15;
        state.omega = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::Drone2D<SPEC>&,
                                     typename rl::environments::Drone2D<SPEC>::Parameters&,
                                     typename rl::environments::Drone2D<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.7, (T)0.7, rng);
        state.y = random::uniform_real_distribution(device.random, (T)-0.6, (T)0.5, rng);
        state.vx = random::uniform_real_distribution(device.random, (T)-0.3, (T)0.3, rng);
        state.vy = random::uniform_real_distribution(device.random, (T)-0.3, (T)0.3, rng);
        state.theta = random::uniform_real_distribution(device.random, (T)-0.25, (T)0.25, rng);
        state.omega = random::uniform_real_distribution(device.random, (T)-0.3, (T)0.3, rng);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::Drone2D<SPEC>&,
                          typename rl::environments::Drone2D<SPEC>::Parameters&,
                          const typename rl::environments::Drone2D<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::Drone2D<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T u_l = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T u_r = math::clamp(device.math, get(action, 0, 1), (T)-1.0, (T)1.0);

        // Map [-1, +1] -> [0, max_thrust_per_rotor]
        T t_l = (u_l + (T)1.0) * (T)0.5 * P::max_thrust_per_rotor;
        T t_r = (u_r + (T)1.0) * (T)0.5 * P::max_thrust_per_rotor;
        T total_thrust = t_l + t_r;
        T torque = (t_r - t_l) * P::arm_length;

        T ax = -total_thrust * math::sin(device.math, state.theta) / P::mass;
        T ay = (total_thrust * math::cos(device.math, state.theta)) / P::mass - P::gravity;
        T alpha = torque / P::inertia;

        next_state.vx = math::clamp(device.math, state.vx + ax * P::dt, (T)-4.0, (T)4.0);
        next_state.vy = math::clamp(device.math, state.vy + ay * P::dt, (T)-4.0, (T)4.0);
        next_state.x = state.x + next_state.vx * P::dt;
        next_state.y = state.y + next_state.vy * P::dt;
        next_state.omega = math::clamp(device.math, state.omega + alpha * P::dt, (T)-6.0, (T)6.0);
        next_state.theta = state.theta + next_state.omega * P::dt;
        return P::dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::Drone2D<SPEC>&,
                                   typename rl::environments::Drone2D<SPEC>::Parameters&,
                                   const typename rl::environments::Drone2D<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::Drone2D<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        const auto& rs = studio::reward_shaping();
        T dist = math::sqrt(device.math, next_state.x * next_state.x + next_state.y * next_state.y);
        T u_l = get(action, 0, 0);
        T u_r = get(action, 0, 1);
        T r = (T)1.5 - (T)1.2 * dist
              - static_cast<T>(rs.drone_tilt_weight) * math::abs(device.math, next_state.theta)
              - static_cast<T>(rs.drone_vel_weight) * (next_state.vx * next_state.vx + next_state.vy * next_state.vy)
              - static_cast<T>(rs.drone_effort_weight) * (u_l * u_l + u_r * u_r);
        if (math::abs(device.math, next_state.x) > P::pos_limit ||
            math::abs(device.math, next_state.y) > P::pos_limit ||
            math::abs(device.math, next_state.theta) > P::angle_limit) {
            r -= (T)10.0;
        }
        return r;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::Drone2D<SPEC>&,
                        const typename rl::environments::Drone2D<SPEC>::Parameters&,
                        const typename rl::environments::Drone2D<SPEC>::State& state,
                        const rl::environments::drone_2d::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 6);
        using T = typename SPEC::T;
        set(observation, 0, 0, state.x / (T)1.5);
        set(observation, 0, 1, state.y / (T)1.5);
        set(observation, 0, 2, state.vx / (T)4.0);
        set(observation, 0, 3, state.vy / (T)4.0);
        set(observation, 0, 4, state.theta / (T)1.2);
        set(observation, 0, 5, state.omega / (T)6.0);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::Drone2D<SPEC>&,
                           typename rl::environments::Drone2D<SPEC>::Parameters&,
                           const typename rl::environments::Drone2D<SPEC>::State state, RNG&) {
        using P = typename SPEC::PARAMETERS;
        return (math::abs(device.math, state.x) > P::pos_limit) ||
               (math::abs(device.math, state.y) > P::pos_limit) ||
               (math::abs(device.math, state.theta) > P::angle_limit);
    }

} // namespace rl_tools
