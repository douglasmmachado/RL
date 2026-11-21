/**
 * @file rltools_reacher.hpp
 * @brief RLtools-compatible 2-DOF Robotic Arm Reacher environment (MuJoCo Reacher-v4 inspired).
 *
 * State:       {q1, q2, q1_dot, q2_dot, target_x, target_y}
 * Observation: {cos(q1), sin(q1), cos(q2), sin(q2), q1_dot/8, q2_dot/8, tip_x - target_x, tip_y - target_y} (8D)
 * Action:      2D continuous joint torques {tau1, tau2} in [-1, +1]
 * Goal:        Move the robotic fingertip to the glowing green target dot smoothly!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>

namespace rl_tools::rl::environments::reacher_2d {

    template <typename T>
    struct DefaultParameters {
        constexpr static T l1 = 0.5;
        constexpr static T l2 = 0.5;
        constexpr static T dt = 0.04;
        constexpr static T max_torque = 6.0;
        constexpr static T damping = 0.35;
        constexpr static T max_vel = 8.0;
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
        static constexpr TI DIM = 6;
        T q1, q2, q1_dot, q2_dot;
        T target_x, target_y;
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 8;
    };

} // namespace rl_tools::rl::environments::reacher_2d

namespace rl_tools::rl::environments {

    template <typename T_SPEC>
    struct Reacher2D : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = reacher_2d::State<T, TI>;
        using Observation = reacher_2d::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 100;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::Reacher2D<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::Reacher2D<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::Reacher2D<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::Reacher2D<SPEC>&,
                                   typename rl::environments::Reacher2D<SPEC>::Parameters&) {}
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::Reacher2D<SPEC>&,
                                          typename rl::environments::Reacher2D<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::Reacher2D<SPEC>&,
                              typename rl::environments::Reacher2D<SPEC>::Parameters&,
                              typename rl::environments::Reacher2D<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.q1 = (T)-1.2; state.q2 = (T)0.6;
        state.q1_dot = (T)0.0; state.q2_dot = (T)0.0;
        state.target_x = (T)0.55; state.target_y = (T)0.45;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::Reacher2D<SPEC>&,
                                     typename rl::environments::Reacher2D<SPEC>::Parameters&,
                                     typename rl::environments::Reacher2D<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.q1 = random::uniform_real_distribution(device.random, (T)-3.14, (T)3.14, rng);
        state.q2 = random::uniform_real_distribution(device.random, (T)-2.5, (T)2.5, rng);
        state.q1_dot = (T)0.0;
        state.q2_dot = (T)0.0;
        T ang = random::uniform_real_distribution(device.random, (T)-3.14, (T)3.14, rng);
        T rad = random::uniform_real_distribution(device.random, (T)0.25, (T)0.88, rng);
        state.target_x = rad * math::cos(device.math, ang);
        state.target_y = rad * math::sin(device.math, ang);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::Reacher2D<SPEC>&,
                          typename rl::environments::Reacher2D<SPEC>::Parameters&,
                          const typename rl::environments::Reacher2D<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::Reacher2D<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T u1 = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0) * P::max_torque;
        T u2 = math::clamp(device.math, get(action, 0, 1), (T)-1.0, (T)1.0) * P::max_torque;

        next_state.q1_dot = math::clamp(device.math, state.q1_dot + (u1 - P::damping * state.q1_dot) * P::dt, -P::max_vel, P::max_vel);
        next_state.q2_dot = math::clamp(device.math, state.q2_dot + (u2 - P::damping * state.q2_dot) * P::dt, -P::max_vel, P::max_vel);
        next_state.q1 = state.q1 + next_state.q1_dot * P::dt;
        next_state.q2 = state.q2 + next_state.q2_dot * P::dt;
        next_state.target_x = state.target_x;
        next_state.target_y = state.target_y;
        return P::dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::Reacher2D<SPEC>&,
                                   typename rl::environments::Reacher2D<SPEC>::Parameters&,
                                   const typename rl::environments::Reacher2D<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::Reacher2D<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;
        T tip_x = P::l1 * math::cos(device.math, next_state.q1) + P::l2 * math::cos(device.math, next_state.q1 + next_state.q2);
        T tip_y = P::l1 * math::sin(device.math, next_state.q1) + P::l2 * math::sin(device.math, next_state.q1 + next_state.q2);
        T dx = tip_x - next_state.target_x;
        T dy = tip_y - next_state.target_y;
        T dist = math::sqrt(device.math, dx * dx + dy * dy);
        T u1 = get(action, 0, 0), u2 = get(action, 0, 1);
        return -dist - (T)0.05 * (u1 * u1 + u2 * u2);
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE& device, const rl::environments::Reacher2D<SPEC>&,
                        const typename rl::environments::Reacher2D<SPEC>::Parameters&,
                        const typename rl::environments::Reacher2D<SPEC>::State& state,
                        const rl::environments::reacher_2d::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 8);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;
        T tip_x = P::l1 * math::cos(device.math, state.q1) + P::l2 * math::cos(device.math, state.q1 + state.q2);
        T tip_y = P::l1 * math::sin(device.math, state.q1) + P::l2 * math::sin(device.math, state.q1 + state.q2);
        set(observation, 0, 0, math::cos(device.math, state.q1));
        set(observation, 0, 1, math::sin(device.math, state.q1));
        set(observation, 0, 2, math::cos(device.math, state.q2));
        set(observation, 0, 3, math::sin(device.math, state.q2));
        set(observation, 0, 4, state.q1_dot / P::max_vel);
        set(observation, 0, 5, state.q2_dot / P::max_vel);
        set(observation, 0, 6, tip_x - state.target_x);
        set(observation, 0, 7, tip_y - state.target_y);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE&, const rl::environments::Reacher2D<SPEC>&,
                           typename rl::environments::Reacher2D<SPEC>::Parameters&,
                           const typename rl::environments::Reacher2D<SPEC>::State, RNG&) {
        return false;
    }

} // namespace rl_tools
