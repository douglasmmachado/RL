/**
 * @file rltools_double_cartpole.hpp
 * @brief RLtools-compatible Inverted Double Pendulum on a Cart (hard nonlinear continuous control!).
 *
 * Two stacked poles attached to a moving cart. The agent applies horizontal force u in [-1, 1]
 * to keep BOTH poles balanced upright simultaneously!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>

namespace rl_tools::rl::environments::double_cartpole {

    template <typename T>
    struct DefaultParameters {
        constexpr static T g = 9.81;
        constexpr static T m_c = 1.0;
        constexpr static T m1 = 0.1;
        constexpr static T m2 = 0.1;
        constexpr static T l1 = 0.5;
        constexpr static T l2 = 0.5;
        constexpr static T force_mag = 14.0;
        constexpr static T dt = 0.02;
        constexpr static T x_limit = 2.4;
        constexpr static T angle_limit = 0.45; // ~25 deg per pole
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
        T x, x_dot;
        T theta1, theta1_dot;
        T theta2, theta2_dot;
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 6;
    };

} // namespace rl_tools::rl::environments::double_cartpole

namespace rl_tools::rl::environments {

    template <typename T_SPEC>
    struct DoubleCartPole : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = double_cartpole::State<T, TI>;
        using Observation = double_cartpole::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 1;
        static constexpr TI EPISODE_STEP_LIMIT = 400;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&) {}
    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&,
                                   typename rl::environments::DoubleCartPole<SPEC>::Parameters&) {}
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&,
                                          typename rl::environments::DoubleCartPole<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&,
                              typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                              typename rl::environments::DoubleCartPole<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.x = (T)0.0; state.x_dot = (T)0.0;
        state.theta1 = (T)0.04; state.theta1_dot = (T)0.0;
        state.theta2 = (T)-0.03; state.theta2_dot = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::DoubleCartPole<SPEC>&,
                                     typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                                     typename rl::environments::DoubleCartPole<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.x = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.x_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta1 = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta1_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta2 = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
        state.theta2_dot = random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::DoubleCartPole<SPEC>&,
                          typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                          const typename rl::environments::DoubleCartPole<SPEC>::State& s,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::DoubleCartPole<SPEC>::State& ns, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T f = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0) * P::force_mag;
        // Coupled Euler-Lagrange approximation for inverted double pendulum on cart (2 substeps for stability)
        T x = s.x, xd = s.x_dot, t1 = s.theta1, t1d = s.theta1_dot, t2 = s.theta2, t2d = s.theta2_dot;
        T sub_dt = P::dt * (T)0.5;
        for (int sub = 0; sub < 2; ++sub) {
            T s1 = math::sin(device.math, t1), c1 = math::cos(device.math, t1);
            T s2 = math::sin(device.math, t2), c2 = math::cos(device.math, t2);
            T s12 = math::sin(device.math, t1 - t2), c12 = math::cos(device.math, t1 - t2);

            T xdd = (f + P::m1 * P::l1 * t1d * t1d * s1 + P::m2 * P::l2 * t2d * t2d * s2) / (P::m_c + P::m1 + P::m2);
            T t1dd = ((P::m1 + P::m2) * P::g * s1 - (P::m1 + P::m2) * xdd * c1 - P::m2 * P::l2 * t2d * t2d * s12)
                   / ((P::m1 + P::m2) * P::l1 * (T)0.85);
            T t2dd = (P::g * s2 - xdd * c2 - P::l1 * t1dd * c12 + P::l1 * t1d * t1d * s12) / (P::l2 * (T)0.85);

            xd += xdd * sub_dt; x += xd * sub_dt;
            t1d += t1dd * sub_dt; t1 += t1d * sub_dt;
            t2d += t2dd * sub_dt; t2 += t2d * sub_dt;
        }
        ns.x = x; ns.x_dot = xd;
        ns.theta1 = t1; ns.theta1_dot = t1d;
        ns.theta2 = t2; ns.theta2_dot = t2d;
        return P::dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::DoubleCartPole<SPEC>&,
                                   typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                                   const typename rl::environments::DoubleCartPole<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>&,
                                   const typename rl::environments::DoubleCartPole<SPEC>::State& ns, RNG&) {
        using T = typename SPEC::T;
        return (T)1.0 - (T)0.1 * math::abs(device.math, ns.x)
               - (T)0.3 * (math::abs(device.math, ns.theta1) + math::abs(device.math, ns.theta2));
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::DoubleCartPole<SPEC>&,
                        const typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                        const typename rl::environments::DoubleCartPole<SPEC>::State& s,
                        const rl::environments::double_cartpole::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& obs, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 6);
        using T = typename SPEC::T;
        set(obs, 0, 0, s.x / (T)2.4);
        set(obs, 0, 1, s.x_dot / (T)3.0);
        set(obs, 0, 2, s.theta1 / (T)0.45);
        set(obs, 0, 3, s.theta1_dot / (T)4.0);
        set(obs, 0, 4, s.theta2 / (T)0.45);
        set(obs, 0, 5, s.theta2_dot / (T)4.0);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::DoubleCartPole<SPEC>&,
                           typename rl::environments::DoubleCartPole<SPEC>::Parameters&,
                           const typename rl::environments::DoubleCartPole<SPEC>::State s, RNG&) {
        using P = typename SPEC::PARAMETERS;
        return math::abs(device.math, s.x) > P::x_limit ||
               math::abs(device.math, s.theta1) > P::angle_limit ||
               math::abs(device.math, s.theta2) > P::angle_limit;
    }

} // namespace rl_tools
