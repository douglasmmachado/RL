/**
 * @file rltools_mountaincar.hpp
 * @brief RLtools-compatible MountainCarContinuous-v0 environment for SAC, PPO, and TD3.
 *
 * State:       {position, velocity}
 * Action:      1D continuous engine force u in [-1.0, +1.0]
 * Goal:        Reach position >= 0.45 (+100.0 reward + episode ends)
 * Reward:      Shaped potential + energy penalty (-0.1 * u^2) + 100.0 on reaching the hilltop
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>
#include "rl/reward_shaping.hpp"

namespace rl_tools::rl::environments::mountain_car {

    /**
     * @brief Compile-time physical constants matching Gymnasium MountainCarContinuous-v0.
     * @tparam T Floating-point scalar type.
     */
    template <typename T>
    struct DefaultParameters {
        constexpr static T min_position = -1.2;
        constexpr static T max_position = 0.6;
        constexpr static T max_speed = 0.07;
        constexpr static T goal_position = 0.45;
        constexpr static T power = 0.0015;
        constexpr static T gravity = 0.0025;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    /**
     * @brief Simulator state storing car horizontal position and velocity.
     */
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 2;
        T position;
        T velocity;
    };

    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 2;
    };

} // namespace rl_tools::rl::environments::mountain_car

namespace rl_tools::rl::environments {

    /**
     * @brief RLtools Continuous MountainCar environment.
     */
    template <typename T_SPEC>
    struct MountainCar : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = mountain_car::State<T, TI>;
        using Observation = mountain_car::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 1;
        static constexpr TI EPISODE_STEP_LIMIT = 200;
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::MountainCar<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::MountainCar<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::MountainCar<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::MountainCar<SPEC>&,
                                   typename rl::environments::MountainCar<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::MountainCar<SPEC>&,
                                          typename rl::environments::MountainCar<SPEC>::Parameters&, RNG&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::MountainCar<SPEC>&,
                              typename rl::environments::MountainCar<SPEC>::Parameters&,
                              typename rl::environments::MountainCar<SPEC>::State& state) {
        using T = typename SPEC::T;
        state.position = (T)-0.5;
        state.velocity = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::MountainCar<SPEC>&,
                                     typename rl::environments::MountainCar<SPEC>::Parameters&,
                                     typename rl::environments::MountainCar<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        state.position = random::uniform_real_distribution(device.random, (T)-0.6, (T)-0.4, rng);
        state.velocity = (T)0.0;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::MountainCar<SPEC>&,
                          typename rl::environments::MountainCar<SPEC>::Parameters&,
                          const typename rl::environments::MountainCar<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::MountainCar<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 1);
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        T force = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T vel = state.velocity + force * P::power - math::cos(device.math, (T)3.0 * state.position) * P::gravity;
        vel = math::clamp(device.math, vel, -P::max_speed, P::max_speed);
        T pos = state.position + vel;
        pos = math::clamp(device.math, pos, P::min_position, P::max_position);
        if (pos <= P::min_position && vel < (T)0.0) {
            vel = (T)0.0;
        }
        next_state.position = pos;
        next_state.velocity = vel;
        return (T)1.0;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::MountainCar<SPEC>&,
                                   typename rl::environments::MountainCar<SPEC>::Parameters&,
                                   const typename rl::environments::MountainCar<SPEC>::State& state,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::MountainCar<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        using P = typename SPEC::PARAMETERS;

        const auto& rs = studio::reward_shaping();
        T u = get(action, 0, 0);
        T r = -static_cast<T>(rs.mcar_effort_weight) * (u * u);
        if (rs.mcar_dense_shaping) {
            T height_gain = math::sin(device.math, (T)3.0 * next_state.position) - math::sin(device.math, (T)3.0 * state.position);
            r += (T)2.0 * height_gain + (T)10.0 * math::abs(device.math, next_state.velocity);
        }
        if (next_state.position >= P::goal_position) {
            r += (T)100.0;
        }
        return r;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::MountainCar<SPEC>&,
                        const typename rl::environments::MountainCar<SPEC>::Parameters&,
                        const typename rl::environments::MountainCar<SPEC>::State& state,
                        const rl::environments::mountain_car::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 2);
        using T = typename SPEC::T;
        set(observation, 0, 0, (state.position + (T)0.3) / (T)0.9);
        set(observation, 0, 1, state.velocity / (T)0.07);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE&, const rl::environments::MountainCar<SPEC>&,
                           typename rl::environments::MountainCar<SPEC>::Parameters&,
                           const typename rl::environments::MountainCar<SPEC>::State state, RNG&) {
        using P = typename SPEC::PARAMETERS;
        return state.position >= P::goal_position;
    }

} // namespace rl_tools
