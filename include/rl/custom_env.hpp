/**
 * @file custom_env.hpp
 * @brief RLtools-compatible continuous 2D navigation environment with runtime-configurable
 *        random Trap & Goal layouts (up to 4 circular Traps + dynamic Goal position).
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/environments/operations_generic.h>
#include "rl/reward_shaping.hpp"
#include <array>

namespace rl_tools::rl::environments::continuous_grid_2d {

    /**
     * @brief Runtime-configurable Arena Map shared by both SAC and PPO environments.
     *        Supports up to 4 circular Trap hazards and a configurable Goal position!
     */
    struct ArenaMap {
        static constexpr int MAX_TRAPS = 4;
        float start_x = -0.8f;
        float start_y = -0.8f;
        float target_x = 0.8f;
        float target_y = 0.8f;
        float goal_radius = 0.15f;
        int num_traps = 1;
        std::array<float, MAX_TRAPS> trap_x{0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, MAX_TRAPS> trap_y{0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, MAX_TRAPS> trap_r{0.25f, 0.20f, 0.20f, 0.20f};
    };

    /// @brief Active global map configuration read by ContinuousGrid2D physics & reward functions.
    inline ArenaMap& active_map() {
        static ArenaMap map;
        return map;
    }

    /**
     * @brief Compile-time parameters for the 2D continuous navigation arena.
     * @tparam T Floating-point scalar type (float or double).
     */
    template <typename T>
    struct DefaultParameters {
        constexpr static T max_step = 0.18;     ///< Max displacement per action axis per step.
        constexpr static T boundary = 1.0;      ///< Arena wall limits [-1, 1].
    };

    /**
     * @brief Specification bundling scalar type, index type, and parameters.
     */
    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    /**
     * @brief Simulator state storing continuous 2D coordinates (x, y).
     */
    template <typename T_T, typename T_TI>
    struct State {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI DIM = 2;
        T x; ///< Current X position in [-1, 1].
        T y; ///< Current Y position in [-1, 1].
    };

    /**
     * @brief Observation vector shape fed to Actor & Critic neural networks (x, y).
     */
    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 2;
    };

} // namespace rl_tools::rl::environments::continuous_grid_2d

namespace rl_tools::rl::environments {

    /**
     * @brief RLtools Continuous 2D Navigation Environment (with Goal and dynamic Traps).
     * @tparam T_SPEC Specification struct defining T, TI, and PARAMETERS.
     */
    template <typename T_SPEC>
    struct ContinuousGrid2D : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        using State = continuous_grid_2d::State<T, TI>;
        using Observation = continuous_grid_2d::Observation<TI>;
        using ObservationPrivileged = Observation;

        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;         ///< 2D continuous velocity (u_x, u_y).
        static constexpr TI EPISODE_STEP_LIMIT = 40; ///< Max steps per episode.
    };

} // namespace rl_tools::rl::environments

namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    static void malloc(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void free(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void init(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&) {}

    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&,
                                   typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&) {}

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&,
                                          typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&, RNG&) {}

    /**
     * @brief Sets the deterministic evaluation starting state from active_map().
     */
    template <typename DEVICE, typename SPEC>
    static void initial_state(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&,
                              typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                              typename rl::environments::ContinuousGrid2D<SPEC>::State& state) {
        using T = typename SPEC::T;
        const auto& m = rl::environments::continuous_grid_2d::active_map();
        state.x = static_cast<T>(m.start_x);
        state.y = static_cast<T>(m.start_y);
    }

    /**
     * @brief Samples a random training starting state near start_x, start_y (avoiding Traps).
     */
    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_state(DEVICE& device, const rl::environments::ContinuousGrid2D<SPEC>&,
                                     typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                                     typename rl::environments::ContinuousGrid2D<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        const auto& m = rl::environments::continuous_grid_2d::active_map();
        T sx = static_cast<T>(m.start_x);
        T sy = static_cast<T>(m.start_y);
        state.x = math::clamp(device.math, sx + random::uniform_real_distribution(device.random, (T)-0.15, (T)0.25, rng), (T)-0.95, (T)0.95);
        state.y = math::clamp(device.math, sy + random::uniform_real_distribution(device.random, (T)-0.15, (T)0.25, rng), (T)-0.95, (T)0.95);
    }

    /**
     * @brief Advances the continuous 2D agent by applying clipped velocity action (u_x, u_y).
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, const rl::environments::ContinuousGrid2D<SPEC>&,
                          typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                          const typename rl::environments::ContinuousGrid2D<SPEC>::State& state,
                          const Matrix<ACTION_SPEC>& action,
                          typename rl::environments::ContinuousGrid2D<SPEC>::State& next_state, RNG&) {
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;

        T ux = math::clamp(device.math, get(action, 0, 0), (T)-1.0, (T)1.0);
        T uy = math::clamp(device.math, get(action, 0, 1), (T)-1.0, (T)1.0);

        next_state.x = math::clamp(device.math, state.x + ux * PARAMS::max_step, -PARAMS::boundary, PARAMS::boundary);
        next_state.y = math::clamp(device.math, state.y + uy * PARAMS::max_step, -PARAMS::boundary, PARAMS::boundary);
        return (T)1.0;
    }

    /**
     * @brief Computes shaped distance reward plus +10.0 Goal bonus and -10.0 Trap penalty.
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::ContinuousGrid2D<SPEC>&,
                                   typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                                   const typename rl::environments::ContinuousGrid2D<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::ContinuousGrid2D<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        const auto& m = rl::environments::continuous_grid_2d::active_map();

        T dx_g = next_state.x - static_cast<T>(m.target_x);
        T dy_g = next_state.y - static_cast<T>(m.target_y);
        T dist_goal = math::sqrt(device.math, dx_g * dx_g + dy_g * dy_g);

        const auto& rs = studio::reward_shaping();
        T ux = get(action, 0, 0);
        T uy = get(action, 0, 1);

        T r = -static_cast<T>(rs.cgrid_effort_weight) * (ux * ux + uy * uy);
        if (rs.cgrid_dense_shaping) {
            r -= dist_goal * (T)0.5;
        }
        if (dist_goal <= static_cast<T>(m.goal_radius)) {
            return r + (T)10.0;
        }
        for (int i = 0; i < m.num_traps; ++i) {
            T dx_t = next_state.x - static_cast<T>(m.trap_x[i]);
            T dy_t = next_state.y - static_cast<T>(m.trap_y[i]);
            if (math::sqrt(device.math, dx_t * dx_t + dy_t * dy_t) <= static_cast<T>(m.trap_r[i])) {
                return r - (T)10.0;
            }
        }
        return r;
    }

    /**
     * @brief Writes the 2D observation vector [x, y] into the neural network input matrix.
     */
    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    static void observe(DEVICE&, const rl::environments::ContinuousGrid2D<SPEC>&,
                        const typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                        const typename rl::environments::ContinuousGrid2D<SPEC>::State& state,
                        const rl::environments::continuous_grid_2d::Observation<OBS_TYPE_SPEC>&,
                        Matrix<OBS_SPEC>& observation, RNG&) {
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == 2);
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
    }

    /**
     * @brief Returns true if the agent reached the Goal or entered any Trap hazard.
     */
    template <typename DEVICE, typename SPEC, typename RNG>
    static bool terminated(DEVICE& device, const rl::environments::ContinuousGrid2D<SPEC>&,
                           typename rl::environments::ContinuousGrid2D<SPEC>::Parameters&,
                           const typename rl::environments::ContinuousGrid2D<SPEC>::State state, RNG&) {
        using T = typename SPEC::T;
        const auto& m = rl::environments::continuous_grid_2d::active_map();

        T dx_g = state.x - static_cast<T>(m.target_x);
        T dy_g = state.y - static_cast<T>(m.target_y);
        if (math::sqrt(device.math, dx_g * dx_g + dy_g * dy_g) <= static_cast<T>(m.goal_radius)) {
            return true;
        }
        for (int i = 0; i < m.num_traps; ++i) {
            T dx_t = state.x - static_cast<T>(m.trap_x[i]);
            T dy_t = state.y - static_cast<T>(m.trap_y[i]);
            if (math::sqrt(device.math, dx_t * dx_t + dy_t * dy_t) <= static_cast<T>(m.trap_r[i])) {
                return true;
            }
        }
        return false;
    }

} // namespace rl_tools
