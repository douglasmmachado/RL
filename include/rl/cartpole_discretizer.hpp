/**
 * @file cartpole_discretizer.hpp
 * @brief Maps a 4D continuous CartPoleState into a single discrete integer state ID
 *        for tabular RL algorithms (Q-Learning / SARSA).
 */

#pragma once

#include "rl/cartpole.hpp"
#include <algorithm>
#include <array>

namespace rl {

/**
 * @brief Uniform bin discretizer for the 4D continuous CartPole state space.
 *
 * Bins each of {x, x_dot, theta, theta_dot} within [low[i], high[i]] into `bins[i]`
 * buckets and flattens the 4D bin tuple (b0, b1, b2, b3) into a 1D integer index.
 */
class CartPoleDiscretizer {
public:
    /// @brief Number of discrete buckets per dimension: {x, x_dot, theta, theta_dot}.
    static constexpr std::array<int, 4> bins = {6, 6, 12, 12};

    /// @brief Lower bounds for each state dimension.
    static constexpr std::array<double, 4> low = {-2.4, -3.0, -0.21, -3.5};

    /// @brief Upper bounds for each state dimension.
    static constexpr std::array<double, 4> high = {2.4, 3.0, 0.21, 3.5};

    /**
     * @brief Returns the total number of discrete states (6 * 6 * 12 * 12 = 5184).
     * @return Total number of discrete states.
     */
    [[nodiscard]] static constexpr int num_states() noexcept {
        return bins[0] * bins[1] * bins[2] * bins[3];
    }

    /**
     * @brief Maps a continuous CartPoleState {x, x_dot, theta, theta_dot} into
     *        a single discrete state ID in [0, num_states() - 1].
     *
     * @param state Continuous 4D CartPole state.
     * @return Flattened discrete integer state ID.
     */
    [[nodiscard]] static int discretize(const CartPoleState& state) noexcept {
        int state_id = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            double ratio = (state[i] - low[i]) / (high[i] - low[i]);
            ratio = std::clamp(ratio, 0.0, 1.0);
            int bin_idx = static_cast<int>(ratio * bins[i]);
            bin_idx = std::clamp(bin_idx, 0, bins[i] - 1);
            state_id = state_id * bins[i] + bin_idx;
        }
        return state_id;
    }
};

} // namespace rl
