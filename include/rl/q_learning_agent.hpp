/**
 * @file q_learning_agent.hpp
 * @brief Off-policy tabular Q-Learning agent inheriting shared Q-table logic from TabularAgent.
 */

#pragma once

#include "rl/tabular_agent.hpp"

namespace rl {

/**
 * @brief Off-policy Tabular Q-Learning agent for discrete state and action spaces.
 *
 * Inherits Q-table storage, `select_action`, and `greedy_action` from TabularAgent,
 * and implements the off-policy Bellman optimality update:
 *   Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_{a'} Q(s', a') - Q(s, a))
 */
class QLearningAgent : public TabularAgent {
public:
    /**
     * @brief Inherit the constructor directly from TabularAgent (C++11/20 inheriting constructors).
     */
    using TabularAgent::TabularAgent;

    /**
     * @brief Returns the human-readable algorithm name.
     * @return "Q-Learning".
     */
    [[nodiscard]] std::string name() const override { return "Q-Learning"; }

    /**
     * @brief Updates the Q-table using the off-policy Q-Learning Bellman update rule.
     * @param state Previous state ID (s).
     * @param action Action taken (a).
     * @param reward Reward received (r).
     * @param next_state Next state ID reached (s').
     * @param terminated True if next_state is a terminal state (Goal or Trap).
     * @return The absolute Temporal-Difference (TD) error |target - Q(s, a)|.
     */
    double update(int state, GridAction action, double reward, int next_state, bool terminated);
};

} // namespace rl
