/**
 * @file sarsa_agent.hpp
 * @brief On-policy tabular SARSA agent inheriting shared Q-table logic from TabularAgent.
 */

#pragma once

#include "rl/tabular_agent.hpp"

namespace rl {

/**
 * @brief On-policy Tabular SARSA (State-Action-Reward-State-Action) agent.
 *
 * Inherits Q-table storage, `select_action`, and `greedy_action` from TabularAgent,
 * and implements the on-policy SARSA update:
 *   Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
 */
class SarsaAgent : public TabularAgent {
public:
    /**
     * @brief Inherit the constructor directly from TabularAgent.
     */
    using TabularAgent::TabularAgent;

    /**
     * @brief Returns the human-readable algorithm name.
     * @return "SARSA".
     */
    [[nodiscard]] std::string name() const override { return "SARSA"; }

    /**
     * @brief Updates the Q-table using the on-policy SARSA update rule.
     * @param state Previous state ID (s).
     * @param action Action taken in state s (a).
     * @param reward Reward received (r).
     * @param next_state Next state ID reached (s').
     * @param next_action Next action chosen by the policy in next_state (a').
     * @param terminated True if next_state is a terminal state (Goal or Trap).
     * @return The absolute Temporal-Difference (TD) error |target - Q(s, a)|.
     */
    double update(int state, GridAction action, double reward,
                  int next_state, GridAction next_action, bool terminated);
};

} // namespace rl
