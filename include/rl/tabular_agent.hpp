/**
 * @file tabular_agent.hpp
 * @brief Base class for tabular Reinforcement Learning agents (Q-Learning, SARSA, Expected SARSA).
 */

#pragma once

#include "rl/gridworld.hpp"
#include <random>
#include <string>
#include <vector>

namespace rl {

/**
 * @brief Base class managing a 1D flattened Q-table and epsilon-greedy exploration
 *        for discrete-state, discrete-action tabular RL algorithms.
 */
class TabularAgent {
public:
    /**
     * @brief Constructs a TabularAgent with a zero-initialized Q-table.
     * @param num_states Total number of discrete states in the environment.
     * @param num_actions Total number of discrete actions available in each state.
     * @param alpha Learning rate (step size) in (0, 1].
     * @param gamma Discount factor in [0, 1].
     * @param epsilon Exploration probability for epsilon-greedy action selection in [0, 1].
     * @param seed Random seed for reproducible exploration.
     */
    TabularAgent(int num_states, int num_actions, double alpha = 0.2,
                 double gamma = 0.95, double epsilon = 0.2, unsigned int seed = 42);

    virtual ~TabularAgent() = default;

    /**
     * @brief Returns the human-readable name of the algorithm (e.g., "Q-Learning", "SARSA").
     * @return Algorithm name string.
     */
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Selects an action using the epsilon-greedy policy.
     * @param state Current discrete state ID.
     * @return The chosen GridAction.
     */
    GridAction select_action(int state);

    /**
     * @brief Selects the purely greedy action (argmax_a Q(state, a)) without exploration.
     * @param state Current discrete state ID.
     * @return The greedy GridAction.
     */
    [[nodiscard]] GridAction greedy_action(int state) const;

    /**
     * @brief Reads the Q-value for a (state, action) pair.
     * @param state Discrete state ID.
     * @param action_idx Discrete action index in [0, num_actions_ - 1].
     * @return Current estimate of Q(state, action_idx).
     */
    [[nodiscard]] double q_value(int state, int action_idx) const {
        return q_table_[state * num_actions_ + action_idx];
    }

protected:
    int num_states_;               ///< Number of states in the environment.
    int num_actions_;              ///< Number of actions per state.
    double alpha_;                 ///< Learning rate.
    double gamma_;                 ///< Discount factor.
    double epsilon_;               ///< Exploration probability.
    std::vector<double> q_table_;  ///< Flattened 1D Q-table of size num_states_ * num_actions_.
    std::mt19937 rng_;             ///< Mersenne Twister pseudo-random number generator.

    /**
     * @brief Returns a mutable reference to Q(state, action_idx) in the flattened table.
     * @param state Discrete state ID.
     * @param action_idx Discrete action index in [0, num_actions_ - 1].
     * @return Reference to the Q-value entry.
     */
    double& q_entry(int state, int action_idx) {
        return q_table_[state * num_actions_ + action_idx];
    }
};

} // namespace rl
