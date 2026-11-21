/**
 * @file q_learning_agent.cpp
 * @brief Implementation of the off-policy Q-Learning update rule.
 */

#include "rl/q_learning_agent.hpp"
#include <cmath>

namespace rl {

double QLearningAgent::update(int state, GridAction action, double reward,
                              int next_state, bool terminated) {
    int action_idx = static_cast<int>(action);

    double max_next_q = 0.0;
    if (!terminated) {
        int best_next_action = static_cast<int>(greedy_action(next_state));
        max_next_q = q_value(next_state, best_next_action);
    }

    double target = reward + gamma_ * max_next_q;
    double current_q = q_value(state, action_idx);
    double td_error = target - current_q;

    q_entry(state, action_idx) = current_q + alpha_ * td_error;

    return std::abs(td_error);
}

} // namespace rl
