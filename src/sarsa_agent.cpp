/**
 * @file sarsa_agent.cpp
 * @brief Implementation of the on-policy SARSA update rule.
 */

#include "rl/sarsa_agent.hpp"
#include <cmath>

namespace rl {

double SarsaAgent::update(int state, GridAction action, double reward,
                          int next_state, GridAction next_action, bool terminated) {
    int action_idx = static_cast<int>(action);

    double next_q = 0.0;
    if (!terminated) {
        next_q = q_value(next_state, static_cast<int>(next_action));
    }

    double target = reward + gamma_ * next_q;
    double current_q = q_value(state, action_idx);
    double td_error = target - current_q;

    q_entry(state, action_idx) = current_q + alpha_ * td_error;

    return std::abs(td_error);
}

} // namespace rl
