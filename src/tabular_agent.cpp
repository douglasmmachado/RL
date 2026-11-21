/**
 * @file tabular_agent.cpp
 * @brief Shared implementation of Q-table management and epsilon-greedy exploration for TabularAgent.
 */

#include "rl/tabular_agent.hpp"

namespace rl {

TabularAgent::TabularAgent(int num_states, int num_actions, double alpha,
                           double gamma, double epsilon, unsigned int seed)
    : num_states_(num_states),
      num_actions_(num_actions),
      alpha_(alpha),
      gamma_(gamma),
      epsilon_(epsilon),
      q_table_(num_states * num_actions, 0.0),
      rng_(seed) {}

GridAction TabularAgent::greedy_action(int state) const {
    int best_action = 0;
    double best_value = q_value(state, 0);

    for (int a = 1; a < num_actions_; ++a) {
        if (q_value(state, a) > best_value) {
            best_value = q_value(state, a);
            best_action = a;
        }
    }
    return static_cast<GridAction>(best_action);
}

GridAction TabularAgent::select_action(int state) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<int> action_dist(0, num_actions_ - 1);

    if (prob_dist(rng_) < epsilon_) {
        int random_action = action_dist(rng_);
        return static_cast<GridAction>(random_action);
    }
    return greedy_action(state);
}

} // namespace rl
