/**
 * @file test_q_learning.cpp
 * @brief Unit tests for the tabular QLearningAgent class and its convergence on GridWorld.
 */

#include "rl/gridworld.hpp"
#include "rl/q_learning_agent.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief Checks if two floating-point values are equal within a tolerance.
 * @param a First value.
 * @param b Second value.
 * @param eps Tolerance threshold.
 * @return True if |a - b| < eps.
 */
bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

/**
 * @brief Tests a single Q-Learning Bellman update on a non-terminal and terminal transition.
 */
void test_single_bellman_update() {
    // 4 states, 4 actions, alpha = 0.5, gamma = 0.9, epsilon = 0.0
    rl::QLearningAgent agent(4, 4, 0.5, 0.9, 0.0);

    // Initially all Q(s, a) == 0.0
    assert(approx_equal(agent.q_value(0, 1), 0.0));

    // 1. Terminal transition: state 2 -> Right (idx 1) -> state 3 (Goal), reward = +10.0, terminated = true
    // Q(2, 1) <- 0.0 + 0.5 * (10.0 + 0.9 * 0.0 - 0.0) = 5.0
    agent.update(2, rl::GridAction::Right, 10.0, 3, true);
    assert(approx_equal(agent.q_value(2, 1), 5.0));

    // 2. Non-terminal transition into state 2:
    // state 1 -> Right (idx 1) -> state 2, reward = -0.1, terminated = false
    // max_{a'} Q(2, a') is Q(2, 1) == 5.0
    // target = -0.1 + 0.9 * 5.0 = 4.4
    // Q(1, 1) <- 0.0 + 0.5 * (4.4 - 0.0) = 2.2
    agent.update(1, rl::GridAction::Right, -0.1, 2, false);
    assert(approx_equal(agent.q_value(1, 1), 2.2));
    assert(agent.greedy_action(1) == rl::GridAction::Right);

    std::cout << "[PASS] test_single_bellman_update\n";
}

/**
 * @brief Tests that QLearningAgent learns an optimal trap-avoiding path on a 5x5 GridWorld.
 */
void test_gridworld_training_convergence() {
    rl::GridWorld env(5, 5);
    rl::QLearningAgent agent(env.num_states(), env.num_actions(), 0.2, 0.95, 0.2, 42);

    // Train for 500 episodes
    for (int ep = 0; ep < 500; ++ep) {
        int state = env.reset();
        while (true) {
            rl::GridAction action = agent.select_action(state);
            rl::StepResult<int> res = env.step(action);
            agent.update(state, action, res.reward, res.next_state, res.terminated);
            state = res.next_state;
            if (res.done()) {
                break;
            }
        }
    }

    // Evaluate greedy policy (no exploration) from start state (0, 0)
    int state = env.reset();
    double total_return = 0.0;
    int steps = 0;
    bool reached_goal = false;

    while (steps < 20) {
        rl::GridAction action = agent.greedy_action(state);
        rl::StepResult<int> res = env.step(action);
        total_return += res.reward;
        steps++;
        state = res.next_state;
        if (res.done()) {
            reached_goal = (res.reward > 0.0);
            break;
        }
    }

    // Shortest path on 5x5 grid from (0,0) to (4,4) takes 8 steps:
    // 7 normal steps (-0.7) + 1 goal step (+10.0) = +9.3 total return!
    assert(reached_goal);
    assert(steps == 8);
    assert(approx_equal(total_return, 9.3));

    std::cout << "[PASS] test_gridworld_training_convergence (Solved 5x5 grid in "
              << steps << " steps, Return = " << total_return << ")\n";
}

int main() {
    std::cout << "=== Running Q-Learning Unit Tests ===\n";
    test_single_bellman_update();
    test_gridworld_training_convergence();
    std::cout << "All Q-Learning tests passed!\n";
    return 0;
}
