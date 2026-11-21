/**
 * @file test_sarsa.cpp
 * @brief Unit tests for the on-policy SarsaAgent class and comparison with Q-Learning.
 */

#include "rl/gridworld.hpp"
#include "rl/sarsa_agent.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief Checks if two floating-point values are equal within a tolerance.
 */
bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

/**
 * @brief Tests how SARSA updates Q(s, a) using the actual next action a' (even when a' is suboptimal!).
 */
void test_sarsa_on_policy_update() {
    rl::SarsaAgent agent(4, 4, 0.5, 0.9, 0.0);

    // Seed two distinct Q-values in state 2:
    // Action Right (1) has Q(2, Right) = +5.0 (good move)
    // Action Down  (2) has Q(2, Down)  = -5.0 (bad exploratory move toward a trap!)
    agent.update(2, rl::GridAction::Right, 10.0, 3, rl::GridAction::Up, true); // Q(2, Right) = 5.0
    agent.update(2, rl::GridAction::Down, -10.0, 3, rl::GridAction::Up, true); // Q(2, Down)  = -5.0

    assert(approx_equal(agent.q_value(2, 1), 5.0));
    assert(approx_equal(agent.q_value(2, 2), -5.0));

    // Now the agent transitions from state 1 -> Right -> state 2 (reward = -0.1),
    // and in state 2 its policy chooses next_action = GridAction::Down (the bad move!).
    // Unlike Q-Learning (which would use max_a' Q(2, a') = +5.0),
    // SARSA MUST use Q(2, Down) = -5.0:
    // target = -0.1 + 0.9 * (-5.0) = -4.6
    // Q(1, Right) <- 0.0 + 0.5 * (-4.6 - 0.0) = -2.3
    double td_err = agent.update(1, rl::GridAction::Right, -0.1, 2, rl::GridAction::Down, false);
    assert(approx_equal(agent.q_value(1, 1), -2.3));
    assert(approx_equal(td_err, 4.6));

    std::cout << "[PASS] test_sarsa_on_policy_update\n";
}

/**
 * @brief Tests that SarsaAgent converges to a valid goal-reaching policy on a 5x5 GridWorld.
 */
void test_sarsa_gridworld_convergence() {
    rl::GridWorld env(5, 5);
    rl::SarsaAgent agent(env.num_states(), env.num_actions(), 0.2, 0.95, 0.15, 42);

    // Note the SARSA training loop structure:
    // Choose initial action `a` BEFORE the loop, and inside the loop choose `next_action` (`a'`)
    // before calling `agent.update(s, a, r, s', a', terminated)`!
    for (int ep = 0; ep < 600; ++ep) {
        int state = env.reset();
        rl::GridAction action = agent.select_action(state);

        while (true) {
            rl::StepResult<int> res = env.step(action);
            rl::GridAction next_action = agent.select_action(res.next_state);

            agent.update(state, action, res.reward, res.next_state, next_action, res.terminated);

            state = res.next_state;
            action = next_action; // In SARSA, the next step MUST execute `next_action`!
            if (res.done()) {
                break;
            }
        }
    }

    int state = env.reset();
    int steps = 0;
    bool reached_goal = false;
    while (steps < 20) {
        rl::GridAction action = agent.greedy_action(state);
        rl::StepResult<int> res = env.step(action);
        steps++;
        state = res.next_state;
        if (res.done()) {
            reached_goal = (res.reward > 0.0);
            break;
        }
    }

    assert(reached_goal);
    assert(steps == 8);
    std::cout << "[PASS] test_sarsa_gridworld_convergence (Solved 5x5 grid in " << steps << " steps)\n";
}

int main() {
    std::cout << "=== Running SARSA Unit Tests ===\n";
    test_sarsa_on_policy_update();
    test_sarsa_gridworld_convergence();
    std::cout << "All SARSA tests passed!\n";
    return 0;
}
