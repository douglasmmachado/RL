/**
 * @file test_cartpole.cpp
 * @brief Unit tests for the CartPole environment dynamics.
 */

#include "rl/cartpole.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief Checks if two floating-point values are equal within a tolerance.
 * @param a First value.
 * @param b Second value.
 * @param eps Tolerance.
 * @return True if |a - b| < eps.
 */
bool approx_equal(double a, double b, double eps = 1e-4) {
    return std::abs(a - b) < eps;
}

/**
 * @brief reset() must sample every state variable in [-0.05, 0.05].
 */
void test_reset_bounds() {
    rl::CartPole env(7);
    for (int i = 0; i < 100; ++i) {
        rl::CartPoleState s = env.reset();
        for (double v : s) {
            assert(v >= -0.05 && v <= 0.05);
        }
        assert(env.current_steps() == 0);
    }
    std::cout << "[PASS] test_reset_bounds\n";
}

/**
 * @brief One push to the right from rest: checks the exact Euler integration result.
 */
void test_single_push_right() {
    rl::CartPole env;
    env.reset();
    env.set_state({0.0, 0.0, 0.0, 0.0});

    rl::StepResult<rl::CartPoleState> res = env.step(rl::CartAction::Right);

    // Positions use the OLD velocities (still 0), velocities use the new accelerations.
    assert(approx_equal(res.next_state[0], 0.0));       // x
    assert(approx_equal(res.next_state[1], 0.195122));  // x_dot   = tau * x_acc
    assert(approx_equal(res.next_state[2], 0.0));       // theta
    assert(approx_equal(res.next_state[3], -0.292683)); // theta_dot = tau * theta_acc (pole falls back!)
    assert(approx_equal(res.reward, 1.0));
    assert(!res.terminated && !res.truncated);
    assert(env.current_steps() == 1);
    std::cout << "[PASS] test_single_push_right\n";
}

/**
 * @brief Pushing left must mirror pushing right.
 */
void test_symmetry() {
    rl::CartPole env;
    env.reset();
    env.set_state({0.0, 0.0, 0.0, 0.0});
    auto res = env.step(rl::CartAction::Left);
    assert(approx_equal(res.next_state[1], -0.195122));
    assert(approx_equal(res.next_state[3], 0.292683));
    std::cout << "[PASS] test_symmetry\n";
}

/**
 * @brief Pole angle beyond 12 degrees terminates the episode.
 */
void test_pole_falls() {
    rl::CartPole env;
    env.reset();
    env.set_state({0.0, 0.0, 0.25, 0.0});
    auto res = env.step(rl::CartAction::Right);
    assert(res.terminated);
    assert(!res.truncated);
    assert(approx_equal(res.reward, 1.0));
    std::cout << "[PASS] test_pole_falls\n";
}

/**
 * @brief Cart leaving the track (|x| > 2.4) terminates the episode.
 */
void test_cart_out_of_bounds() {
    rl::CartPole env;
    env.reset();
    env.set_state({2.5, 0.0, 0.0, 0.0});
    auto res = env.step(rl::CartAction::Left);
    assert(res.terminated);
    std::cout << "[PASS] test_cart_out_of_bounds\n";
}

/**
 * @brief An episode that never falls is truncated at max_steps.
 */
void test_truncation() {
    rl::CartPole env;
    env.reset();
    rl::StepResult<rl::CartPoleState> res{};
    for (int i = 0; i < rl::CartPole::max_steps; ++i) {
        env.set_state({0.0, 0.0, 0.0, 0.0}); // keep it perfectly balanced
        res = env.step(i % 2 == 0 ? rl::CartAction::Right : rl::CartAction::Left);
    }
    assert(res.truncated);
    assert(!res.terminated);
    std::cout << "[PASS] test_truncation\n";
}

int main() {
    std::cout << "=== Running CartPole Unit Tests ===\n";
    test_reset_bounds();
    test_single_push_right();
    test_symmetry();
    test_pole_falls();
    test_cart_out_of_bounds();
    test_truncation();
    std::cout << "All CartPole tests passed!\n";
    return 0;
}
