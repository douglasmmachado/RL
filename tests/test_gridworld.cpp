#include "rl/gridworld.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

// Helper to compare floating-point numbers safely
bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

void test_initial_reset() {
    rl::GridWorld env(4, 4);
    int s0 = env.reset();
    assert(s0 == 0); // Top-left (0, 0) -> state ID 0
    std::cout << "[PASS] test_initial_reset\n";
}

void test_wall_clamping() {
    rl::GridWorld env(4, 4);
    env.reset();

    // Moving Up or Left from (0, 0) should keep the agent at (0, 0) (state 0)
    auto res_up = env.step(rl::GridAction::Up);
    assert(res_up.next_state == 0);
    assert(approx_equal(res_up.reward, -0.1));
    assert(!res_up.terminated && !res_up.truncated);

    auto res_left = env.step(rl::GridAction::Left);
    assert(res_left.next_state == 0);
    std::cout << "[PASS] test_wall_clamping\n";
}

void test_reach_trap() {
    // In a 4x4 grid, trap is at (width/2, height/2) = (2, 2) -> state ID = 2*4 + 2 = 10
    rl::GridWorld env(4, 4);
    env.reset();

    env.step(rl::GridAction::Right); // (1, 0)
    env.step(rl::GridAction::Right); // (2, 0)
    env.step(rl::GridAction::Down);  // (2, 1)
    auto res = env.step(rl::GridAction::Down); // (2, 2) -> TRAP!

    assert(res.next_state == 10);
    assert(approx_equal(res.reward, -10.0));
    assert(res.terminated == true);
    assert(res.truncated == false);
    assert(res.done() == true);
    std::cout << "[PASS] test_reach_trap\n";
}

void test_reach_goal() {
    // In a 4x4 grid, goal is at (3, 3) -> state ID = 3*4 + 3 = 15
    rl::GridWorld env(4, 4);
    env.reset();

    // Move along the top edge and right edge to avoid the trap at (2, 2)
    env.step(rl::GridAction::Right); // (1, 0)
    env.step(rl::GridAction::Right); // (2, 0)
    env.step(rl::GridAction::Right); // (3, 0)
    env.step(rl::GridAction::Down);  // (3, 1)
    env.step(rl::GridAction::Down);  // (3, 2)
    auto res = env.step(rl::GridAction::Down); // (3, 3) -> GOAL!

    assert(res.next_state == 15);
    assert(approx_equal(res.reward, 10.0));
    assert(res.terminated == true);
    assert(res.truncated == false);
    std::cout << "[PASS] test_reach_goal\n";
}

int main() {
    std::cout << "=== Running GridWorld Unit Tests ===\n";
    test_initial_reset();
    test_wall_clamping();
    test_reach_trap();
    test_reach_goal();
    std::cout << "All GridWorld tests passed!\n";
    return 0;
}
