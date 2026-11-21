/**
 * @file test_cartpole_discretizer.cpp
 * @brief Unit tests for CartPoleDiscretizer (bin boundaries, center, clamping, and Q-Learning).
 */

#include "rl/cartpole_discretizer.hpp"
#include <cassert>
#include <iostream>

/**
 * @brief Tests extreme low/high clamping and center state mapping.
 */
void test_discretizer_bounds_and_center() {
    using D = rl::CartPoleDiscretizer;
    assert(D::num_states() == 5184);

    // 1. All values below lower bounds -> bin (0, 0, 0, 0) -> state ID 0
    int min_id = D::discretize({-10.0, -10.0, -10.0, -10.0});
    assert(min_id == 0);

    // 2. All values above upper bounds -> bin (5, 5, 11, 11) -> state ID 5183
    int max_id = D::discretize({10.0, 10.0, 10.0, 10.0});
    assert(max_id == D::num_states() - 1);

    // 3. Exact origin (0, 0, 0, 0) -> middle bins (3, 3, 6, 6)
    // Flattened: ((3 * 6 + 3) * 12 + 6) * 12 + 6 = (21 * 12 + 6) * 12 + 6 = 258 * 12 + 6 = 3102
    int mid_id = D::discretize({0.0, 0.0, 0.0, 0.0});
    assert(mid_id == 3102);

    // 4. Changing only theta_dot slightly should produce a different ID within [0, 5183]
    int tilted_id = D::discretize({0.0, 0.0, 0.0, 1.0});
    assert(tilted_id != mid_id);
    assert(tilted_id >= 0 && tilted_id < D::num_states());

    std::cout << "[PASS] test_discretizer_bounds_and_center (min=" << min_id
              << ", mid=" << mid_id << ", max=" << max_id << ")\n";
}

int main() {
    std::cout << "=== Running CartPole Discretizer Unit Tests ===\n";
    test_discretizer_bounds_and_center();
    std::cout << "All Discretizer tests passed!\n";
    return 0;
}
