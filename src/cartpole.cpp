/**
 * @file cartpole.cpp
 * @brief Implementation of the CartPole environment dynamics.
 */

#include "rl/cartpole.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace rl {

CartPole::CartPole(unsigned int seed) : rng_(seed) {}

CartPoleState CartPole::reset() {
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    for (double& v : state_) {
        v = dist(rng_);
    }
    current_steps_ = 0;
    return state_;
}

StepResult<CartPoleState> CartPole::step(const CartAction& action) {
    // TODO(Stage 4 Exercise): implement the CartPole dynamics.
    //
    // Goal: compute the accelerations from the current state and the applied force,
    // integrate one time step of length `tau` with the Euler method, then decide
    // the reward, termination and truncation. The equations are in the chat.
    auto& [x, x_dot, theta, theta_dot] = state_;

    double force = (action == CartAction::Right) ? force_mag : -force_mag;

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double pole_mass_length = mass_pole * half_length;

    double temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) /
                  (mass_cart + mass_pole);

    double theta_double_dot = (gravity * sin_theta - cos_theta * temp) /
                              (half_length * (4.0/3.0 - mass_pole * cos_theta * cos_theta /
                                                         (mass_cart + mass_pole)));
                                                
    double x_double_dot = temp - pole_mass_length * theta_double_dot * cos_theta /
                                      (mass_cart + mass_pole);

    state_[0] += x_dot * tau;
    state_[1] += x_double_dot * tau;
    state_[2] += theta_dot * tau;
    state_[3] += theta_double_dot * tau;

    current_steps_++;
    bool terminated = false;
    if (std::abs(state_[0]) > x_threshold || std::abs(state_[2]) > theta_threshold) {
    terminated = true;
}
    bool truncated = false;
    if (!terminated && current_steps_ >= max_steps) {
    truncated = true;
}

    double reward = 1.0;

    return StepResult<CartPoleState>{state_, reward, terminated, truncated};
}

void CartPole::render() const {
    std::cout << std::fixed << std::setprecision(3)
              << "x=" << state_[0] << "  x_dot=" << state_[1]
              << "  theta=" << state_[2] << "  theta_dot=" << state_[3]
              << "  step=" << current_steps_ << '\n';
}

} // namespace rl
