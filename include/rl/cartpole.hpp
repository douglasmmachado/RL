/**
 * @file cartpole.hpp
 * @brief Classic CartPole balancing environment (Barto, Sutton & Anderson, 1983).
 *
 * A pole is attached by an un-actuated joint to a cart moving along a frictionless track.
 * The agent pushes the cart left or right with a fixed force to keep the pole upright.
 */

#pragma once

#include "rl/environment.hpp"
#include <array>
#include <random>

namespace rl {

/**
 * @brief Continuous CartPole state: {x, x_dot, theta, theta_dot}.
 *
 * - x         : cart position [m]
 * - x_dot     : cart velocity [m/s]
 * - theta     : pole angle from vertical [rad] (0 = upright, positive = leaning right)
 * - theta_dot : pole angular velocity [rad/s]
 */
using CartPoleState = std::array<double, 4>;

/**
 * @brief Discrete actions available in CartPole.
 */
enum class CartAction : int {
    Left = 0,  ///< Push the cart with force -force_mag
    Right = 1  ///< Push the cart with force +force_mag
};

/**
 * @brief CartPole environment implementing the generic Environment interface.
 *
 * Reward: +1.0 for every step (including the terminating step).
 * Termination: |x| > x_threshold or |theta| > theta_threshold.
 * Truncation: after max_steps steps.
 */
class CartPole : public Environment<CartPoleState, CartAction> {
public:
    // ---- Physical constants (same values as Gymnasium CartPole-v1) ----
    static constexpr double gravity = 9.8;          ///< Gravitational acceleration [m/s^2]
    static constexpr double mass_cart = 1.0;        ///< Cart mass [kg]
    static constexpr double mass_pole = 0.1;        ///< Pole mass [kg]
    static constexpr double total_mass = mass_cart + mass_pole; ///< Combined mass [kg]
    static constexpr double half_length = 0.5;      ///< Half of the pole length [m]
    static constexpr double force_mag = 10.0;       ///< Magnitude of the push force [N]
    static constexpr double tau = 0.02;             ///< Integration time step [s]
    static constexpr double x_threshold = 2.4;      ///< Cart position limit [m]
    static constexpr double theta_threshold = 12.0 * 3.14159265358979323846 / 180.0; ///< ~0.2094 rad
    static constexpr int max_steps = 500;           ///< Episode truncation limit

    /**
     * @brief Constructs a CartPole environment.
     * @param seed Random seed used to sample initial states in reset().
     */
    explicit CartPole(unsigned int seed = 42);

    /**
     * @brief Samples each state variable uniformly in [-0.05, 0.05] and zeroes the step counter.
     * @return The initial CartPoleState.
     */
    CartPoleState reset() override;

    /**
     * @brief Applies a push force and integrates the dynamics for one time step (Euler method).
     * @param action Push direction (Left or Right).
     * @return StepResult containing the new state, reward (+1.0), terminated and truncated flags.
     */
    StepResult<CartPoleState> step(const CartAction& action) override;

    /**
     * @brief Prints the current state values to the terminal.
     */
    void render() const override;

    /**
     * @brief Overwrites the internal state (useful for unit tests and debugging).
     * @param s The state to set.
     */
    void set_state(const CartPoleState& s) noexcept { state_ = s; }

    /**
     * @brief Returns the current internal state.
     * @return Const reference to the current CartPoleState.
     */
    [[nodiscard]] const CartPoleState& state() const noexcept { return state_; }

    /**
     * @brief Returns the number of steps taken in the current episode.
     * @return Step counter.
     */
    [[nodiscard]] int current_steps() const noexcept { return current_steps_; }

private:
    CartPoleState state_{0.0, 0.0, 0.0, 0.0}; ///< Current {x, x_dot, theta, theta_dot}.
    int current_steps_{0};                     ///< Steps taken in the current episode.
    std::mt19937 rng_;                         ///< Random engine for initial states.
};

} // namespace rl
