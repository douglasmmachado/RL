/**
 * @file environment.hpp
 * @brief Generic Reinforcement Learning environment interface and step result types.
 */

#pragma once

namespace rl {

/**
 * @brief Returned by Environment::step(action), analogous to Gymnasium's step tuple in Python.
 *
 * @tparam StateType The representation of an environment state (e.g., int for discrete grids,
 *                   std::array<double, N> for continuous control).
 */
template <typename StateType>
struct StepResult {
    StateType next_state;   ///< The state reached after applying the action.
    double reward{0.0};     ///< Scalar reward obtained from the transition.
    bool terminated{false}; ///< True if the episode reached a terminal state (e.g., Goal or Trap).
    bool truncated{false};  ///< True if the episode ended due to a time/step limit.

    /**
     * @brief Checks whether the episode has ended (either terminated or truncated).
     * @return True if terminated or truncated, false otherwise.
     */
    [[nodiscard]] bool done() const noexcept {
        return terminated || truncated;
    }
};

/**
 * @brief Abstract base class template for Reinforcement Learning environments.
 *
 * @tparam StateType The data type representing an environment state.
 * @tparam ActionType The data type representing an agent action.
 */
template <typename StateType, typename ActionType>
class Environment {
public:
    virtual ~Environment() = default;

    /**
     * @brief Resets the environment to its initial state.
     * @return The starting StateType for the new episode.
     */
    virtual StateType reset() = 0;

    /**
     * @brief Applies an action to advance the environment by one time step.
     * @param action The action chosen by the agent.
     * @return StepResult<StateType> containing (next_state, reward, terminated, truncated).
     */
    virtual StepResult<StateType> step(const ActionType& action) = 0;

    /**
     * @brief Prints a visual representation of the current environment state to the terminal.
     */
    virtual void render() const = 0;
};

} // namespace rl
