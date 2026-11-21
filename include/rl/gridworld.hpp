/**
 * @file gridworld.hpp
 * @brief 2D discrete GridWorld environment supporting multiple traps, walls, and random layout generation.
 */

#pragma once

#include "rl/environment.hpp"
#include <iostream>
#include <vector>

namespace rl {

/**
 * @brief Strongly-typed discrete actions available in the 2D GridWorld.
 */
enum class GridAction : int {
    Up = 0,    ///< Move one cell up (y - 1)
    Right = 1, ///< Move one cell right (x + 1)
    Down = 2,  ///< Move one cell down (y + 1)
    Left = 3   ///< Move one cell left (x - 1)
};

/**
 * @brief Types of cells in a 2D GridWorld.
 */
enum class CellType : int {
    Empty = 0, ///< Normal walkable tile (-0.1 step reward)
    Wall = 1,  ///< Impassable obstacle (agent bounces back if walking into it)
    Trap = 2,  ///< Terminal hazard tile (-10.0 reward, ends episode)
    Goal = 3   ///< Terminal target tile (+10.0 reward, ends episode)
};

/**
 * @brief A 2D GridWorld environment supporting configurable dimensions, multiple traps,
 *        impassable walls, and procedural random map generation.
 */
class GridWorld : public Environment<int, GridAction> {
public:
    /**
     * @brief Constructs a standard GridWorld of the given dimensions with a default central trap.
     * @param width Number of columns in the grid.
     * @param height Number of rows in the grid.
     */
    GridWorld(int width, int height);

    /**
     * @brief Procedurally generates a new random GridWorld layout (dimensions, walls, and traps)
     *        guaranteed to have a valid solvable path from (0, 0) to the Goal!
     *
     * @param width New number of columns (e.g., 5 to 10).
     * @param height New number of rows (e.g., 5 to 10).
     * @param trap_prob Probability in [0.0, 0.4] that an open cell is a Trap.
     * @param wall_prob Probability in [0.0, 0.4] that an open cell is an impassable Wall.
     * @param seed Random seed for layout generation.
     */
    void randomize_layout(int width, int height, double trap_prob, double wall_prob, unsigned int seed);

    /**
     * @brief Loads the classic Sutton & Barto "Cliff Walking" layout (8x4) where the bottom row
     *        between Start (0, 3) and Goal (7, 3) is lined with Traps — ideal for comparing
     *        off-policy Q-Learning (optimal risky cliff-edge path) vs. on-policy SARSA (safe upper path)!
     */
    void set_cliff_layout();

    /**
     * @brief Resets the agent to the top-left corner (0, 0) and zeroes the step counter.
     * @return The initial state ID (0).
     */
    int reset() override;

    /**
     * @brief Moves the agent one step in the specified direction (blocked by grid edges and Walls).
     *
     * Rewards:
     * - Goal cell: +10.0 (terminated = true)
     * - Trap cell: -10.0 (terminated = true)
     * - Normal/Wall cell: -0.1 (terminated = false)
     *
     * @param action The direction to move (Up, Right, Down, Left).
     * @return StepResult<int> containing the new state ID, reward, terminated, and truncated flags.
     */
    StepResult<int> step(const GridAction& action) override;

    /**
     * @brief Prints an ASCII representation of the grid ('A'=Agent, 'G'=Goal, 'X'=Trap, '#'=Wall).
     */
    void render() const override;

    /**
     * @brief Returns the CellType at coordinates (x, y).
     * @param x Column index.
     * @param y Row index.
     * @return CellType (Empty, Wall, Trap, or Goal).
     */
    [[nodiscard]] CellType cell_at(int x, int y) const noexcept {
        return cells_[to_state_id(x, y)];
    }

    /**
     * @brief Returns the total number of discrete states in the grid.
     * @return width * height.
     */
    [[nodiscard]] int num_states() const noexcept { return width_ * height_; }

    /**
     * @brief Returns the total number of discrete actions available (4).
     * @return 4.
     */
    [[nodiscard]] static constexpr int num_actions() noexcept { return 4; }

    /// @brief Returns the grid width (columns).
    [[nodiscard]] int width() const noexcept { return width_; }

    /// @brief Returns the grid height (rows).
    [[nodiscard]] int height() const noexcept { return height_; }

    /// @brief Returns the agent's current X (column) coordinate.
    [[nodiscard]] int agent_x() const noexcept { return agent_x_; }

    /// @brief Returns the agent's current Y (row) coordinate.
    [[nodiscard]] int agent_y() const noexcept { return agent_y_; }

    /// @brief Returns the Goal cell's X coordinate.
    [[nodiscard]] int goal_x() const noexcept { return goal_x_; }

    /// @brief Returns the Goal cell's Y coordinate.
    [[nodiscard]] int goal_y() const noexcept { return goal_y_; }

    /// @brief Returns the primary Trap cell's X coordinate.
    [[nodiscard]] int trap_x() const noexcept { return trap_x_; }

    /// @brief Returns the primary Trap cell's Y coordinate.
    [[nodiscard]] int trap_y() const noexcept { return trap_y_; }

    /// @brief Returns the number of steps taken in the current episode.
    [[nodiscard]] int current_steps() const noexcept { return current_steps_; }

private:
    int width_;                     ///< Number of columns in the grid.
    int height_;                    ///< Number of rows in the grid.
    int agent_x_{0};                ///< Current horizontal coordinate of the agent.
    int agent_y_{0};                ///< Current vertical coordinate of the agent.
    int goal_x_;                    ///< Horizontal coordinate of the Goal cell.
    int goal_y_;                    ///< Vertical coordinate of the Goal cell.
    int trap_x_;                    ///< Horizontal coordinate of the default Trap cell.
    int trap_y_;                    ///< Vertical coordinate of the default Trap cell.
    int current_steps_{0};          ///< Steps taken in the current episode.
    int max_steps_{100};            ///< Maximum steps before episode truncation.
    std::vector<CellType> cells_;   ///< Flattened 2D map of CellTypes (size = width_ * height_).

    /**
     * @brief Converts 2D (x, y) grid coordinates into a flattened 1D state index.
     * @param x Column index in [0, width_ - 1].
     * @param y Row index in [0, height_ - 1].
     * @return Flattened state ID: y * width_ + x.
     */
    [[nodiscard]] int to_state_id(int x, int y) const noexcept {
        return y * width_ + x;
    }

    /**
     * @brief Checks via Breadth-First Search (BFS) whether a valid path of Empty cells
     *        exists from (0, 0) to (goal_x_, goal_y_).
     * @return True if the Goal is reachable without stepping on Walls or Traps.
     */
    [[nodiscard]] bool is_solvable() const;
};

} // namespace rl
