/**
 * @file gridworld.cpp
 * @brief Implementation of the 2D GridWorld environment with walls, multiple traps, and BFS solvability check.
 */

#include "rl/gridworld.hpp"
#include <algorithm>
#include <iostream>
#include <queue>
#include <random>

namespace rl {

GridWorld::GridWorld(int width, int height)
    : width_(width),
      height_(height),
      goal_x_(width - 1),
      goal_y_(height - 1),
      trap_x_(width / 2),
      trap_y_(height / 2),
      cells_(width * height, CellType::Empty) {
    cells_[to_state_id(trap_x_, trap_y_)] = CellType::Trap;
    cells_[to_state_id(goal_x_, goal_y_)] = CellType::Goal;
}

bool GridWorld::is_solvable() const {
    std::vector<bool> visited(width_ * height_, false);
    std::queue<std::pair<int, int>> q;

    q.push({0, 0});
    visited[0] = true;

    const int dx[4] = {0, 1, 0, -1};
    const int dy[4] = {-1, 0, 1, 0};

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();

        if (cx == goal_x_ && cy == goal_y_) {
            return true;
        }

        for (int dir = 0; dir < 4; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];
            if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
                int id = to_state_id(nx, ny);
                if (!visited[id] && (cells_[id] == CellType::Empty || cells_[id] == CellType::Goal)) {
                    visited[id] = true;
                    q.push({nx, ny});
                }
            }
        }
    }
    return false;
}

void GridWorld::randomize_layout(int width, int height, double trap_prob, double wall_prob, unsigned int seed) {
    width_ = std::max(3, width);
    height_ = std::max(3, height);
    goal_x_ = width_ - 1;
    goal_y_ = height_ - 1;
    trap_x_ = width_ / 2;
    trap_y_ = height_ / 2;
    max_steps_ = width_ * height_ * 4;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Retry layout generation until BFS confirms a solvable path from (0,0) to Goal
    for (int attempt = 0; attempt < 200; ++attempt) {
        cells_.assign(width_ * height_, CellType::Empty);
        cells_[to_state_id(goal_x_, goal_y_)] = CellType::Goal;

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if ((x == 0 && y == 0) || (x == goal_x_ && y == goal_y_)) {
                    continue;
                }
                double r = dist(rng);
                if (r < trap_prob) {
                    cells_[to_state_id(x, y)] = CellType::Trap;
                    trap_x_ = x;
                    trap_y_ = y;
                } else if (r < trap_prob + wall_prob) {
                    cells_[to_state_id(x, y)] = CellType::Wall;
                }
            }
        }

        if (is_solvable()) {
            break;
        }
    }
    reset();
}

void GridWorld::set_cliff_layout() {
    width_ = 8;
    height_ = 4;
    goal_x_ = width_ - 1;
    goal_y_ = 0;
    trap_x_ = 1;
    trap_y_ = 0;
    max_steps_ = 100;

    cells_.assign(width_ * height_, CellType::Empty);
    // Start is at (0, 0) (top-left), Goal is at (7, 0) (top-right).
    // Row 0 between x = 1..6 is a cliff of Traps!
    // Q-Learning hugs row 1 (right next to the cliff); SARSA takes the safer lower rows (row 2-3)!
    for (int x = 1; x < width_ - 1; ++x) {
        cells_[to_state_id(x, 0)] = CellType::Trap;
    }
    cells_[to_state_id(goal_x_, goal_y_)] = CellType::Goal;
    reset();
}

int GridWorld::reset() {
    agent_x_ = 0;
    agent_y_ = 0;
    current_steps_ = 0;
    return to_state_id(agent_x_, agent_y_);
}

StepResult<int> GridWorld::step(const GridAction& action) {
    current_steps_++;

    int next_x = agent_x_;
    int next_y = agent_y_;

    // 2. Compute candidate position clamped to outer walls
    switch (action) {
        case GridAction::Up:
            next_y = std::clamp(agent_y_ - 1, 0, height_ - 1);
            break;
        case GridAction::Down:
            next_y = std::clamp(agent_y_ + 1, 0, height_ - 1);
            break;
        case GridAction::Left:
            next_x = std::clamp(agent_x_ - 1, 0, width_ - 1);
            break;
        case GridAction::Right:
            next_x = std::clamp(agent_x_ + 1, 0, width_ - 1);
            break;
    }

    // If the candidate tile is not an internal Wall, commit the move
    if (cell_at(next_x, next_y) != CellType::Wall) {
        agent_x_ = next_x;
        agent_y_ = next_y;
    }

    // 3. Reward and termination based on CellType
    bool terminated = false;
    double reward = 0.0;
    CellType current_cell = cell_at(agent_x_, agent_y_);

    if (current_cell == CellType::Goal) {
        reward = 10.0;
        terminated = true;
    } else if (current_cell == CellType::Trap) {
        reward = -10.0;
        terminated = true;
    } else {
        reward = -0.1;
    }

    // 4. Truncation
    bool truncated = (current_steps_ >= max_steps_) && !terminated;

    return StepResult<int>{to_state_id(agent_x_, agent_y_), reward, terminated, truncated};
}

void GridWorld::render() const {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            if (x == agent_x_ && y == agent_y_) {
                std::cout << " A ";
            } else if (cell_at(x, y) == CellType::Goal) {
                std::cout << " G ";
            } else if (cell_at(x, y) == CellType::Trap) {
                std::cout << " X ";
            } else if (cell_at(x, y) == CellType::Wall) {
                std::cout << " # ";
            } else {
                std::cout << " . ";
            }
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

} // namespace rl
