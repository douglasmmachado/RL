/**
 * @file dp_solver.hpp
 * @brief Exact Model-Based Dynamic Programming (Value Iteration) solver for 2D GridWorld!
 *
 * Allows stepping through exact Bellman optimality sweeps 1 iteration at a time in RL Studio
 * to compare Model-Based Planning (Value Iteration) vs. Model-Free Learning (Q-Learning & SARSA).
 */

#pragma once

#include "rl/gridworld.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace studio {

/**
 * @brief Exact Value Iteration solver for rl::GridWorld (supporting Walls, Traps, Wind, and Slip!).
 */
struct ValueIterationSolver {
    std::vector<double> V;            ///< State value function V(s) for each grid cell.
    std::vector<int> policy;          ///< Greedy action index (0..3) per state.
    int iterations = 0;               ///< Number of Bellman sweeps performed.
    double max_delta = 0.0;           ///< Max |V_new(s) - V_old(s)| from last sweep.

    /**
     * @brief Resets V(s) = 0 for all states in `env`.
     */
    void reset(const rl::GridWorld& env) {
        V.assign(env.num_states(), 0.0);
        policy.assign(env.num_states(), 0);
        iterations = 0;
        max_delta = 0.0;
    }

    /**
     * @brief Performs `num_sweeps` exact Bellman Optimality sweeps:
     *        V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') ]
     * @param env Reference GridWorld (reads walls, traps, goal, wind, and slip_prob).
     * @param gamma Discount factor.
     * @param num_sweeps Number of full-grid Bellman sweeps to execute.
     */
    void sweep(const rl::GridWorld& env, double gamma, int num_sweeps = 1) {
        int W = env.width();
        int H = env.height();
        if (static_cast<int>(V.size()) != W * H) reset(env);

        double slip = env.slip_prob();

        // Deterministic transition helper for a single (x, y, action_idx)
        auto transition = [&](int x, int y, int act_idx, int& nx, int& ny, double& rew, bool& term) {
            int wy = std::clamp(y - env.wind_at(x), 0, H - 1);
            int cx = x, cy = wy;
            if (act_idx == 0) cy = std::clamp(wy - 1, 0, H - 1);      // Up
            else if (act_idx == 1) cx = std::clamp(x + 1, 0, W - 1);  // Right
            else if (act_idx == 2) cy = std::clamp(wy + 1, 0, H - 1); // Down
            else if (act_idx == 3) cx = std::clamp(x - 1, 0, W - 1);  // Left

            if (env.cell_at(cx, cy) == rl::CellType::Wall) {
                nx = x; ny = y;
            } else {
                nx = cx; ny = cy;
            }

            auto ct = env.cell_at(nx, ny);
            if (ct == rl::CellType::Goal) { rew = 10.0; term = true; }
            else if (ct == rl::CellType::Trap) { rew = -10.0; term = true; }
            else { rew = -0.1; term = false; }
        };

        for (int sw = 0; sw < num_sweeps; ++sw) {
            std::vector<double> V_next = V;
            max_delta = 0.0;

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int s = y * W + x;
                    auto ct = env.cell_at(x, y);
                    if (ct != rl::CellType::Empty) {
                        V_next[s] = 0.0;
                        continue;
                    }

                    double best_q = -1e18;
                    int best_a = 0;

                    for (int a = 0; a < 4; ++a) {
                        // Expected value over intended move (1 - slip) and perpendicular slips (slip/2 each)
                        int acts[3] = {a, (a + 1) % 4, (a + 3) % 4};
                        double probs[3] = {1.0 - slip, slip * 0.5, slip * 0.5};
                        double q_val = 0.0;

                        for (int k = 0; k < 3; ++k) {
                            if (probs[k] <= 0.0) continue;
                            int nx, ny;
                            double r;
                            bool term;
                            transition(x, y, acts[k], nx, ny, r, term);
                            q_val += probs[k] * (r + (term ? 0.0 : gamma * V[ny * W + nx]));
                        }

                        if (q_val > best_q) {
                            best_q = q_val;
                            best_a = a;
                        }
                    }

                    V_next[s] = best_q;
                    policy[s] = best_a;
                    max_delta = std::max(max_delta, std::abs(V_next[s] - V[s]));
                }
            }

            V = std::move(V_next);
            iterations++;
        }
    }
};

} // namespace studio
