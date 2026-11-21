/**
 * @file taxi_lab.hpp
 * @brief Classic Gymnasium Taxi-v3 (500 discrete states, 6 actions) Hierarchical Tabular RL Lab!
 *
 * A taxi on a 5x5 grid with walls must:
 *   1. Drive to the Passenger at one of 4 colored landmarks (R, G, Y, B)
 *   2. Execute Pickup (+0, or -10 if illegal!)
 *   3. Drive to the Destination landmark
 *   4. Execute Dropoff (+20 + episode ends!)
 * Compares Off-Policy Q-Learning vs. On-Policy SARSA side by side!
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace studio {

/**
 * @brief Taxi-v3 environment + parallel Q-Learning vs. SARSA trainers for RL Studio.
 */
struct TaxiLab {
    static constexpr int W = 5, H = 5;
    static constexpr int NUM_STATES = 5 * 5 * 5 * 4; // 500 states: (row, col, pass_loc[0..4], dest[0..3])
    static constexpr int NUM_ACTIONS = 6;            // 0=Up, 1=Right, 2=Down, 3=Left, 4=Pickup, 5=Dropoff

    static constexpr std::array<std::pair<int, int>, 4> LANDMARKS = {{{0, 0}, {4, 0}, {0, 4}, {3, 4}}}; // R, G, Y, B

    struct TaxiState {
        int x = 2, y = 2;
        int pass_loc = 0; // 0..3 = at landmark, 4 = inside taxi!
        int dest = 1;     // 0..3 = target landmark
    };

    float alpha = 0.25f;
    float gamma = 0.95f;
    float epsilon = 0.20f;
    int episodes = 0;

    std::vector<double> q_qlearn;
    std::vector<double> q_sarsa;
    std::mt19937 rng{42};

    std::vector<double> x_ep, ret_q, ret_s;
    std::vector<TaxiState> path_q, path_s;

    TaxiLab() { reset(); }

    static int encode(const TaxiState& s) noexcept {
        return (((s.y * 5 + s.x) * 5) + s.pass_loc) * 4 + s.dest;
    }

    /**
     * @brief Checks internal vertical walls between cells (matching classic Taxi-v3 barriers).
     */
    static bool blocked(int x, int y, int nx, int ny) noexcept {
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) return true;
        if (y == ny) {
            int lo = std::min(x, nx);
            if ((y == 0 || y == 1) && lo == 1) return true; // Wall between col 1 & 2 on top rows
            if ((y == 3 || y == 4) && (lo == 0 || lo == 2)) return true; // Walls on bottom rows
        }
        return false;
    }

    TaxiState sample_start() {
        TaxiState s;
        s.x = std::uniform_int_distribution<int>(0, 4)(rng);
        s.y = std::uniform_int_distribution<int>(0, 4)(rng);
        s.pass_loc = std::uniform_int_distribution<int>(0, 3)(rng);
        s.dest = (s.pass_loc + std::uniform_int_distribution<int>(1, 3)(rng)) % 4;
        return s;
    }

    std::pair<double, bool> step_env(TaxiState& s, int action) const {
        if (action == 0) { if (!blocked(s.x, s.y, s.x, s.y - 1)) s.y--; return {-1.0, false}; }
        if (action == 1) { if (!blocked(s.x, s.y, s.x + 1, s.y)) s.x++; return {-1.0, false}; }
        if (action == 2) { if (!blocked(s.x, s.y, s.x, s.y + 1)) s.y++; return {-1.0, false}; }
        if (action == 3) { if (!blocked(s.x, s.y, s.x - 1, s.y)) s.x--; return {-1.0, false}; }
        if (action == 4) { // Pickup
            if (s.pass_loc < 4 && s.x == LANDMARKS[s.pass_loc].first && s.y == LANDMARKS[s.pass_loc].second) {
                s.pass_loc = 4;
                return {-1.0, false};
            }
            return {-10.0, false};
        }
        // action == 5: Dropoff
        if (s.pass_loc == 4 && s.x == LANDMARKS[s.dest].first && s.y == LANDMARKS[s.dest].second) {
            s.pass_loc = s.dest;
            return {20.0, true};
        }
        return {-10.0, false};
    }

    int greedy(const std::vector<double>& Q, int sid) const {
        int best = 0;
        double bv = Q[sid * NUM_ACTIONS];
        for (int a = 1; a < NUM_ACTIONS; ++a) {
            if (Q[sid * NUM_ACTIONS + a] > bv) { bv = Q[sid * NUM_ACTIONS + a]; best = a; }
        }
        return best;
    }

    int eps_greedy(const std::vector<double>& Q, int sid) {
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < epsilon) {
            return std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1)(rng);
        }
        return greedy(Q, sid);
    }

    void record_demo_paths() {
        auto trace = [&](const std::vector<double>& Q, std::vector<TaxiState>& out) {
            out.clear();
            TaxiState s{2, 2, 0, 3}; // Start center, pick up at R(0,0), drop at B(3,4)
            out.push_back(s);
            for (int t = 0; t < 35; ++t) {
                auto [r, done] = step_env(s, greedy(Q, encode(s)));
                out.push_back(s);
                if (done) break;
            }
        };
        trace(q_qlearn, path_q);
        trace(q_sarsa, path_s);
    }

    void reset() {
        q_qlearn.assign(NUM_STATES * NUM_ACTIONS, 0.0);
        q_sarsa.assign(NUM_STATES * NUM_ACTIONS, 0.0);
        episodes = 0;
        x_ep.clear(); ret_q.clear(); ret_s.clear();
        record_demo_paths();
    }

    void train_episodes(int n) {
        for (int ep = 0; ep < n; ++ep) {
            // Q-Learning
            TaxiState sq = sample_start();
            double rq = 0.0;
            for (int t = 0; t < 100; ++t) {
                int sid = encode(sq);
                int a = eps_greedy(q_qlearn, sid);
                auto [r, done] = step_env(sq, a);
                int nsid = encode(sq);
                double max_nq = done ? 0.0 : q_qlearn[nsid * NUM_ACTIONS + greedy(q_qlearn, nsid)];
                q_qlearn[sid * NUM_ACTIONS + a] += alpha * (r + gamma * max_nq - q_qlearn[sid * NUM_ACTIONS + a]);
                rq += r;
                if (done) break;
            }

            // SARSA
            TaxiState ss = sample_start();
            int as = eps_greedy(q_sarsa, encode(ss));
            double rs = 0.0;
            for (int t = 0; t < 100; ++t) {
                int sid = encode(ss);
                auto [r, done] = step_env(ss, as);
                int nsid = encode(ss);
                int nas = eps_greedy(q_sarsa, nsid);
                double nq = done ? 0.0 : q_sarsa[nsid * NUM_ACTIONS + nas];
                q_sarsa[sid * NUM_ACTIONS + as] += alpha * (r + gamma * nq - q_sarsa[sid * NUM_ACTIONS + as]);
                rs += r;
                as = nas;
                if (done) break;
            }

            episodes++;
            if (episodes <= 50 || episodes % 5 == 0) {
                x_ep.push_back(static_cast<double>(episodes));
                double pq = ret_q.empty() ? rq : 0.90 * ret_q.back() + 0.10 * rq;
                double ps = ret_s.empty() ? rs : 0.90 * ret_s.back() + 0.10 * rs;
                ret_q.push_back(pq);
                ret_s.push_back(ps);
            }
        }
        record_demo_paths();
    }
};

} // namespace studio
