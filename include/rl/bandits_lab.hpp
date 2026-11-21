/**
 * @file bandits_lab.hpp
 * @brief Interactive Multi-Armed Bandits Lab comparing Epsilon-Greedy, UCB1, and Thompson Sampling!
 *
 * Demonstrates the core Exploration vs. Exploitation dilemma (stateless RL, S = 1) with
 * live true vs. estimated arm means, cumulative regret, and % optimal arm selection curves.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace studio {

/**
 * @brief K-Armed Gaussian Bandit lab running Epsilon-Greedy, UCB1, and Thompson Sampling in parallel.
 */
struct BanditsLab {
    static constexpr int K = 10; ///< Number of slot-machine arms (10-armed testbed from Sutton & Barto Ch. 2).

    std::array<double, K> true_means{};
    int optimal_arm = 0;

    // Hyperparameters
    float epsilon = 0.10f;
    float ucb_c = 2.0f;
    float reward_std = 1.0f;
    int seed = 42;

    // 3 Agents: 0 = Epsilon-Greedy, 1 = UCB1, 2 = Thompson Sampling
    struct Agent {
        std::array<double, K> q_est{};
        std::array<int, K> pulls{};
        double total_reward = 0.0;
        double cum_regret = 0.0;
        int optimal_pulls = 0;
    };
    std::array<Agent, 3> agents{};
    int total_steps = 0;
    std::mt19937 rng{42};

    // Plot histories
    std::vector<double> x_step;
    std::array<std::vector<double>, 3> hist_avg_reward;
    std::array<std::vector<double>, 3> hist_cum_regret;
    std::array<std::vector<double>, 3> hist_opt_pct;

    BanditsLab() { reset(); }

    /**
     * @brief Samples a fresh 10-armed bandit testbed and resets all 3 agents & curves.
     */
    void reset() {
        rng.seed(static_cast<unsigned int>(seed));
        std::normal_distribution<double> mean_dist(0.0, 1.0);
        optimal_arm = 0;
        for (int i = 0; i < K; ++i) {
            true_means[i] = mean_dist(rng);
            if (true_means[i] > true_means[optimal_arm]) optimal_arm = i;
        }
        agents = {};
        total_steps = 0;
        x_step.clear();
        for (int a = 0; a < 3; ++a) {
            hist_avg_reward[a].clear();
            hist_cum_regret[a].clear();
            hist_opt_pct[a].clear();
        }
    }

    /**
     * @brief Runs `n_steps` parallel pulls for Epsilon-Greedy, UCB1, and Thompson Sampling.
     * @param n_steps Number of steps to simulate.
     */
    void step_n(int n_steps) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::uniform_int_distribution<int> rand_arm(0, K - 1);
        std::normal_distribution<double> noise(0.0, reward_std);

        for (int s = 0; s < n_steps; ++s) {
            total_steps++;
            double opt_mean = true_means[optimal_arm];

            for (int ag = 0; ag < 3; ++ag) {
                auto& A = agents[ag];
                int chosen = 0;

                if (ag == 0) {
                    // 1. Epsilon-Greedy
                    if (u01(rng) < epsilon) {
                        chosen = rand_arm(rng);
                    } else {
                        chosen = static_cast<int>(std::max_element(A.q_est.begin(), A.q_est.end()) - A.q_est.begin());
                    }
                } else if (ag == 1) {
                    // 2. UCB1: argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]
                    double best_val = -1e18;
                    for (int i = 0; i < K; ++i) {
                        if (A.pulls[i] == 0) { chosen = i; break; }
                        double bonus = ucb_c * std::sqrt(std::log(static_cast<double>(total_steps)) / A.pulls[i]);
                        if (A.q_est[i] + bonus > best_val) {
                            best_val = A.q_est[i] + bonus;
                            chosen = i;
                        }
                    }
                } else {
                    // 3. Gaussian Thompson Sampling (sample from posterior N(q_est, 1 / (N + 1)))
                    double best_sample = -1e18;
                    for (int i = 0; i < K; ++i) {
                        double post_std = static_cast<double>(reward_std) / std::sqrt(static_cast<double>(A.pulls[i] + 1));
                        std::normal_distribution<double> post(A.q_est[i], post_std);
                        double sample = post(rng);
                        if (sample > best_sample) {
                            best_sample = sample;
                            chosen = i;
                        }
                    }
                }

                // Pull chosen arm and update sample-average estimate
                double r = true_means[chosen] + noise(rng);
                A.pulls[chosen]++;
                A.q_est[chosen] += (r - A.q_est[chosen]) / static_cast<double>(A.pulls[chosen]);
                A.total_reward += r;
                A.cum_regret += (opt_mean - true_means[chosen]);
                if (chosen == optimal_arm) A.optimal_pulls++;
            }

            // Record telemetry every 5 steps (or first 50 steps) for smooth plots
            if (total_steps <= 50 || total_steps % 5 == 0) {
                x_step.push_back(static_cast<double>(total_steps));
                for (int ag = 0; ag < 3; ++ag) {
                    hist_avg_reward[ag].push_back(agents[ag].total_reward / total_steps);
                    hist_cum_regret[ag].push_back(agents[ag].cum_regret);
                    hist_opt_pct[ag].push_back(100.0 * agents[ag].optimal_pulls / total_steps);
                }
            }
        }
    }
};

} // namespace studio
