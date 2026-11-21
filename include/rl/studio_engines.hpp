/**
 * @file studio_engines.hpp
 * @brief Backend training engines for RL Studio:
 *        - GridWorldEngine (Tabular Q-Learning vs. SARSA with live alpha, gamma, epsilon, decay, Cliff/Random maps)
 *        - RLtoolsSuiteEngine (RLtools SAC vs. PPO vs. TD3 across 6 continuous environments:
 *          CartPole, Pendulum-v1, Acrobot, MountainCar, Drone2D, and ContinuousGrid2D)
 *        with runtime-tunable learning rate, gamma, exploration/entropy, and replayable rollouts!
 */

#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/acrobot/operations_generic.h>
#include "rl/rltools_cartpole.hpp"
#include "rl/rltools_mountaincar.hpp"
#include "rl/rltools_drone2d.hpp"
#include "rl/rltools_lunarlander.hpp"
#include "rl/rltools_reacher.hpp"
#include "rl/rltools_ballbeam.hpp"
#include "rl/rltools_double_cartpole.hpp"
#include "rl/custom_env.hpp"
#include "rl/reward_shaping.hpp"

namespace rl_tools::rl::environments {
    /**
     * @brief AcrobotSwingup subclass with runtime-tunable velocity & torque reward shaping!
     */
    template <typename T_SPEC>
    struct AcrobotShaped : AcrobotSwingup<T_SPEC> {};
}

namespace rl_tools {
    /**
     * @brief Shaped reward for AcrobotShaped: angle cost + optional velocity penalty (stops crazy spinning!) + torque cost.
     */
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::AcrobotShaped<SPEC>&,
                                   typename rl::environments::AcrobotShaped<SPEC>::Parameters&,
                                   const typename rl::environments::AcrobotShaped<SPEC>::State&,
                                   const Matrix<ACTION_SPEC>& action,
                                   const typename rl::environments::AcrobotShaped<SPEC>::State& next_state, RNG&) {
        using T = typename SPEC::T;
        const auto& rs = studio::reward_shaping();
        typename DEVICE::SPEC::MATH dm;
        T a_cost = math::abs(device.math, rl::environments::acrobot::angle_normalize(dm, next_state.theta_1 - math::PI<T>));
        T b_cost = math::abs(device.math, rl::environments::acrobot::angle_normalize(dm, next_state.theta_2));
        T r = (-a_cost - b_cost) / (T)100.0 * SPEC::PARAMETERS::DT / (T)0.02;
        if (rs.acrobot_vel_penalty_on) {
            T vel_sq = next_state.theta_1_dot * next_state.theta_1_dot + next_state.theta_2_dot * next_state.theta_2_dot;
            r -= static_cast<T>(rs.acrobot_vel_weight) * vel_sq;
        }
        T u = get(action, 0, 0);
        r -= static_cast<T>(rs.acrobot_torque_weight) * (u * u);
        return r;
    }
}

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>

#include "rl/gridworld.hpp"
#include "rl/q_learning_agent.hpp"
#include "rl/sarsa_agent.hpp"
#include "rl/dp_solver.hpp"
#include "rl/bandits_lab.hpp"
#include "rl/taxi_lab.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rlt = rl_tools;

namespace studio {

/**
 * @brief Generic pose + action + reward snapshot for 60 FPS playback and scrubber inspection.
 */
struct RenderFrame {
    std::array<float, 6> s{0, 0, 0, 0, 0, 0}; ///< Environment-specific pose variables.
    std::array<float, 2> a{0, 0};             ///< Applied continuous action(s).
    float reward{0.0f};                       ///< Immediate step reward.
    bool terminated{false};                   ///< Whether this step terminated the episode.
};

/**
 * @brief Time-series telemetry and rollout history for one algorithm.
 */
struct AlgoTelemetry {
    std::string name;
    bool enabled = true;
    int env_steps = 0;
    double wall_ms = 0.0;
    double last_return = 0.0;
    int last_ep_len = 0;

    // Benchmark "Train Until Threshold" results
    bool solved = false;
    int solved_at_steps = -1;
    int solved_at_round = -1;
    double solved_at_ms = -1.0;

    std::vector<double> x_round;
    std::vector<double> x_steps;
    std::vector<double> x_time_s;
    std::vector<double> y_return;
    std::vector<double> y_ep_len;
    std::vector<RenderFrame> rollout;
};

/**
 * @brief Saved snapshot of a previous training run for side-by-side hyperparameter comparison.
 */
struct SavedRun {
    std::string label;
    std::vector<double> x_steps;
    std::vector<double> x_time_s;
    std::vector<double> y_return;
};

/**
 * @brief Runtime-tunable hyperparameters exposed as sliders in RL Studio.
 */
struct ContinuousHyperparams {
    float lr = 0.001f;             ///< Adam learning rate (applied live to Actor & Critic optimizers!).
    float gamma = 0.99f;           ///< Discount factor gamma (applied live to SAC/PPO/TD3!).
    float sac_alpha = 0.2f;        ///< SAC entropy temperature (higher = more exploration).
    float ppo_clip = 0.2f;         ///< PPO ratio clipping epsilon.
    float td3_noise = 0.1f;        ///< TD3 exploration noise std.
    int seed = 42;                 ///< Random seed for reproducibility.
};

/**
 * @brief Abstract interface for a 3-way (SAC, PPO, TD3) continuous RLtools environment suite.
 */
class IContinuousSuite {
public:
    virtual ~IContinuousSuite() = default;
    [[nodiscard]] virtual std::string id() const = 0;
    [[nodiscard]] virtual std::string title() const = 0;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual int env_kind() const = 0; // 0=CartPole, 1=Pendulum, 2=Acrobot, 3=MountainCar, 4=Drone2D, 5=Grid2D
    [[nodiscard]] virtual float default_target_return() const = 0;

    virtual void reset(const ContinuousHyperparams& hp) = 0;
    virtual void apply_hyperparams(const ContinuousHyperparams& hp) = 0;
    virtual void train_step(int sac_steps, int ppo_iters, int td3_steps, const ContinuousHyperparams& hp) = 0;
    virtual void check_threshold(double target_return) = 0;
    /**
     * @brief Evaluates the trained neural Actor policy across a 2D state grid (N x N)
     *        to produce a live Neural Policy Heatmap (Pendulum, MountainCar, ContinuousGrid2D)!
     * @param algo_idx 0=SAC, 1=PPO, 2=TD3.
     * @param grid_n Resolution (e.g. 25x25).
     * @return Flattened grid_n * grid_n vector of continuous action values in [-1, 1].
     */
    virtual std::vector<double> compute_policy_heatmap(int algo_idx, int grid_n) = 0;
    [[nodiscard]] virtual AlgoTelemetry& telemetry(int algo_idx) = 0;
    [[nodiscard]] virtual const AlgoTelemetry& telemetry(int algo_idx) const = 0;
};

/**
 * @brief Templated 3-way RLtools Engine (SAC + PPO + TD3) with live runtime hyperparameter injection.
 */
template <typename ENV, typename FrameExtractor>
class RLtoolsContinuousSuite : public IContinuousSuite {
public:
    using DEVICE = rlt::devices::DefaultCPU;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    struct SAC_PARAMS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
        struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENV::ACTION_DIM> {
            static constexpr TI ACTOR_BATCH_SIZE = 64;
            static constexpr TI CRITIC_BATCH_SIZE = 64;
        };
        static constexpr TI STEP_LIMIT = 500000;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 32;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 32;
    };
    using SAC_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENV, SAC_PARAMS>;
    using SAC_STATE = typename SAC_CONFIG::template State<SAC_CONFIG>;

    struct PPO_PARAMS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
        static constexpr TI BATCH_SIZE = 128;
        static constexpr TI ACTOR_HIDDEN_DIM = 32;
        static constexpr TI CRITIC_HIDDEN_DIM = 32;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
        static constexpr TI N_ENVIRONMENTS = 4;
        static constexpr TI STEP_LIMIT = 10000;
        static constexpr TI EPISODE_STEP_LIMIT = ENV::EPISODE_STEP_LIMIT;
        using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
        using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
        struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01f;
            static constexpr TI N_EPOCHS = 4;
            static constexpr T GAMMA = 0.98f;
            static constexpr T INITIAL_ACTION_STD = 0.8f;
            static constexpr bool NORMALIZE_OBSERVATIONS = false;
        };
    };
    using PPO_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
        TYPE_POLICY, TI, RNG, ENV, PPO_PARAMS,
        rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
    using PPO_STATE = typename PPO_CONFIG::template State<PPO_CONFIG>;

    struct TD3_PARAMS : rlt::rl::algorithms::td3::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
        struct TD3_PARAMETERS : rlt::rl::algorithms::td3::DefaultParameters<TYPE_POLICY, TI> {
            static constexpr TI ACTOR_BATCH_SIZE = 64;
            static constexpr TI CRITIC_BATCH_SIZE = 64;
        };
        static constexpr TI STEP_LIMIT = 500000;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 32;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 32;
    };
    using TD3_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<TYPE_POLICY, TI, RNG, ENV, TD3_PARAMS>;
    using TD3_STATE = typename TD3_CONFIG::template State<TD3_CONFIG>;

    RLtoolsContinuousSuite(std::string id, std::string title, std::string desc, int kind, float target_ret, FrameExtractor ext)
        : id_(std::move(id)), title_(std::move(title)), desc_(std::move(desc)),
          kind_(kind), default_target_(target_ret), extractor_(ext) {
        tel_[0].name = "SAC (Off-Policy Stochastic)";
        tel_[1].name = "PPO (On-Policy Stochastic)";
        tel_[2].name = "TD3 (Off-Policy Deterministic)";
    }

    ~RLtoolsContinuousSuite() override { free_all(); }

    [[nodiscard]] std::string id() const override { return id_; }
    [[nodiscard]] std::string title() const override { return title_; }
    [[nodiscard]] std::string description() const override { return desc_; }
    [[nodiscard]] int env_kind() const override { return kind_; }
    [[nodiscard]] float default_target_return() const override { return default_target_; }

    void check_threshold(double target_return) override {
        for (int k = 0; k < 3; ++k) {
            if (!tel_[k].solved && tel_[k].enabled && tel_[k].last_return >= target_return) {
                tel_[k].solved = true;
                tel_[k].solved_at_steps = tel_[k].env_steps;
                tel_[k].solved_at_round = round_idx_;
                tel_[k].solved_at_ms = tel_[k].wall_ms;
            }
        }
    }

    void apply_hyperparams(const ContinuousHyperparams& hp) override {
        auto set_lr = [&](auto& opt) {
            auto p = rlt::get(device_, opt.parameters, 0);
            p.alpha = hp.lr;
            rlt::set(device_, opt.parameters, p, 0);
        };
        if (sac_) {
            set_lr(sac_->actor_critic.actor_optimizer);
            set_lr(sac_->actor_critic.critic_optimizers[0]);
            set_lr(sac_->actor_critic.critic_optimizers[1]);
            set_lr(sac_->actor_critic.alpha_optimizer);
        }
        if (ppo_) {
            set_lr(ppo_->actor_optimizer);
            set_lr(ppo_->critic_optimizer);
        }
        if (td3_) {
            set_lr(td3_->actor_critic.actor_optimizer);
            set_lr(td3_->actor_critic.critic_optimizers[0]);
            set_lr(td3_->actor_critic.critic_optimizers[1]);
        }
    }

    void reset(const ContinuousHyperparams& hp) override {
        free_all();
        sac_ = new SAC_STATE();
        ppo_ = new PPO_STATE();
        td3_ = new TD3_STATE();

        rlt::malloc(device_, *sac_);
        rlt::malloc(device_, *ppo_);
        rlt::malloc(device_, *td3_);

        rlt::init(device_, *sac_, static_cast<TI>(hp.seed));
        rlt::init(device_, *ppo_, static_cast<TI>(hp.seed));
        rlt::init(device_, *td3_, static_cast<TI>(hp.seed));

        apply_hyperparams(hp);

        bool en0 = tel_[0].enabled, en1 = tel_[1].enabled, en2 = tel_[2].enabled;
        tel_[0] = AlgoTelemetry{"SAC (Off-Policy Stochastic)", en0};
        tel_[1] = AlgoTelemetry{"PPO (On-Policy Stochastic)", en1};
        tel_[2] = AlgoTelemetry{"TD3 (Off-Policy Deterministic)", en2};
        round_idx_ = 0;

        evaluate_all();
    }

    void train_step(int sac_steps, int ppo_iters, int td3_steps, const ContinuousHyperparams& hp) override {
        if (!sac_) reset(hp);
        apply_hyperparams(hp);
        round_idx_++;

        if (tel_[0].enabled && !tel_[0].solved) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < sac_steps; ++i) {
                if (rlt::step(device_, *sac_)) break;
                tel_[0].env_steps++;
            }
            auto t1 = std::chrono::steady_clock::now();
            tel_[0].wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        if (tel_[1].enabled && !tel_[1].solved) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < ppo_iters; ++i) {
                if (rlt::step(device_, *ppo_)) break;
                tel_[1].env_steps += PPO_PARAMS::N_ENVIRONMENTS * PPO_PARAMS::ON_POLICY_RUNNER_STEPS_PER_ENV;
            }
            auto t1 = std::chrono::steady_clock::now();
            tel_[1].wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        if (tel_[2].enabled && !tel_[2].solved) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < td3_steps; ++i) {
                if (rlt::step(device_, *td3_)) break;
                tel_[2].env_steps++;
            }
            auto t1 = std::chrono::steady_clock::now();
            tel_[2].wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        evaluate_all();
    }

    [[nodiscard]] AlgoTelemetry& telemetry(int algo_idx) override { return tel_[algo_idx]; }
    [[nodiscard]] const AlgoTelemetry& telemetry(int algo_idx) const override { return tel_[algo_idx]; }

    std::vector<double> compute_policy_heatmap(int algo_idx, int grid_n) override {
        std::vector<double> out(grid_n * grid_n, 0.0);
        if (!sac_ || !ppo_ || !td3_) return out;

        auto eval_grid = [&](auto& actor) {
            using ACTOR_T = std::remove_reference_t<decltype(actor)>;
            using EVAL_A = typename ACTOR_T::template CHANGE_BATCH_SIZE<TI, 1>;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENV::Observation::DIM>>> obs_t;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENV::ACTION_DIM>>> act_t;
            typename EVAL_A::template Buffer<> buf;
            rlt::malloc(device_, obs_t);
            rlt::malloc(device_, act_t);
            rlt::malloc(device_, buf);
            auto obs_m = rlt::matrix_view(device_, obs_t);
            auto act_m = rlt::matrix_view(device_, act_t);
            RNG rng;
            rlt::init(device_, rng, 7);

            for (int r = 0; r < grid_n; ++r) {
                // ImPlot heatmap row 0 is TOP (y_max), row grid_n-1 is BOTTOM (y_min)
                float ny = 1.0f - (static_cast<float>(r) / (grid_n - 1)) * 2.0f; // +1 down to -1
                for (int c = 0; c < grid_n; ++c) {
                    float nx = -1.0f + (static_cast<float>(c) / (grid_n - 1)) * 2.0f; // -1 to +1
                    rlt::set_all(device_, obs_m, (T)0.0);
                    if (kind_ == 1 && ENV::Observation::DIM >= 3) {
                        // Pendulum: x-axis = theta in [-pi, pi], y-axis = theta_dot in [-8, 8]
                        float th = nx * 3.14159265f;
                        rlt::set(obs_m, 0, 0, std::cos(th));
                        rlt::set(obs_m, 0, 1, std::sin(th));
                        rlt::set(obs_m, 0, 2, ny * 8.0f);
                    } else if (kind_ == 0 && ENV::Observation::DIM >= 4) {
                        // CartPole: x-axis = theta_norm in [-1, 1], y-axis = theta_dot_norm in [-1, 1]
                        rlt::set(obs_m, 0, 2, nx);
                        rlt::set(obs_m, 0, 3, ny);
                    } else if (ENV::Observation::DIM >= 2) {
                        // MountainCar / ContinuousGrid2D / Drone2D / LunarLander: obs[0]=nx, obs[1]=ny
                        rlt::set(obs_m, 0, 0, nx);
                        rlt::set(obs_m, 0, 1, ny);
                    }
                    rlt::evaluate(device_, actor, obs_t, act_t, buf, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
                    out[r * grid_n + c] = std::clamp(static_cast<double>(rlt::get(act_m, 0, 0)), -1.0, 1.0);
                }
            }
            rlt::free(device_, obs_t);
            rlt::free(device_, act_t);
            rlt::free(device_, buf);
        };

        if (algo_idx == 0) eval_grid(rlt::get_actor(*sac_));
        else if (algo_idx == 1) eval_grid(rlt::get_actor(*ppo_));
        else eval_grid(rlt::get_actor(*td3_));
        return out;
    }

private:
    std::string id_, title_, desc_;
    int kind_;
    float default_target_ = 450.0f;
    FrameExtractor extractor_;
    int round_idx_ = 0;

    DEVICE device_;
    SAC_STATE* sac_ = nullptr;
    PPO_STATE* ppo_ = nullptr;
    TD3_STATE* td3_ = nullptr;
    std::array<AlgoTelemetry, 3> tel_;

    void free_all() {
        if (sac_) { rlt::free(device_, *sac_); delete sac_; sac_ = nullptr; }
        if (ppo_) { rlt::free(device_, *ppo_); delete ppo_; ppo_ = nullptr; }
        if (td3_) { rlt::free(device_, *td3_); delete td3_; td3_ = nullptr; }
    }

    template <typename ACTOR_TYPE>
    void rollout_actor(ACTOR_TYPE& actor, AlgoTelemetry& out) {
        out.rollout.clear();
        ENV env;
        typename ENV::Parameters params;
        typename ENV::State state, next_state;
        RNG rng;
        rlt::init(device_, rng, 123);
        rlt::initial_parameters(device_, env, params);
        rlt::initial_state(device_, env, params, state);

        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENV::Observation::DIM>>> obs_t;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENV::ACTION_DIM>>> act_t;
        using EVAL_ACTOR = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, 1>;
        typename EVAL_ACTOR::template Buffer<> buf;
        rlt::malloc(device_, obs_t);
        rlt::malloc(device_, act_t);
        rlt::malloc(device_, buf);

        auto obs_m = rlt::matrix_view(device_, obs_t);
        auto act_m = rlt::matrix_view(device_, act_t);
        rlt::set_all(device_, act_m, (T)0.0);
        out.rollout.push_back(extractor_(state, act_m, 0.0f, false));

        double total_ret = 0.0;
        int steps = 0;
        for (TI t = 0; t < ENV::EPISODE_STEP_LIMIT; ++t) {
            rlt::observe(device_, env, params, state, typename ENV::Observation{}, obs_m, rng);
            rlt::evaluate(device_, actor, obs_t, act_t, buf, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
            rlt::step(device_, env, params, state, act_m, next_state, rng);
            T r = rlt::reward(device_, env, params, state, act_m, next_state, rng);
            total_ret += r;
            steps++;
            bool term = rlt::terminated(device_, env, params, next_state, rng);
            out.rollout.push_back(extractor_(next_state, act_m, static_cast<float>(r), term));
            state = next_state;
            if (term) break;
        }

        rlt::free(device_, obs_t);
        rlt::free(device_, act_t);
        rlt::free(device_, buf);

        out.last_return = total_ret;
        out.last_ep_len = steps;
        out.x_round.push_back(static_cast<double>(round_idx_));
        out.x_steps.push_back(static_cast<double>(out.env_steps));
        out.x_time_s.push_back(out.wall_ms / 1000.0);
        out.y_return.push_back(total_ret);
        out.y_ep_len.push_back(static_cast<double>(steps));
    }

    void evaluate_all() {
        if ((tel_[0].enabled && !tel_[0].solved) || tel_[0].y_return.empty()) rollout_actor(rlt::get_actor(*sac_), tel_[0]);
        if ((tel_[1].enabled && !tel_[1].solved) || tel_[1].y_return.empty()) rollout_actor(rlt::get_actor(*ppo_), tel_[1]);
        if ((tel_[2].enabled && !tel_[2].solved) || tel_[2].y_return.empty()) rollout_actor(rlt::get_actor(*td3_), tel_[2]);
    }
};

/**
 * @brief Tabular GridWorld Studio Engine comparing Off-Policy Q-Learning vs. On-Policy SARSA
 *        with live alpha, gamma, epsilon, GLIE decay, Cliff-Walking & Procedural BFS Maze maps.
 */
struct GridWorldStudio {
    rl::GridWorld env_q{8, 4};
    rl::GridWorld env_s{8, 4};
    rl::QLearningAgent q_agent{32, 4, 0.2, 0.95, 0.2, 42};
    rl::SarsaAgent s_agent{32, 4, 0.2, 0.95, 0.2, 42};
    ValueIterationSolver dp_solver;

    float alpha = 0.20f;
    float gamma = 0.95f;
    float init_epsilon = 0.25f;
    float slip_prob = 0.0f;
    bool use_glie_decay = true;
    float decay_rate = 0.995f;
    int map_seed = 100;
    int paint_brush = 1; // 0=Empty, 1=Wall, 2=Fire Trap
    bool show_dp_overlay = false;

    int episodes = 0;
    std::vector<double> x_ep;
    std::vector<double> q_ret, s_ret;
    std::vector<double> q_loss, s_loss;
    std::vector<double> q_mean_q, s_mean_q;
    std::vector<std::pair<int, int>> q_path; ///< Recorded greedy rollout path (x, y) for Robot animation!
    std::vector<std::pair<int, int>> s_path; ///< Recorded greedy rollout path (x, y) for Robot animation!

    GridWorldStudio() { load_cliff(); }

    /**
     * @brief Records the current greedy policy path for both Q-Learning and SARSA robots so they can animate!
     */
    void record_greedy_paths() {
        auto trace = [](rl::GridWorld& env, const auto& agent, std::vector<std::pair<int, int>>& path) {
            path.clear();
            int s = env.reset();
            path.push_back({env.agent_x(), env.agent_y()});
            for (int step = 0; step < env.width() * env.height() + 10; ++step) {
                auto res = env.step(agent.greedy_action(s));
                path.push_back({env.agent_x(), env.agent_y()});
                s = res.next_state;
                if (res.done()) break;
            }
            env.reset();
        };
        trace(env_q, q_agent, q_path);
        trace(env_s, s_agent, s_path);
    }

    void load_cliff() {
        env_q.set_cliff_layout();
        env_s.set_cliff_layout();
        env_q.set_slip_prob(slip_prob);
        env_s.set_slip_prob(slip_prob);
        reset_agents();
    }

    void load_windy() {
        env_q.set_windy_layout();
        env_s.set_windy_layout();
        env_q.set_slip_prob(slip_prob);
        env_s.set_slip_prob(slip_prob);
        reset_agents();
    }

    void load_random_map() {
        map_seed += 19;
        std::mt19937 rng(map_seed);
        int w = std::uniform_int_distribution<int>(5, 9)(rng);
        int h = std::uniform_int_distribution<int>(4, 7)(rng);
        env_q.randomize_layout(w, h, 0.14, 0.15, map_seed);
        env_s.randomize_layout(w, h, 0.14, 0.15, map_seed);
        env_q.set_slip_prob(slip_prob);
        env_s.set_slip_prob(slip_prob);
        reset_agents();
    }

    void paint_cell(int x, int y) {
        rl::CellType t = (paint_brush == 0) ? rl::CellType::Empty
                       : ((paint_brush == 1) ? rl::CellType::Wall : rl::CellType::Trap);
        env_q.set_cell(x, y, t);
        env_s.set_cell(x, y, t);
        dp_solver.reset(env_q);
        record_greedy_paths();
    }

    void reset_agents() {
        q_agent = rl::QLearningAgent(env_q.num_states(), env_q.num_actions(), alpha, gamma, init_epsilon, 42);
        s_agent = rl::SarsaAgent(env_s.num_states(), env_s.num_actions(), alpha, gamma, init_epsilon, 42);
        env_q.set_slip_prob(slip_prob);
        env_s.set_slip_prob(slip_prob);
        env_q.reset();
        env_s.reset();
        dp_solver.reset(env_q);
        episodes = 0;
        x_ep.clear(); q_ret.clear(); s_ret.clear();
        q_loss.clear(); s_loss.clear(); q_mean_q.clear(); s_mean_q.clear();
        record_greedy_paths();
    }

    template <typename AgentT>
    double mean_max_q(const rl::GridWorld& env, const AgentT& agent) const {
        double sum = 0.0;
        int cnt = 0;
        for (int y = 0; y < env.height(); ++y) {
            for (int x = 0; x < env.width(); ++x) {
                if (env.cell_at(x, y) == rl::CellType::Empty) {
                    int s = y * env.width() + x;
                    sum += agent.q_value(s, static_cast<int>(agent.greedy_action(s)));
                    cnt++;
                }
            }
        }
        return cnt > 0 ? sum / cnt : 0.0;
    }

    void train_episodes(int n) {
        for (int i = 0; i < n; ++i) {
            // Q-Learning episode
            int sq = env_q.reset();
            double rq = 0.0, lq = 0.0;
            int stq = 0;
            while (true) {
                auto a = q_agent.select_action(sq);
                auto res = env_q.step(a);
                lq += q_agent.update(sq, a, res.reward, res.next_state, res.terminated);
                rq += res.reward;
                stq++;
                sq = res.next_state;
                if (res.done()) break;
            }

            // SARSA episode
            int ss = env_s.reset();
            auto as = s_agent.select_action(ss);
            double rs = 0.0, ls = 0.0;
            int sts = 0;
            while (true) {
                auto res = env_s.step(as);
                auto next_a = s_agent.select_action(res.next_state);
                ls += s_agent.update(ss, as, res.reward, res.next_state, next_a, res.terminated);
                rs += res.reward;
                sts++;
                ss = res.next_state;
                as = next_a;
                if (res.done()) break;
            }

            if (use_glie_decay) {
                q_agent.decay_epsilon(decay_rate, 0.02);
                s_agent.decay_epsilon(decay_rate, 0.02);
            }

            episodes++;
            x_ep.push_back(static_cast<double>(episodes));
            double pq = q_ret.empty() ? rq : (0.88 * q_ret.back() + 0.12 * rq);
            double ps = s_ret.empty() ? rs : (0.88 * s_ret.back() + 0.12 * rs);
            q_ret.push_back(pq);
            s_ret.push_back(ps);
            q_loss.push_back(stq > 0 ? lq / stq : 0.0);
            s_loss.push_back(sts > 0 ? ls / sts : 0.0);
            q_mean_q.push_back(mean_max_q(env_q, q_agent));
            s_mean_q.push_back(mean_max_q(env_s, s_agent));
        }
        env_q.reset();
        env_s.reset();
        record_greedy_paths();
    }
};

/**
 * @brief Creates all 6 continuous RLtools suites for RL Studio.
 */
inline std::vector<std::unique_ptr<IContinuousSuite>> create_all_continuous_suites() {
    using T = float;
    using TI = rlt::devices::DefaultCPU::index_t;
    std::vector<std::unique_ptr<IContinuousSuite>> out;

    // 0: CartPole-Continuous
    {
        using ENV = rlt::rl::environments::CartPoleContinuous<
            rlt::rl::environments::cartpole_continuous::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.x; f.s[1] = st.theta; f.s[2] = st.x_dot; f.s[3] = st.theta_dot;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f); f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "cartpole", "CartPole-Continuous", "4D State | 1D Continuous Force | Balance pole upright for 500 steps.", 0, 480.0f, ext));
    }
    // 1: Pendulum-v1 (RLtools built-in)
    {
        using ENV = rlt::rl::environments::Pendulum<rlt::rl::environments::pendulum::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.theta; f.s[1] = st.theta_dot;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f); f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "pendulum", "Pendulum-v1 Swing-Up", "RLtools Built-in | 3D Fourier Obs | Swing up and balance inverted (Return -> 0).", 1, -250.0f, ext));
    }
    // 2: Acrobot Swing-Up (RLtools Built-in + Live Reward Shaping)
    {
        using ENV = rlt::rl::environments::AcrobotShaped<
            rlt::rl::environments::acrobot::Specification<T, TI, rlt::rl::environments::acrobot::EasyParameters<T>>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.theta_1; f.s[1] = st.theta_2;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f); f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "acrobot", "Acrobot 2-Link Swing-Up", "RLtools Built-in | 6D Obs | Under-actuated chaotic 2-link robot arm.", 2, -100.0f, ext));
    }
    // 3: MountainCar-Continuous
    {
        using ENV = rlt::rl::environments::MountainCar<rlt::rl::environments::mountain_car::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.position; f.s[1] = st.velocity;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f); f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "mountaincar", "MountainCar-Continuous", "2D State | 1D Engine Force | Pump momentum to reach flag at x >= 0.45.", 3, 95.0f, ext));
    }
    // 4: Drone2D Hover
    {
        using ENV = rlt::rl::environments::Drone2D<rlt::rl::environments::drone_2d::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.x; f.s[1] = st.y; f.s[2] = st.theta; f.s[3] = st.vx; f.s[4] = st.vy;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.a[1] = std::clamp(rlt::get(act, 0, 1), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "drone2d", "Drone2D Quadrotor Hover", "6D State | 2D Dual-Rotor Thrust | Stabilize and hover at (0, 0) against gravity.", 4, 250.0f, ext));
    }
    // 5: ContinuousGrid2D (Random Trap Arena)
    {
        using ENV = rlt::rl::environments::ContinuousGrid2D<
            rlt::rl::environments::continuous_grid_2d::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f; f.s[0] = st.x; f.s[1] = st.y;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.a[1] = std::clamp(rlt::get(act, 0, 1), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "contgrid2d", "ContinuousGrid2D (Trap Arena)", "2D State | 2D Velocity | Navigate around circular Traps to Goal (+10).", 5, 3.5f, ext));
    }
    // 6: LunarLander (Moon Landing)
    {
        using ENV = rlt::rl::environments::LunarLander<
            rlt::rl::environments::lunar_lander::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f;
            f.s[0] = st.x; f.s[1] = st.y; f.s[2] = st.theta;
            f.s[3] = st.landed ? 1.0f : (st.crashed ? -1.0f : 0.0f);
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.a[1] = std::clamp(rlt::get(act, 0, 1), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "lunarlander", "LunarLander (Moon Landing)", "6D State | Main + RCS Thrusters | Touch down softly on lunar pad (+100).", 6, 150.0f, ext));
    }
    // 7: 2-DOF Robotic Arm Reacher
    {
        using ENV = rlt::rl::environments::Reacher2D<rlt::rl::environments::reacher_2d::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f;
            f.s[0] = st.q1; f.s[1] = st.q2; f.s[2] = st.target_x; f.s[3] = st.target_y;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.a[1] = std::clamp(rlt::get(act, 0, 1), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "reacher", "Reacher-2DOF Robotic Arm", "8D Obs | 2D Joint Torques | Reach the glowing green target dot smoothly!", 7, -18.0f, ext));
    }
    // 8: Ball-and-Beam
    {
        using ENV = rlt::rl::environments::BallBeam<rlt::rl::environments::ball_beam::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f;
            f.s[0] = st.x; f.s[1] = st.x_dot; f.s[2] = st.theta; f.s[3] = st.theta_dot;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "ballbeam", "Ball-and-Beam (Control Theory)", "4D State | Beam Tilt Control | Balance rolling steel ball at beam center!", 8, 220.0f, ext));
    }
    // 9: Inverted Double CartPole
    {
        using ENV = rlt::rl::environments::DoubleCartPole<rlt::rl::environments::double_cartpole::Specification<T, TI>>;
        auto ext = [](const typename ENV::State& st, const auto& act, float r, bool term) {
            RenderFrame f;
            f.s[0] = st.x; f.s[1] = st.theta1; f.s[2] = st.theta2; f.s[3] = st.theta1_dot;
            f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
            f.reward = r; f.terminated = term;
            return f;
        };
        out.push_back(std::make_unique<RLtoolsContinuousSuite<ENV, decltype(ext)>>(
            "doublecartpole", "Inverted Double CartPole (Hard!)", "6D State | 1D Cart Force | Balance TWO stacked poles simultaneously!", 9, 250.0f, ext));
    }
    return out;
}

} // namespace studio
