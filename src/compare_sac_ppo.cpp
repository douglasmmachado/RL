/**
 * @file compare_sac_ppo.cpp
 * @brief Side-by-side benchmark comparing RLtools' Off-Policy SAC vs. On-Policy PPO
 *        on the exact same custom ContinuousGrid2D environment (with central Trap & Goal).
 *
 * Prints sample efficiency (total environment steps), wall-clock training time,
 * final evaluation return, and the learned 2D trajectory of each algorithm!
 */

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

// Custom 2D Continuous Navigation Environment
#include "rl/custom_env.hpp"

// RLtools SAC + PPO Loop headers
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#include <chrono>
#include <iomanip>
#include <iostream>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;

using ENV_SPEC = rlt::rl::environments::continuous_grid_2d::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::ContinuousGrid2D<ENV_SPEC>;

// ============================================================================
// 1. Off-Policy SAC Configuration (uses Replay Buffer -> very sample-efficient!)
// ============================================================================
struct SAC_LOOP_PARAMS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
        static constexpr TI ACTOR_BATCH_SIZE = 64;
        static constexpr TI CRITIC_BATCH_SIZE = 64;
    };
    static constexpr TI STEP_LIMIT = 4000; // 4,000 total environment steps
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
};
using SAC_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, SAC_LOOP_PARAMS>;
struct SAC_EVAL_PARAMS : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, SAC_CORE_CONFIG> {
    static constexpr TI EVALUATION_INTERVAL = 1000;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = SAC_LOOP_PARAMS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using SAC_LOOP_CONFIG = rlt::rl::loop::steps::evaluation::Config<SAC_CORE_CONFIG, SAC_EVAL_PARAMS>;
using SAC_LOOP_STATE = SAC_LOOP_CONFIG::State<SAC_LOOP_CONFIG>;

// ============================================================================
// 2. On-Policy PPO Configuration (uses parallel rollout batches -> super fast wall-clock!)
// ============================================================================
struct PPO_LOOP_PARAMS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    static constexpr TI BATCH_SIZE = 128;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
    static constexpr TI N_ENVIRONMENTS = 4; // 4 * 256 = 1,024 env steps per PPO iteration
    static constexpr TI TOTAL_STEP_LIMIT = 40960; // ~40 PPO iterations
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS);
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
    using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
    struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01f;
        static constexpr TI N_EPOCHS = 4;
        static constexpr T GAMMA = 0.95f;
        static constexpr T INITIAL_ACTION_STD = 0.8f;
        static constexpr bool NORMALIZE_OBSERVATIONS = false;
    };
};
using PPO_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
    TYPE_POLICY, TI, RNG, ENVIRONMENT, PPO_LOOP_PARAMS,
    rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
struct PPO_EVAL_PARAMS : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, PPO_CORE_CONFIG> {
    static constexpr TI EVALUATION_INTERVAL = 10;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = PPO_LOOP_PARAMS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using PPO_LOOP_CONFIG = rlt::rl::loop::steps::evaluation::Config<PPO_CORE_CONFIG, PPO_EVAL_PARAMS>;
using PPO_LOOP_STATE = PPO_LOOP_CONFIG::State<PPO_LOOP_CONFIG>;

/**
 * @brief Runs a deterministic evaluation rollout of a trained RLtools Actor and prints its 2D path.
 * @tparam ACTOR_TYPE The RLtools Actor neural network type.
 * @param device RLtools CPU device.
 * @param actor Trained actor network.
 * @param algo_name Display name ("SAC" or "PPO").
 * @return Total episode return achieved on the evaluation rollout.
 */
template <typename ACTOR_TYPE>
double evaluate_and_print_trajectory(DEVICE& device, ACTOR_TYPE& actor, const std::string& algo_name) {
    ENVIRONMENT env;
    typename ENVIRONMENT::Parameters params;
    typename ENVIRONMENT::State state, next_state;
    RNG rng;
    rlt::init(device, rng, 123);
    rlt::initial_parameters(device, env, params);
    rlt::initial_state(device, env, params, state);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::Observation::DIM>>> obs_tensor;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::ACTION_DIM>>> action_tensor;
    using EVAL_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, 1>;
    typename EVAL_ACTOR_TYPE::template Buffer<> actor_buffer;
    rlt::malloc(device, obs_tensor);
    rlt::malloc(device, action_tensor);
    rlt::malloc(device, actor_buffer);

    double total_return = 0.0;
    int steps = 0;

    std::cout << "\n[" << algo_name << " Trained Policy Rollout] Start=(-0.80, -0.80) | Trap=(0.00, 0.00) | Goal=(+0.80, +0.80)\n";
    std::cout << std::fixed << std::setprecision(2);

    auto obs_mat = rlt::matrix_view(device, obs_tensor);
    auto act_mat = rlt::matrix_view(device, action_tensor);

    for (TI t = 0; t < ENVIRONMENT::EPISODE_STEP_LIMIT; ++t) {
        rlt::observe(device, env, params, state, typename ENVIRONMENT::Observation{}, obs_mat, rng);
        rlt::evaluate(device, actor, obs_tensor, action_tensor, actor_buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        rlt::step(device, env, params, state, act_mat, next_state, rng);
        T r = rlt::reward(device, env, params, state, act_mat, next_state, rng);
        total_return += r;
        steps++;

        if (t < 8 || rlt::terminated(device, env, params, next_state, rng)) {
            std::cout << "  Step " << std::setw(2) << steps
                      << ": pos=(" << std::setw(5) << next_state.x << ", " << std::setw(5) << next_state.y << ")"
                      << " | act=(" << std::setw(5) << rlt::get(act_mat, 0, 0) << ", " << std::setw(5) << rlt::get(act_mat, 0, 1) << ")"
                      << " | r=" << std::setw(6) << r << "\n";
        }
        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) {
            std::cout << "  --> Terminated at step " << steps
                      << (r > 0 ? " (REACHED GOAL!)" : " (HIT TRAP!)") << "\n";
            break;
        }
    }

    rlt::free(device, obs_tensor);
    rlt::free(device, action_tensor);
    rlt::free(device, actor_buffer);
    return total_return;
}

int main() {
    DEVICE device;

    std::cout << "====================================================================\n";
    std::cout << "  RLtools Deep RL Benchmark: Off-Policy SAC vs. On-Policy PPO\n";
    std::cout << "  Environment: ContinuousGrid2D (2D position, 2D velocity, Trap & Goal)\n";
    std::cout << "====================================================================\n\n";

    // --- 1. Train Soft Actor-Critic (SAC) ---
    std::cout << ">>> [1/2] Training Off-Policy SAC (" << SAC_LOOP_PARAMS::STEP_LIMIT << " env steps)...\n";
    auto t0_sac = std::chrono::steady_clock::now();
    SAC_LOOP_STATE sac_ts;
    rlt::malloc(device, sac_ts);
    rlt::init(device, sac_ts, 42);
    while (!rlt::step(device, sac_ts)) {}
    auto t1_sac = std::chrono::steady_clock::now();
    double sac_ms = std::chrono::duration<double, std::milli>(t1_sac - t0_sac).count();
    double sac_ret = evaluate_and_print_trajectory(device, rlt::get_actor(sac_ts), "SAC");
    rlt::free(device, sac_ts);

    // --- 2. Train Proximal Policy Optimization (PPO) ---
    std::cout << "\n>>> [2/2] Training On-Policy PPO (" << PPO_LOOP_PARAMS::TOTAL_STEP_LIMIT << " env steps across 4 parallel envs)...\n";
    auto t0_ppo = std::chrono::steady_clock::now();
    PPO_LOOP_STATE ppo_ts;
    rlt::malloc(device, ppo_ts);
    rlt::init(device, ppo_ts, 42);
    while (!rlt::step(device, ppo_ts)) {}
    auto t1_ppo = std::chrono::steady_clock::now();
    double ppo_ms = std::chrono::duration<double, std::milli>(t1_ppo - t0_ppo).count();
    double ppo_ret = evaluate_and_print_trajectory(device, rlt::get_actor(ppo_ts), "PPO");
    rlt::free(device, ppo_ts);

    // --- 3. Final Comparison Summary Table ---
    std::cout << "\n====================================================================\n";
    std::cout << "  COMPARISON SUMMARY (ContinuousGrid2D)\n";
    std::cout << "====================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Algorithm | Type       | Env Steps | Wall Time  | Final Eval Return\n";
    std::cout << "  ----------|------------|-----------|------------|------------------\n";
    std::cout << "  SAC       | Off-Policy | " << std::setw(9) << SAC_LOOP_PARAMS::STEP_LIMIT
              << " | " << std::setw(7) << sac_ms << " ms | " << std::setw(8) << sac_ret << "\n";
    std::cout << "  PPO       | On-Policy  | " << std::setw(9) << PPO_LOOP_PARAMS::TOTAL_STEP_LIMIT
              << " | " << std::setw(7) << ppo_ms << " ms | " << std::setw(8) << ppo_ret << "\n";
    std::cout << "====================================================================\n";

    return 0;
}
