#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

// Include our custom C++ environment!
#include "rl/custom_env.hpp"

// Include RLtools SAC Loop headers
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#include <iostream>

namespace rlt = rl_tools;

// 1. Device and data types (all known at compile time!)
using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;

// 2. Plug in our custom GridWorld1D environment
using ENV_SPEC = rlt::rl::environments::gridworld_1d::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::GridWorld1D<ENV_SPEC>;

// 3. Configure Soft Actor-Critic (SAC) hyperparameters at compile time
struct LOOP_CORE_PARAMETERS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
        static constexpr TI ACTOR_BATCH_SIZE = 64;
        static constexpr TI CRITIC_BATCH_SIZE = 64;
    };
    static constexpr TI STEP_LIMIT = 3000;      // Total training steps
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;

// 4. Wrap the training loop with periodic Evaluation and Timing steps
struct LOOP_EVAL_PARAMETERS : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, LOOP_CORE_CONFIG> {
    static constexpr TI EVALUATION_INTERVAL = 500;
    static constexpr TI EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main() {
    std::cout << "=== Training SAC on Custom GridWorld1D Environment with RLtools ===\n";
    std::cout << "Target position: x = +0.8 (Start at x = -0.5 during eval)\n\n";

    DEVICE device;
    LOOP_STATE ts;

    // Allocate memory and initialize weights + replay buffer
    rlt::malloc(device, ts);
    rlt::init(device, ts, 42); // seed = 42

    // Run the compile-time optimized training loop!
    while (!rlt::step(device, ts)) {
        // Every 500 steps, LOOP_EVAL_CONFIG automatically evaluates the policy and prints mean return
    }

    rlt::free(device, ts);
    std::cout << "\nTraining complete!\n";
    return 0;
}
