/**
 * @file visualize_sac_ppo.cpp
 * @brief Native Windows GUI Dashboard comparing RLtools Off-Policy SAC vs. On-Policy PPO
 *        on the ContinuousGrid2D environment!
 *
 * Visualizes:
 * - Left Panel:  Continuous 2D Arena [-1, 1]^2 with circular Goal (+0.8, +0.8), circular Trap (0, 0),
 *                animated SAC (Cyan) and PPO (Magenta) agents, and their full smooth trajectories!
 * - Right Panel: Live Training Comparison Charts:
 *                1. Evaluation Return vs. Training Progress (SAC Cyan vs. PPO Magenta)
 *                2. Episode Length (Steps to Goal) vs. Training Progress
 *
 * Controls:
 * - [T]: Train BOTH SAC (+1,000 env steps) and PPO (+5 PPO iterations / +5,120 env steps) and update plots & trajectories!
 * - [Space]: Replay animated trajectory race between trained SAC (Cyan) & PPO (Magenta) agents!
 * - [C]: Reset / Re-initialize neural network weights & plots from scratch
 * - [Esc]: Quit
 */

#ifndef UNICODE
#define UNICODE
#endif

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

#include "rl/custom_env.hpp"

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>

#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;

namespace {

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;

using ENV_SPEC = rlt::rl::environments::continuous_grid_2d::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::ContinuousGrid2D<ENV_SPEC>;
using PARAMS = typename ENVIRONMENT::Parameters;

// ============================================================================
// RLtools SAC & PPO Core Loop Configs (high step limits so user can press [T] repeatedly!)
// ============================================================================
struct SAC_LOOP_PARAMS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
        static constexpr TI ACTOR_BATCH_SIZE = 64;
        static constexpr TI CRITIC_BATCH_SIZE = 64;
    };
    static constexpr TI STEP_LIMIT = 100000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
};
using SAC_LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, SAC_LOOP_PARAMS>;
using SAC_LOOP_STATE = SAC_LOOP_CONFIG::State<SAC_LOOP_CONFIG>;

struct PPO_LOOP_PARAMS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    static constexpr TI BATCH_SIZE = 128;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
    static constexpr TI N_ENVIRONMENTS = 4; // 1024 env steps per PPO iteration
    static constexpr TI STEP_LIMIT = 2000;
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
using PPO_LOOP_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
    TYPE_POLICY, TI, RNG, ENVIRONMENT, PPO_LOOP_PARAMS,
    rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
using PPO_LOOP_STATE = PPO_LOOP_CONFIG::State<PPO_LOOP_CONFIG>;

/// @brief A single 2D coordinate point in a rollout trajectory.
struct Point2D {
    float x;
    float y;
};

/// @brief Global RLtools device, training states, trajectories, and metrics.
DEVICE g_device;
SAC_LOOP_STATE* g_sac_ts = nullptr;
PPO_LOOP_STATE* g_ppo_ts = nullptr;

std::vector<Point2D> g_sac_traj;
std::vector<Point2D> g_ppo_traj;
double g_sac_last_return = 0.0;
double g_ppo_last_return = 0.0;
double g_sac_total_ms = 0.0;
double g_ppo_total_ms = 0.0;
int g_sac_env_steps = 0;
int g_ppo_env_steps = 0;

std::vector<double> g_hist_sac_return;
std::vector<double> g_hist_ppo_return;
std::vector<double> g_hist_sac_len;
std::vector<double> g_hist_ppo_len;

std::size_t g_anim_step = 0;

enum class ArenaMode {
    Idle = 0,
    RaceAnim = 1,
    LiveLearning = 2
};
ArenaMode g_mode = ArenaMode::Idle;
unsigned int g_map_seed = 77;

/**
 * @brief Procedurally generates a new random ContinuousGrid2D map (random Goal quadrant + 2 to 4 random Traps)
 *        ensuring traps never overlap the Start or Goal!
 */
void RandomizeContinuousMap() {
    g_map_seed += 23;
    std::mt19937 rng(g_map_seed);
    std::uniform_real_distribution<float> pos_dist(-0.6f, 0.6f);
    std::uniform_real_distribution<float> rad_dist(0.16f, 0.25f);
    std::uniform_int_distribution<int> count_dist(2, 4);
    std::uniform_int_distribution<int> corner_dist(0, 3);

    auto& m = rlt::rl::environments::continuous_grid_2d::active_map();
    // Pick random opposite corners for Start and Goal
    int c = corner_dist(rng);
    m.start_x = (c == 0 || c == 2) ? -0.8f : 0.8f;
    m.start_y = (c == 0 || c == 1) ? -0.8f : 0.8f;
    m.target_x = -m.start_x;
    m.target_y = -m.start_y;
    m.goal_radius = 0.15f;

    m.num_traps = count_dist(rng);
    for (int i = 0; i < m.num_traps; ++i) {
        for (int attempt = 0; attempt < 50; ++attempt) {
            float tx = pos_dist(rng);
            float ty = pos_dist(rng);
            float tr = rad_dist(rng);
            float d_start = std::hypot(tx - m.start_x, ty - m.start_y);
            float d_goal = std::hypot(tx - m.target_x, ty - m.target_y);
            if (d_start > tr + 0.35f && d_goal > tr + 0.35f) {
                m.trap_x[i] = tx;
                m.trap_y[i] = ty;
                m.trap_r[i] = tr;
                break;
            }
        }
    }
}

/**
 * @brief Evaluates an RLtools Actor on ContinuousGrid2D from (-0.8, -0.8) and records its 2D path.
 * @tparam ACTOR_TYPE Actor neural network type.
 * @param actor Trained actor network.
 * @param out_traj Output vector populated with (x, y) coordinates at each step.
 * @return Total episode return achieved.
 */
template <typename ACTOR_TYPE>
double RolloutPolicy(ACTOR_TYPE& actor, std::vector<Point2D>& out_traj) {
    out_traj.clear();
    ENVIRONMENT env;
    PARAMS params;
    typename ENVIRONMENT::State state, next_state;
    RNG rng;
    rlt::init(g_device, rng, 123);
    rlt::initial_parameters(g_device, env, params);
    rlt::initial_state(g_device, env, params, state);
    out_traj.push_back({state.x, state.y});

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::Observation::DIM>>> obs_t;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::ACTION_DIM>>> act_t;
    using EVAL_ACTOR = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, 1>;
    typename EVAL_ACTOR::template Buffer<> buf;
    rlt::malloc(g_device, obs_t);
    rlt::malloc(g_device, act_t);
    rlt::malloc(g_device, buf);

    auto obs_m = rlt::matrix_view(g_device, obs_t);
    auto act_m = rlt::matrix_view(g_device, act_t);

    double total_return = 0.0;
    for (TI t = 0; t < ENVIRONMENT::EPISODE_STEP_LIMIT; ++t) {
        rlt::observe(g_device, env, params, state, typename ENVIRONMENT::Observation{}, obs_m, rng);
        rlt::evaluate(g_device, actor, obs_t, act_t, buf, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        rlt::step(g_device, env, params, state, act_m, next_state, rng);
        total_return += rlt::reward(g_device, env, params, state, act_m, next_state, rng);
        out_traj.push_back({next_state.x, next_state.y});
        state = next_state;
        if (rlt::terminated(g_device, env, params, state, rng)) {
            break;
        }
    }

    rlt::free(g_device, obs_t);
    rlt::free(g_device, act_t);
    rlt::free(g_device, buf);
    return total_return;
}

/**
 * @brief Evaluates both SAC and PPO actors and records their returns and episode lengths.
 */
void EvaluateBothAndRecord() {
    g_sac_last_return = RolloutPolicy(rlt::get_actor(*g_sac_ts), g_sac_traj);
    g_ppo_last_return = RolloutPolicy(rlt::get_actor(*g_ppo_ts), g_ppo_traj);

    g_hist_sac_return.push_back(g_sac_last_return);
    g_hist_ppo_return.push_back(g_ppo_last_return);
    g_hist_sac_len.push_back(static_cast<double>(g_sac_traj.size() > 0 ? g_sac_traj.size() - 1 : 0));
    g_hist_ppo_len.push_back(static_cast<double>(g_ppo_traj.size() > 0 ? g_ppo_traj.size() - 1 : 0));
    g_anim_step = std::max(g_sac_traj.size(), g_ppo_traj.size());
}

/**
 * @brief Allocates and initializes fresh SAC and PPO training states.
 */
void InitTrainers() {
    if (g_sac_ts) {
        rlt::free(g_device, *g_sac_ts);
        delete g_sac_ts;
    }
    if (g_ppo_ts) {
        rlt::free(g_device, *g_ppo_ts);
        delete g_ppo_ts;
    }

    g_sac_ts = new SAC_LOOP_STATE();
    g_ppo_ts = new PPO_LOOP_STATE();
    rlt::malloc(g_device, *g_sac_ts);
    rlt::malloc(g_device, *g_ppo_ts);
    rlt::init(g_device, *g_sac_ts, 42);
    rlt::init(g_device, *g_ppo_ts, 42);

    g_sac_total_ms = 0.0;
    g_ppo_total_ms = 0.0;
    g_sac_env_steps = 0;
    g_ppo_env_steps = 0;
    g_hist_sac_return.clear();
    g_hist_ppo_return.clear();
    g_hist_sac_len.clear();
    g_hist_ppo_len.clear();

    EvaluateBothAndRecord();
}

/**
 * @brief Advances a fast micro-batch of BOTH SAC (+200 env steps) and PPO (+1 iteration = +1024 env steps)
 *        for smooth Live Learning updates!
 */
void TrainLiveTick() {
    // 1. Train SAC for 200 environment steps (~200ms)
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i) {
        if (rlt::step(g_device, *g_sac_ts)) break;
        g_sac_env_steps++;
    }
    auto t1 = std::chrono::steady_clock::now();
    g_sac_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Train PPO for 1 iteration (4 envs * 256 steps = 1024 env steps, ~25ms)
    auto t2 = std::chrono::steady_clock::now();
    if (!rlt::step(g_device, *g_ppo_ts)) {
        g_ppo_env_steps += PPO_LOOP_PARAMS::N_ENVIRONMENTS * PPO_LOOP_PARAMS::ON_POLICY_RUNNER_STEPS_PER_ENV;
    }
    auto t3 = std::chrono::steady_clock::now();
    g_ppo_total_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

    EvaluateBothAndRecord();
}

/**
 * @brief Advances both SAC (+1000 env steps) and PPO (+5 iterations = +5120 env steps).
 */
void TrainOneRound() {
    // 1. Train SAC for 1000 environment steps
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        if (rlt::step(g_device, *g_sac_ts)) break;
        g_sac_env_steps++;
    }
    auto t1 = std::chrono::steady_clock::now();
    g_sac_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Train PPO for 5 iterations (5 * 4 envs * 256 steps = 5120 env steps)
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) {
        if (rlt::step(g_device, *g_ppo_ts)) break;
        g_ppo_env_steps += PPO_LOOP_PARAMS::N_ENVIRONMENTS * PPO_LOOP_PARAMS::ON_POLICY_RUNNER_STEPS_PER_ENV;
    }
    auto t3 = std::chrono::steady_clock::now();
    g_ppo_total_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

    EvaluateBothAndRecord();
}

/**
 * @brief Converts continuous arena coordinate in [-1, 1] to pixel coordinate inside `arena_rc`.
 */
POINT WorldToScreen(float wx, float wy, const RECT& arena_rc) {
    int w = arena_rc.right - arena_rc.left;
    int h = arena_rc.bottom - arena_rc.top;
    int px = arena_rc.left + static_cast<int>(((wx + 1.0f) * 0.5f) * w);
    int py = arena_rc.bottom - static_cast<int>(((wy + 1.0f) * 0.5f) * h);
    return POINT{px, py};
}

/**
 * @brief Draws a dual-series comparison chart (SAC Cyan vs. PPO Magenta).
 */
void DrawChart(HDC hdc, const RECT& plot_rect,
               const std::vector<double>& data_sac,
               const std::vector<double>& data_ppo,
               const std::wstring& title) {
    HBRUSH bg = CreateSolidBrush(RGB(32, 38, 50));
    FillRect(hdc, &plot_rect, bg);
    DeleteObject(bg);

    HPEN bp = CreatePen(PS_SOLID, 1, RGB(65, 75, 95));
    HGDIOBJ op = SelectObject(hdc, bp);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, plot_rect.left, plot_rect.top, plot_rect.right, plot_rect.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(bp);

    RECT trc{plot_rect.left + 10, plot_rect.top + 6, plot_rect.right - 10, plot_rect.top + 26};
    SetTextColor(hdc, RGB(225, 232, 245));
    DrawTextW(hdc, title.c_str(), -1, &trc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    if (data_sac.empty() && data_ppo.empty()) return;

    double min_v = 1e9, max_v = -1e9;
    for (double v : data_sac) { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
    for (double v : data_ppo) { min_v = std::min(min_v, v); max_v = std::max(max_v, v); }
    if (std::abs(max_v - min_v) < 1e-4) { max_v += 1.0; min_v -= 1.0; }

    std::wstringstream leg;
    leg << std::fixed << std::setprecision(1)
        << L"SAC: " << data_sac.back() << L"   |   PPO: " << data_ppo.back();
    SetTextColor(hdc, RGB(200, 215, 235));
    DrawTextW(hdc, leg.str().c_str(), -1, &trc, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    int il = plot_rect.left + 16;
    int ir = plot_rect.right - 16;
    int it = plot_rect.top + 34;
    int ib = plot_rect.bottom - 14;
    int w = std::max(1, ir - il);
    int h = std::max(1, ib - it);

    if (min_v < 0.0 && max_v > 0.0) {
        int zy = ib - static_cast<int>(((0.0 - min_v) / (max_v - min_v)) * h);
        HPEN zp = CreatePen(PS_DOT, 1, RGB(90, 100, 120));
        op = SelectObject(hdc, zp);
        MoveToEx(hdc, il, zy, nullptr);
        LineTo(hdc, ir, zy);
        SelectObject(hdc, op);
        DeleteObject(zp);
    }

    auto draw_curve = [&](const std::vector<double>& v, COLORREF col) {
        if (v.empty()) return;
        HPEN cp = CreatePen(PS_SOLID, 2, col);
        HGDIOBJ old = SelectObject(hdc, cp);
        for (std::size_t i = 0; i < v.size(); ++i) {
            int px = il + (v.size() > 1 ? static_cast<int>((static_cast<double>(i) / (v.size() - 1)) * w) : w / 2);
            int py = ib - static_cast<int>(((v[i] - min_v) / (max_v - min_v)) * h);
            if (i == 0) MoveToEx(hdc, px, py, nullptr);
            else LineTo(hdc, px, py);
        }
        SelectObject(hdc, old);
        DeleteObject(cp);
    };

    draw_curve(data_sac, RGB(56, 189, 248));  // Cyan = SAC
    draw_curve(data_ppo, RGB(232, 121, 249)); // Magenta = PPO
}

/**
 * @brief Renders the Continuous 2D Arena (left) and SAC vs. PPO comparison charts (right).
 */
void DrawDashboard(HDC hdc, const RECT& client_rect) {
    int win_w = client_rect.right - client_rect.left;
    int win_h = client_rect.bottom - client_rect.top;

    HBRUSH bg = CreateSolidBrush(RGB(20, 24, 32));
    FillRect(hdc, &client_rect, bg);
    DeleteObject(bg);
    SetBkMode(hdc, TRANSPARENT);

    int pad = 18;
    int hud_h = 98;

    // 1. Top HUD
    std::wstring mode_txt = L"Idle";
    if (g_mode == ArenaMode::LiveLearning) mode_txt = L"LIVE LEARNING (Press L to Pause)";
    else if (g_mode == ArenaMode::RaceAnim) mode_txt = L"POLICY RACE";

    std::wstringstream top_ss;
    top_ss << std::fixed << std::setprecision(1)
           << L"Mode: " << mode_txt
           << L"   |   SAC (Cyan): " << g_sac_env_steps << L" steps (" << g_sac_total_ms << L" ms)"
           << L"   |   PPO (Magenta): " << g_ppo_env_steps << L" steps (" << g_ppo_total_ms << L" ms)";
    RECT hud1{pad, 10, win_w - pad, 38};
    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextW(hdc, top_ss.str().c_str(), -1, &hud1, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl1 =
        L"[L]: Toggle LIVE Learning   |   [M]: New Random Map (2-4 Traps!)   |   [T]: +1 Big Round   |   [Space]: Race Anim   |   [C]: Reset   |   [Esc]: Quit";
    RECT hud2{pad, 44, win_w - pad, 72};
    SetTextColor(hdc, RGB(125, 211, 252));
    DrawTextW(hdc, ctrl1.c_str(), -1, &hud2, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    // 2. Left Panel: Continuous 2D Square Arena [-1, 1]^2
    int content_top = hud_h;
    int content_h = win_h - content_top - pad;
    int left_w = (win_w - 3 * pad) * 52 / 100;
    int arena_size = std::min(left_w, content_h);
    int ax = pad + (left_w - arena_size) / 2;
    int ay = content_top + (content_h - arena_size) / 2;
    RECT arena_rc{ax, ay, ax + arena_size, ay + arena_size};

    HBRUSH arena_bg = CreateSolidBrush(RGB(30, 36, 48));
    FillRect(hdc, &arena_rc, arena_bg);
    DeleteObject(arena_bg);

    // Subtle coordinate grid lines
    HPEN grid_pen = CreatePen(PS_SOLID, 1, RGB(44, 52, 68));
    HGDIOBJ op = SelectObject(hdc, grid_pen);
    for (int i = 1; i < 4; ++i) {
        int gx = arena_rc.left + (arena_size * i) / 4;
        int gy = arena_rc.top + (arena_size * i) / 4;
        MoveToEx(hdc, gx, arena_rc.top, nullptr);
        LineTo(hdc, gx, arena_rc.bottom);
        MoveToEx(hdc, arena_rc.left, gy, nullptr);
        LineTo(hdc, arena_rc.right, gy);
    }
    SelectObject(hdc, op);
    DeleteObject(grid_pen);

    // Arena border
    HPEN border_pen = CreatePen(PS_SOLID, 2, RGB(80, 95, 120));
    op = SelectObject(hdc, border_pen);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, arena_rc.left, arena_rc.top, arena_rc.right, arena_rc.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(border_pen);

    auto draw_circle = [&](float wx, float wy, float radius, COLORREF fill_c, COLORREF edge_c, const wchar_t* label) {
        POINT c = WorldToScreen(wx, wy, arena_rc);
        int r_px = static_cast<int>((radius * 0.5f) * arena_size);
        HBRUSH br = CreateSolidBrush(fill_c);
        HPEN pn = CreatePen(PS_SOLID, 2, edge_c);
        HGDIOBJ old_b = SelectObject(hdc, br);
        HGDIOBJ old_p = SelectObject(hdc, pn);
        Ellipse(hdc, c.x - r_px, c.y - r_px, c.x + r_px, c.y + r_px);
        SelectObject(hdc, old_b);
        SelectObject(hdc, old_p);
        DeleteObject(br);
        DeleteObject(pn);
        if (label) {
            RECT lrc{c.x - 50, c.y - 12, c.x + 50, c.y + 12};
            SetTextColor(hdc, RGB(255, 255, 255));
            DrawTextW(hdc, label, -1, &lrc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }
    };

    // Draw Start, all active Traps, and Goal circles from active_map()
    const auto& amap = rlt::rl::environments::continuous_grid_2d::active_map();
    draw_circle(amap.start_x, amap.start_y, 0.08f, RGB(70, 82, 105), RGB(160, 175, 200), L"START");
    for (int i = 0; i < amap.num_traps; ++i) {
        draw_circle(amap.trap_x[i], amap.trap_y[i], amap.trap_r[i], RGB(185, 45, 45), RGB(255, 130, 130), L"TRAP");
    }
    draw_circle(amap.target_x, amap.target_y, amap.goal_radius, RGB(34, 139, 87), RGB(120, 240, 175), L"GOAL");

    // Draw trajectories & agent heads for SAC (Cyan) and PPO (Magenta)
    auto draw_trajectory = [&](const std::vector<Point2D>& traj, COLORREF col) {
        if (traj.empty()) return;
        std::size_t limit = std::min(g_anim_step, traj.size());
        if (limit == 0) limit = 1;

        HPEN tp = CreatePen(PS_SOLID, 3, col);
        HGDIOBJ old_p = SelectObject(hdc, tp);
        for (std::size_t i = 0; i < limit; ++i) {
            POINT pt = WorldToScreen(traj[i].x, traj[i].y, arena_rc);
            if (i == 0) MoveToEx(hdc, pt.x, pt.y, nullptr);
            else LineTo(hdc, pt.x, pt.y);
        }
        SelectObject(hdc, old_p);
        DeleteObject(tp);

        // Agent head dot at current animation step
        POINT head = WorldToScreen(traj[limit - 1].x, traj[limit - 1].y, arena_rc);
        HBRUSH hb = CreateSolidBrush(col);
        HPEN hp = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
        HGDIOBJ old_b = SelectObject(hdc, hb);
        old_p = SelectObject(hdc, hp);
        Ellipse(hdc, head.x - 9, head.y - 9, head.x + 9, head.y + 9);
        SelectObject(hdc, old_b);
        SelectObject(hdc, old_p);
        DeleteObject(hb);
        DeleteObject(hp);
    };

    draw_trajectory(g_sac_traj, RGB(56, 189, 248));  // Cyan = SAC
    draw_trajectory(g_ppo_traj, RGB(232, 121, 249)); // Magenta = PPO

    // 3. Right Panel: 2 Comparison Charts
    int rx = 2 * pad + left_w;
    int right_w = (win_w - 3 * pad) - left_w;
    int gap = 16;
    int chart_h = (content_h - gap) / 2;

    RECT p1{rx, content_top, rx + right_w, content_top + chart_h};
    RECT p2{rx, content_top + chart_h + gap, rx + right_w, content_top + content_h};

    DrawChart(hdc, p1, g_hist_sac_return, g_hist_ppo_return,
              L"1. Eval Return per Training Round (Cyan=SAC, Magenta=PPO)");
    DrawChart(hdc, p2, g_hist_sac_len, g_hist_ppo_len,
              L"2. Steps to Goal / Episode Length (Lower is Faster!)");
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN: {
            switch (wParam) {
                case 'L':
                    if (g_mode == ArenaMode::LiveLearning) {
                        g_mode = ArenaMode::Idle;
                        KillTimer(hwnd, 1);
                    } else {
                        g_mode = ArenaMode::LiveLearning;
                        SetTimer(hwnd, 1, 60, nullptr); // Smooth live learning ticks!
                    }
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'M':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    RandomizeContinuousMap();
                    InitTrainers();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'T':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    TrainOneRound();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_SPACE:
                    g_anim_step = 1;
                    g_mode = ArenaMode::RaceAnim;
                    SetTimer(hwnd, 1, 90, nullptr);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'C':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    InitTrainers();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_ESCAPE:
                    PostQuitMessage(0);
                    break;
            }
            return 0;
        }

        case WM_TIMER: {
            if (g_mode == ArenaMode::LiveLearning) {
                TrainLiveTick();
                InvalidateRect(hwnd, nullptr, FALSE);
            } else if (g_mode == ArenaMode::RaceAnim) {
                std::size_t max_len = std::max(g_sac_traj.size(), g_ppo_traj.size());
                if (g_anim_step < max_len) {
                    g_anim_step++;
                } else {
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                }
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;
        }

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT rc;
            GetClientRect(hwnd, &rc);
            HDC mem_dc = CreateCompatibleDC(hdc);
            HBITMAP mem_bmp = CreateCompatibleBitmap(hdc, rc.right - rc.left, rc.bottom - rc.top);
            HGDIOBJ old_bmp = SelectObject(mem_dc, mem_bmp);

            DrawDashboard(mem_dc, rc);

            BitBlt(hdc, 0, 0, rc.right - rc.left, rc.bottom - rc.top, mem_dc, 0, 0, SRCCOPY);
            SelectObject(mem_dc, old_bmp);
            DeleteObject(mem_bmp);
            DeleteDC(mem_dc);

            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_SIZE:
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;

        case WM_DESTROY:
            if (g_sac_ts) { rlt::free(g_device, *g_sac_ts); delete g_sac_ts; g_sac_ts = nullptr; }
            if (g_ppo_ts) { rlt::free(g_device, *g_ppo_ts); delete g_ppo_ts; g_ppo_ts = nullptr; }
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

} // namespace

int main() {
    InitTrainers();

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    const wchar_t CLASS_NAME[] = L"RLtoolsSacPpoArenaClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"RLtools C++ Deep RL Arena: SAC (Off-Policy) vs. PPO (On-Policy)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 780,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hwnd) {
        return 1;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return 0;
}
