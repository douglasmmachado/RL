/**
 * @file visualize_cartpole_rltools.cpp
 * @brief Native Windows GUI Arena comparing RLtools SAC vs. PPO vs. TD3 in parallel on CartPole!
 *
 * Layout:
 * - Left Panel:  3 stacked live CartPole physics stages (from top to bottom):
 *                  1. SAC (Cyan — Off-Policy Stochastic)
 *                  2. PPO (Magenta — On-Policy Stochastic)
 *                  3. TD3 (Amber — Off-Policy Deterministic)
 * - Right Panel: Live 3-Curve Learning Comparison Plot (Steps Balanced out of 500)
 *                + Wall-Clock Time & Step Count Telemetry.
 *
 * Controls:
 * - [L]: Toggle LIVE Learning (trains SAC, PPO, and TD3 in micro-batches & updates plots live!)
 * - [T]: Fast-Train +1 Round (SAC +500 steps, PPO +4096 steps, TD3 +500 steps)
 * - [Space]: Watch all 3 trained CartPole agents balance in real time (50 Hz) simultaneously!
 * - [C]: Reset all 3 neural networks & plots
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
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "rl/rltools_cartpole.hpp"

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>

#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
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

using ENV_SPEC = rlt::rl::environments::cartpole_continuous::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::CartPoleContinuous<ENV_SPEC>;
using ENV_STATE = typename ENVIRONMENT::State;
using ENV_PARAMS = typename ENVIRONMENT::Parameters;

// ============================================================================
// 1. SAC Configuration
// ============================================================================
struct SAC_PARAMS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
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
using SAC_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, SAC_PARAMS>;
using SAC_STATE = SAC_CONFIG::State<SAC_CONFIG>;

// ============================================================================
// 2. PPO Configuration
// ============================================================================
struct PPO_PARAMS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
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
        static constexpr T GAMMA = 0.99f;
        static constexpr T INITIAL_ACTION_STD = 0.8f;
        static constexpr bool NORMALIZE_OBSERVATIONS = false;
    };
};
using PPO_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
    TYPE_POLICY, TI, RNG, ENVIRONMENT, PPO_PARAMS,
    rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
using PPO_STATE = PPO_CONFIG::State<PPO_CONFIG>;

// ============================================================================
// 3. TD3 Configuration
// ============================================================================
struct TD3_PARAMS : rlt::rl::algorithms::td3::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    struct TD3_PARAMETERS : rlt::rl::algorithms::td3::DefaultParameters<TYPE_POLICY, TI> {
        static constexpr TI ACTOR_BATCH_SIZE = 64;
        static constexpr TI CRITIC_BATCH_SIZE = 64;
    };
    static constexpr TI STEP_LIMIT = 100000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
};
using TD3_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, TD3_PARAMS>;
using TD3_STATE = TD3_CONFIG::State<TD3_CONFIG>;

/// @brief Recorded snapshot of CartPole state and applied action for smooth 50 Hz playback.
struct Frame {
    float x;
    float theta;
    float action;
};

/// @brief Runtime state and telemetry for one of the three algorithms (SAC, PPO, TD3).
struct AlgoStats {
    int env_steps = 0;
    double wall_ms = 0.0;
    double last_return = 0.0;
    std::vector<double> hist_return;
    std::vector<Frame> rollout;
};

DEVICE g_device;
SAC_STATE* g_sac = nullptr;
PPO_STATE* g_ppo = nullptr;
TD3_STATE* g_td3 = nullptr;

AlgoStats g_stat_sac;
AlgoStats g_stat_ppo;
AlgoStats g_stat_td3;

enum class ArenaMode {
    Idle = 0,
    RaceAnim = 1,
    LiveLearning = 2
};
ArenaMode g_mode = ArenaMode::Idle;
std::size_t g_anim_frame = 0;

/**
 * @brief Evaluates a trained RLtools Actor on CartPoleContinuous and records its rollout frames.
 */
template <typename ACTOR_TYPE>
void RolloutActor(ACTOR_TYPE& actor, AlgoStats& stats) {
    stats.rollout.clear();
    ENVIRONMENT env;
    ENV_PARAMS params;
    ENV_STATE state, next_state;
    RNG rng;
    rlt::init(g_device, rng, 123);
    rlt::initial_parameters(g_device, env, params);
    rlt::initial_state(g_device, env, params, state);
    stats.rollout.push_back({state.x, state.theta, 0.0f});

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::Observation::DIM>>> obs_t;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::ACTION_DIM>>> act_t;
    using EVAL_ACTOR = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, 1>;
    typename EVAL_ACTOR::template Buffer<> buf;
    rlt::malloc(g_device, obs_t);
    rlt::malloc(g_device, act_t);
    rlt::malloc(g_device, buf);

    auto obs_m = rlt::matrix_view(g_device, obs_t);
    auto act_m = rlt::matrix_view(g_device, act_t);

    double total_ret = 0.0;
    for (TI t = 0; t < ENVIRONMENT::EPISODE_STEP_LIMIT; ++t) {
        rlt::observe(g_device, env, params, state, typename ENVIRONMENT::Observation{}, obs_m, rng);
        rlt::evaluate(g_device, actor, obs_t, act_t, buf, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        float u = std::clamp(rlt::get(act_m, 0, 0), -1.0f, 1.0f);
        rlt::step(g_device, env, params, state, act_m, next_state, rng);
        total_ret += rlt::reward(g_device, env, params, state, act_m, next_state, rng);
        stats.rollout.push_back({next_state.x, next_state.theta, u});
        state = next_state;
        if (rlt::terminated(g_device, env, params, state, rng)) {
            break;
        }
    }

    rlt::free(g_device, obs_t);
    rlt::free(g_device, act_t);
    rlt::free(g_device, buf);

    stats.last_return = total_ret;
    stats.hist_return.push_back(total_ret);
}

/**
 * @brief Evaluates all three trained policies and updates the plots & playback buffers.
 */
void EvaluateAll() {
    RolloutActor(rlt::get_actor(*g_sac), g_stat_sac);
    RolloutActor(rlt::get_actor(*g_ppo), g_stat_ppo);
    RolloutActor(rlt::get_actor(*g_td3), g_stat_td3);
    g_anim_frame = 0;
}

/**
 * @brief Allocates and initializes fresh SAC, PPO, and TD3 training states.
 */
void InitAllTrainers() {
    if (g_sac) { rlt::free(g_device, *g_sac); delete g_sac; }
    if (g_ppo) { rlt::free(g_device, *g_ppo); delete g_ppo; }
    if (g_td3) { rlt::free(g_device, *g_td3); delete g_td3; }

    g_sac = new SAC_STATE();
    g_ppo = new PPO_STATE();
    g_td3 = new TD3_STATE();

    rlt::malloc(g_device, *g_sac);
    rlt::malloc(g_device, *g_ppo);
    rlt::malloc(g_device, *g_td3);

    rlt::init(g_device, *g_sac, 42);
    rlt::init(g_device, *g_ppo, 42);
    rlt::init(g_device, *g_td3, 42);

    g_stat_sac = AlgoStats{};
    g_stat_ppo = AlgoStats{};
    g_stat_td3 = AlgoStats{};

    EvaluateAll();
}

/**
 * @brief Advances all 3 algorithms by a specified number of steps/iterations.
 */
void TrainStep(int sac_steps, int ppo_iters, int td3_steps) {
    // 1. SAC
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < sac_steps; ++i) {
        if (rlt::step(g_device, *g_sac)) break;
        g_stat_sac.env_steps++;
    }
    auto t1 = std::chrono::steady_clock::now();
    g_stat_sac.wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. PPO
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < ppo_iters; ++i) {
        if (rlt::step(g_device, *g_ppo)) break;
        g_stat_ppo.env_steps += PPO_PARAMS::N_ENVIRONMENTS * PPO_PARAMS::ON_POLICY_RUNNER_STEPS_PER_ENV;
    }
    auto t3 = std::chrono::steady_clock::now();
    g_stat_ppo.wall_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 3. TD3
    auto t4 = std::chrono::steady_clock::now();
    for (int i = 0; i < td3_steps; ++i) {
        if (rlt::step(g_device, *g_td3)) break;
        g_stat_td3.env_steps++;
    }
    auto t5 = std::chrono::steady_clock::now();
    g_stat_td3.wall_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

    EvaluateAll();
}

/**
 * @brief Draws a single CartPole mini-stage inside `rc` for one algorithm.
 */
void DrawCartPoleStage(HDC hdc, const RECT& rc, const AlgoStats& st,
                       const std::wstring& label, COLORREF accent) {
    HBRUSH bg = CreateSolidBrush(RGB(28, 34, 46));
    FillRect(hdc, &rc, bg);
    DeleteObject(bg);

    HPEN bp = CreatePen(PS_SOLID, 1, accent);
    HGDIOBJ op = SelectObject(hdc, bp);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(bp);

    std::size_t idx = st.rollout.empty() ? 0 : std::min(g_anim_frame, st.rollout.size() - 1);
    Frame f = st.rollout.empty() ? Frame{0.0f, 0.0f, 0.0f} : st.rollout[idx];
    bool finished_early = (st.rollout.size() > 1 && idx == st.rollout.size() - 1 && st.last_return < 499.5);

    std::wstringstream hdr;
    hdr << std::fixed << std::setprecision(0)
        << label << L"   |   Steps: " << st.env_steps
        << L" (" << st.wall_ms << L" ms)   |   Balanced: " << st.last_return << L" / 500";
    if (st.last_return >= 500.0) hdr << L"  [SOLVED!]";
    RECT hrc{rc.left + 10, rc.top + 4, rc.right - 10, rc.top + 26};
    SetTextColor(hdc, accent);
    DrawTextW(hdc, hdr.str().c_str(), -1, &hrc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    int stage_h = rc.bottom - rc.top;
    int track_y = rc.top + stage_h * 76 / 100;
    int margin = 36;
    int tl = rc.left + margin;
    int tr = rc.right - margin;
    int tw = std::max(1, tr - tl);

    // Track line
    HPEN tp = CreatePen(PS_SOLID, 2, RGB(105, 120, 145));
    op = SelectObject(hdc, tp);
    MoveToEx(hdc, tl, track_y, nullptr);
    LineTo(hdc, tr, track_y);
    SelectObject(hdc, op);
    DeleteObject(tp);

    double nx = std::clamp((static_cast<double>(f.x) + 2.4) / 4.8, 0.0, 1.0);
    int cx = tl + static_cast<int>(nx * tw);
    int cw = 62, ch = 28;
    int ctop = track_y - ch;

    // Cart
    RECT crc{cx - cw / 2, ctop, cx + cw / 2, track_y};
    HBRUSH cbr = CreateSolidBrush(finished_early ? RGB(190, 55, 55) : accent);
    FillRect(hdc, &crc, cbr);
    DeleteObject(cbr);

    // Pole
    int px = cx, py = ctop + 4;
    double plen = std::max(45.0, stage_h * 0.48);
    int tx = px + static_cast<int>(plen * std::sin(f.theta));
    int ty = py - static_cast<int>(plen * std::cos(f.theta));

    HPEN ppen = CreatePen(PS_SOLID, 7, RGB(250, 204, 21));
    op = SelectObject(hdc, ppen);
    MoveToEx(hdc, px, py, nullptr);
    LineTo(hdc, tx, ty);
    SelectObject(hdc, op);
    DeleteObject(ppen);

    // Continuous push force bar proportional to u in [-1, 1]
    if (std::abs(f.action) > 0.05f && !finished_early) {
        HPEN fpen = CreatePen(PS_SOLID, 3, RGB(74, 222, 128));
        op = SelectObject(hdc, fpen);
        int ay = ctop + ch / 2;
        int len = static_cast<int>(f.action * 45.0f);
        int start_fx = (len > 0) ? (crc.right + 4) : (crc.left - 4);
        MoveToEx(hdc, start_fx, ay, nullptr);
        LineTo(hdc, start_fx + len, ay);
        SelectObject(hdc, op);
        DeleteObject(fpen);
    }
}

/**
 * @brief Draws the 3-curve comparison chart (SAC Cyan, PPO Magenta, TD3 Amber).
 */
void DrawThreeCurveChart(HDC hdc, const RECT& rc) {
    HBRUSH bg = CreateSolidBrush(RGB(30, 36, 48));
    FillRect(hdc, &rc, bg);
    DeleteObject(bg);

    HPEN bp = CreatePen(PS_SOLID, 1, RGB(65, 75, 95));
    HGDIOBJ op = SelectObject(hdc, bp);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(bp);

    RECT trc{rc.left + 12, rc.top + 8, rc.right - 12, rc.top + 32};
    SetTextColor(hdc, RGB(230, 236, 248));
    DrawTextW(hdc, L"Steps Balanced per Training Round (Max = 500)", -1, &trc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    std::wstringstream leg;
    leg << std::fixed << std::setprecision(0)
        << L"SAC (Cyan): " << g_stat_sac.last_return
        << L"  |  PPO (Magenta): " << g_stat_ppo.last_return
        << L"  |  TD3 (Amber): " << g_stat_td3.last_return;
    RECT lrc{rc.left + 12, rc.top + 34, rc.right - 12, rc.top + 56};
    SetTextColor(hdc, RGB(190, 205, 225));
    DrawTextW(hdc, leg.str().c_str(), -1, &lrc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    int il = rc.left + 18;
    int ir = rc.right - 18;
    int it = rc.top + 66;
    int ib = rc.bottom - 20;
    int w = std::max(1, ir - il);
    int h = std::max(1, ib - it);

    // Reference 500 line
    HPEN ref_p = CreatePen(PS_DOT, 1, RGB(70, 120, 90));
    op = SelectObject(hdc, ref_p);
    MoveToEx(hdc, il, it, nullptr);
    LineTo(hdc, ir, it);
    SelectObject(hdc, op);
    DeleteObject(ref_p);

    auto draw_curve = [&](const std::vector<double>& v, COLORREF col) {
        if (v.empty()) return;
        HPEN cp = CreatePen(PS_SOLID, 2, col);
        HGDIOBJ old = SelectObject(hdc, cp);
        for (std::size_t i = 0; i < v.size(); ++i) {
            int px = il + (v.size() > 1 ? static_cast<int>((static_cast<double>(i) / (v.size() - 1)) * w) : w / 2);
            int py = ib - static_cast<int>((std::clamp(v[i], 0.0, 500.0) / 500.0) * h);
            if (i == 0) MoveToEx(hdc, px, py, nullptr);
            else LineTo(hdc, px, py);
        }
        SelectObject(hdc, old);
        DeleteObject(cp);
    };

    draw_curve(g_stat_sac.hist_return, RGB(56, 189, 248));  // Cyan = SAC
    draw_curve(g_stat_ppo.hist_return, RGB(232, 121, 249)); // Magenta = PPO
    draw_curve(g_stat_td3.hist_return, RGB(251, 146, 60));  // Amber = TD3
}

/**
 * @brief Renders the full 3-algorithm CartPole comparison dashboard.
 */
void DrawDashboard(HDC hdc, const RECT& client_rect) {
    int win_w = client_rect.right - client_rect.left;
    int win_h = client_rect.bottom - client_rect.top;

    HBRUSH bg = CreateSolidBrush(RGB(18, 22, 30));
    FillRect(hdc, &client_rect, bg);
    DeleteObject(bg);
    SetBkMode(hdc, TRANSPARENT);

    int pad = 16;
    int hud_h = 84;

    std::wstring mode_str = L"Idle";
    if (g_mode == ArenaMode::LiveLearning) mode_str = L"LIVE LEARNING (Press L to Pause)";
    else if (g_mode == ArenaMode::RaceAnim) mode_str = L"3-WAY CARTPOLE BALANCE RACE (50 Hz)";

    std::wstringstream top;
    top << L"RLtools 3-Way Continuous CartPole Arena: SAC vs. PPO vs. TD3   |   Mode: " << mode_str;
    RECT hud1{pad, 8, win_w - pad, 36};
    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextW(hdc, top.str().c_str(), -1, &hud1, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl =
        L"[L]: Toggle LIVE Learning   |   [T]: Train +1 Big Round   |   [Space]: Watch All 3 Balance Live!   |   [C]: Reset   |   [Esc]: Quit";
    RECT hud2{pad, 40, win_w - pad, 68};
    SetTextColor(hdc, RGB(125, 211, 252));
    DrawTextW(hdc, ctrl.c_str(), -1, &hud2, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    int ctop = hud_h;
    int ch = win_h - ctop - pad;
    int left_w = (win_w - 3 * pad) * 56 / 100;
    int right_w = (win_w - 3 * pad) - left_w;
    int gap = 10;
    int stage_h = (ch - 2 * gap) / 3;

    // Left: 3 Stacked CartPole Stages (SAC, PPO, TD3)
    RECT r_sac{pad, ctop, pad + left_w, ctop + stage_h};
    RECT r_ppo{pad, ctop + stage_h + gap, pad + left_w, ctop + 2 * stage_h + gap};
    RECT r_td3{pad, ctop + 2 * (stage_h + gap), pad + left_w, ctop + ch};

    DrawCartPoleStage(hdc, r_sac, g_stat_sac, L"1. SAC (Off-Policy Stochastic)", RGB(56, 189, 248));
    DrawCartPoleStage(hdc, r_ppo, g_stat_ppo, L"2. PPO (On-Policy Stochastic)", RGB(232, 121, 249));
    DrawCartPoleStage(hdc, r_td3, g_stat_td3, L"3. TD3 (Off-Policy Deterministic)", RGB(251, 146, 60));

    // Right: 3-Curve Learning Plot
    RECT plot_rc{2 * pad + left_w, ctop, 2 * pad + left_w + right_w, ctop + ch};
    DrawThreeCurveChart(hdc, plot_rc);
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
                        SetTimer(hwnd, 1, 80, nullptr);
                    }
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'T':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    TrainStep(500, 4, 500);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_SPACE:
                    g_anim_frame = 0;
                    g_mode = ArenaMode::RaceAnim;
                    SetTimer(hwnd, 1, 20, nullptr); // 50 Hz real-time playback!
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'C':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    InitAllTrainers();
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
                TrainStep(150, 1, 150);
                InvalidateRect(hwnd, nullptr, FALSE);
            } else if (g_mode == ArenaMode::RaceAnim) {
                std::size_t max_f = std::max({g_stat_sac.rollout.size(),
                                              g_stat_ppo.rollout.size(),
                                              g_stat_td3.rollout.size()});
                if (g_anim_frame + 1 < max_f) {
                    g_anim_frame++;
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
            if (g_sac) { rlt::free(g_device, *g_sac); delete g_sac; g_sac = nullptr; }
            if (g_ppo) { rlt::free(g_device, *g_ppo); delete g_ppo; g_ppo = nullptr; }
            if (g_td3) { rlt::free(g_device, *g_td3); delete g_td3; g_td3 = nullptr; }
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

} // namespace

int main() {
    InitAllTrainers();

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    const wchar_t CLASS_NAME[] = L"RLtoolsCartPole3WayClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"RLtools 3-Way CartPole Arena: SAC vs. PPO vs. TD3",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1300, 820,
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
