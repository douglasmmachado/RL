/**
 * @file visualize_gym_suite.cpp
 * @brief Unified C++ Gymnasium-Style Multi-Environment RLtools Arena (SAC vs. PPO vs. TD3)!
 *
 * Switch live between 5 classic continuous control environments with number keys [1]..[5]:
 *   [1] CartPole-Continuous   (Balance pole on track — 4D state, 1D force)
 *   [2] Pendulum-v1 Swing-Up  (RLtools built-in Pendulum — swing up & balance inverted)
 *   [3] Acrobot Swing-Up      (RLtools built-in 2-link chaotic robot arm)
 *   [4] MountainCar-Continuous(Underpowered car pumping momentum to climb hill)
 *   [5] Drone2D Hover         (Planar quadrotor with dual rotors hovering against gravity)
 *
 * Controls:
 *   [1]-[5]: Switch Environment
 *   [L]:     Toggle LIVE Learning (trains SAC, PPO, TD3 in parallel & updates curves live)
 *   [T]:     Fast-Train +1 Big Round
 *   [Space]: Animate all 3 trained agents side-by-side at 50 Hz!
 *   [C]:     Reset all 3 neural networks for the active environment
 *   [Esc]:   Quit
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

// All 5 Environments (2 RLtools built-ins + 3 custom C++ environments)
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/acrobot/operations_generic.h>
#include "rl/rltools_cartpole.hpp"
#include "rl/rltools_mountaincar.hpp"
#include "rl/rltools_drone2d.hpp"

// RLtools SAC, PPO, TD3 Core Loops
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>

#include <windows.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
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

/// @brief Generic visual frame snapshot (up to 6 floats for pose + 2 floats for actions).
struct RenderFrame {
    std::array<float, 6> s{0, 0, 0, 0, 0, 0};
    std::array<float, 2> a{0, 0};
    bool terminated{false};
};

/// @brief Runtime telemetry and recorded playback frames for one algorithm.
struct AlgoStats {
    int env_steps = 0;
    double wall_ms = 0.0;
    double last_return = 0.0;
    std::vector<double> hist_return;
    std::vector<RenderFrame> rollout;
};

/**
 * @brief Abstract polymorphic interface for a 3-way (SAC, PPO, TD3) RLtools environment suite runner.
 */
class IEnvSuite {
public:
    virtual ~IEnvSuite() = default;
    [[nodiscard]] virtual std::wstring name() const = 0;
    [[nodiscard]] virtual std::wstring subtitle() const = 0;
    virtual void reset_trainers() = 0;
    virtual void train_step(int sac_steps, int ppo_iters, int td3_steps) = 0;
    virtual void draw_stage(HDC hdc, const RECT& rc, int algo_idx, std::size_t frame_idx,
                            const std::wstring& label, COLORREF accent) const = 0;
    [[nodiscard]] virtual const AlgoStats& stats(int algo_idx) const = 0;
};

/**
 * @brief Generic templated 3-way RLtools Arena (SAC + PPO + TD3) for ANY RLtools Environment!
 * @tparam ENV The RLtools environment type.
 * @tparam FrameExtractor Functor converting (const ENV::State&, const Matrix<ACTION>&, bool) -> RenderFrame.
 * @tparam StageDrawer Functor rendering one RenderFrame into a Win32 RECT.
 */
template <typename ENV, typename FrameExtractor, typename StageDrawer>
class RLtoolsEnvSuite : public IEnvSuite {
public:
    struct SAC_PARAMS : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENV> {
        struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENV::ACTION_DIM> {
            static constexpr TI ACTOR_BATCH_SIZE = 64;
            static constexpr TI CRITIC_BATCH_SIZE = 64;
        };
        static constexpr TI STEP_LIMIT = 200000;
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
        static constexpr TI STEP_LIMIT = 4000;
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
        static constexpr TI STEP_LIMIT = 200000;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 32;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 32;
    };
    using TD3_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<TYPE_POLICY, TI, RNG, ENV, TD3_PARAMS>;
    using TD3_STATE = typename TD3_CONFIG::template State<TD3_CONFIG>;

    RLtoolsEnvSuite(std::wstring title, std::wstring sub, FrameExtractor extractor, StageDrawer drawer)
        : title_(std::move(title)), subtitle_(std::move(sub)), extractor_(extractor), drawer_(drawer) {}

    ~RLtoolsEnvSuite() override { free_trainers(); }

    [[nodiscard]] std::wstring name() const override { return title_; }
    [[nodiscard]] std::wstring subtitle() const override { return subtitle_; }

    void reset_trainers() override {
        free_trainers();
        sac_ = new SAC_STATE();
        ppo_ = new PPO_STATE();
        td3_ = new TD3_STATE();

        rlt::malloc(device_, *sac_);
        rlt::malloc(device_, *ppo_);
        rlt::malloc(device_, *td3_);

        rlt::init(device_, *sac_, 42);
        rlt::init(device_, *ppo_, 42);
        rlt::init(device_, *td3_, 42);

        stats_ = {AlgoStats{}, AlgoStats{}, AlgoStats{}};
        evaluate_all();
    }

    void train_step(int sac_steps, int ppo_iters, int td3_steps) override {
        if (!sac_) reset_trainers();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < sac_steps; ++i) {
            if (rlt::step(device_, *sac_)) break;
            stats_[0].env_steps++;
        }
        auto t1 = std::chrono::steady_clock::now();
        stats_[0].wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < ppo_iters; ++i) {
            if (rlt::step(device_, *ppo_)) break;
            stats_[1].env_steps += PPO_PARAMS::N_ENVIRONMENTS * PPO_PARAMS::ON_POLICY_RUNNER_STEPS_PER_ENV;
        }
        auto t3 = std::chrono::steady_clock::now();
        stats_[1].wall_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        auto t4 = std::chrono::steady_clock::now();
        for (int i = 0; i < td3_steps; ++i) {
            if (rlt::step(device_, *td3_)) break;
            stats_[2].env_steps++;
        }
        auto t5 = std::chrono::steady_clock::now();
        stats_[2].wall_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        evaluate_all();
    }

    void draw_stage(HDC hdc, const RECT& rc, int algo_idx, std::size_t frame_idx,
                    const std::wstring& label, COLORREF accent) const override {
        const AlgoStats& st = stats_[algo_idx];

        HBRUSH bg = CreateSolidBrush(RGB(26, 32, 44));
        FillRect(hdc, &rc, bg);
        DeleteObject(bg);

        HPEN bp = CreatePen(PS_SOLID, 1, accent);
        HGDIOBJ op = SelectObject(hdc, bp);
        HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, ob);
        SelectObject(hdc, op);
        DeleteObject(bp);

        std::wstringstream hdr;
        hdr << std::fixed << std::setprecision(1)
            << label << L"   |   Steps: " << st.env_steps
            << L" (" << std::setprecision(0) << st.wall_ms << L" ms)   |   Eval Return: "
            << std::setprecision(1) << st.last_return;
        RECT hrc{rc.left + 10, rc.top + 4, rc.right - 10, rc.top + 26};
        SetTextColor(hdc, accent);
        DrawTextW(hdc, hdr.str().c_str(), -1, &hrc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

        RECT inner{rc.left + 8, rc.top + 28, rc.right - 8, rc.bottom - 8};
        RenderFrame f{};
        if (!st.rollout.empty()) {
            f = st.rollout[std::min(frame_idx, st.rollout.size() - 1)];
        }
        drawer_(hdc, inner, f, accent);
    }

    [[nodiscard]] const AlgoStats& stats(int algo_idx) const override {
        return stats_[algo_idx];
    }

private:
    std::wstring title_;
    std::wstring subtitle_;
    FrameExtractor extractor_;
    StageDrawer drawer_;

    DEVICE device_;
    SAC_STATE* sac_ = nullptr;
    PPO_STATE* ppo_ = nullptr;
    TD3_STATE* td3_ = nullptr;
    std::array<AlgoStats, 3> stats_;

    void free_trainers() {
        if (sac_) { rlt::free(device_, *sac_); delete sac_; sac_ = nullptr; }
        if (ppo_) { rlt::free(device_, *ppo_); delete ppo_; ppo_ = nullptr; }
        if (td3_) { rlt::free(device_, *td3_); delete td3_; td3_ = nullptr; }
    }

    template <typename ACTOR_TYPE>
    void rollout_actor(ACTOR_TYPE& actor, AlgoStats& out) {
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
        out.rollout.push_back(extractor_(state, act_m, false));

        double total_ret = 0.0;
        for (TI t = 0; t < ENV::EPISODE_STEP_LIMIT; ++t) {
            rlt::observe(device_, env, params, state, typename ENV::Observation{}, obs_m, rng);
            rlt::evaluate(device_, actor, obs_t, act_t, buf, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
            rlt::step(device_, env, params, state, act_m, next_state, rng);
            total_ret += rlt::reward(device_, env, params, state, act_m, next_state, rng);
            bool term = rlt::terminated(device_, env, params, next_state, rng);
            out.rollout.push_back(extractor_(next_state, act_m, term));
            state = next_state;
            if (term) break;
        }

        rlt::free(device_, obs_t);
        rlt::free(device_, act_t);
        rlt::free(device_, buf);

        out.last_return = total_ret;
        out.hist_return.push_back(total_ret);
    }

    void evaluate_all() {
        rollout_actor(rlt::get_actor(*sac_), stats_[0]);
        rollout_actor(rlt::get_actor(*ppo_), stats_[1]);
        rollout_actor(rlt::get_actor(*td3_), stats_[2]);
    }
};

// ============================================================================
// Factory builders for all 5 Environments (with custom 2D GDI renderers)
// ============================================================================

std::unique_ptr<IEnvSuite> MakeCartPoleSuite() {
    using ENV = rlt::rl::environments::CartPoleContinuous<
        rlt::rl::environments::cartpole_continuous::Specification<T, TI>>;
    auto extract = [](const typename ENV::State& st, const auto& act, bool term) {
        RenderFrame f;
        f.s[0] = st.x;
        f.s[1] = st.theta;
        f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
        f.terminated = term;
        return f;
    };
    auto draw = [](HDC hdc, const RECT& rc, const RenderFrame& f, COLORREF accent) {
        int h = rc.bottom - rc.top;
        int ty = rc.top + h * 76 / 100;
        int tl = rc.left + 28, tr = rc.right - 28;
        HPEN tp = CreatePen(PS_SOLID, 2, RGB(105, 120, 145));
        HGDIOBJ op = SelectObject(hdc, tp);
        MoveToEx(hdc, tl, ty, nullptr); LineTo(hdc, tr, ty);
        SelectObject(hdc, op); DeleteObject(tp);

        double nx = std::clamp((f.s[0] + 2.4) / 4.8, 0.0, 1.0);
        int cx = tl + static_cast<int>(nx * (tr - tl));
        int cw = 54, ch = 24;
        RECT crc{cx - cw / 2, ty - ch, cx + cw / 2, ty};
        HBRUSH cbr = CreateSolidBrush(f.terminated ? RGB(195, 55, 55) : accent);
        FillRect(hdc, &crc, cbr); DeleteObject(cbr);

        double plen = std::max(40.0, h * 0.52);
        int px = cx, py = ty - ch + 4;
        int tx = px + static_cast<int>(plen * std::sin(f.s[1]));
        int tip_y = py - static_cast<int>(plen * std::cos(f.s[1]));
        HPEN pp = CreatePen(PS_SOLID, 6, RGB(250, 204, 21));
        op = SelectObject(hdc, pp);
        MoveToEx(hdc, px, py, nullptr); LineTo(hdc, tx, tip_y);
        SelectObject(hdc, op); DeleteObject(pp);
    };
    return std::make_unique<RLtoolsEnvSuite<ENV, decltype(extract), decltype(draw)>>(
        L"[1] CartPole-Continuous", L"4D State | 1D Continuous Force | Goal: Balance 500 steps", extract, draw);
}

std::unique_ptr<IEnvSuite> MakePendulumSuite() {
    using ENV = rlt::rl::environments::Pendulum<
        rlt::rl::environments::pendulum::Specification<T, TI>>;
    auto extract = [](const typename ENV::State& st, const auto& act, bool term) {
        RenderFrame f;
        f.s[0] = st.theta;
        f.s[1] = st.theta_dot;
        f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
        f.terminated = term;
        return f;
    };
    auto draw = [](HDC hdc, const RECT& rc, const RenderFrame& f, COLORREF accent) {
        int cx = (rc.left + rc.right) / 2;
        int cy = (rc.top + rc.bottom) / 2;
        double len = std::min(rc.right - rc.left, rc.bottom - rc.top) * 0.38;

        // Upright target indicator
        HPEN gp = CreatePen(PS_DOT, 1, RGB(60, 140, 90));
        HGDIOBJ op = SelectObject(hdc, gp);
        MoveToEx(hdc, cx, cy, nullptr); LineTo(hdc, cx, cy - static_cast<int>(len * 1.15));
        SelectObject(hdc, op); DeleteObject(gp);

        // In RLtools Pendulum, theta = 0 is upright!
        int bx = cx + static_cast<int>(len * std::sin(f.s[0]));
        int by = cy - static_cast<int>(len * std::cos(f.s[0]));

        HPEN rp = CreatePen(PS_SOLID, 6, RGB(250, 204, 21));
        op = SelectObject(hdc, rp);
        MoveToEx(hdc, cx, cy, nullptr); LineTo(hdc, bx, by);
        SelectObject(hdc, op); DeleteObject(rp);

        HBRUSH bb = CreateSolidBrush(accent);
        HGDIOBJ ob = SelectObject(hdc, bb);
        Ellipse(hdc, bx - 10, by - 10, bx + 10, by + 10);
        Ellipse(hdc, cx - 5, cy - 5, cx + 5, cy + 5);
        SelectObject(hdc, ob); DeleteObject(bb);
    };
    return std::make_unique<RLtoolsEnvSuite<ENV, decltype(extract), decltype(draw)>>(
        L"[2] Pendulum-v1 Swing-Up (RLtools Built-in)", L"3D Fourier Obs | 1D Torque | Goal: Swing upright (Return -> 0)", extract, draw);
}

std::unique_ptr<IEnvSuite> MakeAcrobotSuite() {
    using ENV = rlt::rl::environments::AcrobotSwingup<
        rlt::rl::environments::acrobot::Specification<T, TI, rlt::rl::environments::acrobot::EasyParameters<T>>>;
    auto extract = [](const typename ENV::State& st, const auto& act, bool term) {
        RenderFrame f;
        f.s[0] = st.theta_1;
        f.s[1] = st.theta_2;
        f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
        f.terminated = term;
        return f;
    };
    auto draw = [](HDC hdc, const RECT& rc, const RenderFrame& f, COLORREF accent) {
        int cx = (rc.left + rc.right) / 2;
        int cy = (rc.top + rc.bottom) / 2;
        double l = std::min(rc.right - rc.left, rc.bottom - rc.top) * 0.21;

        // Target height line
        int goal_y = cy - static_cast<int>(l * 1.2);
        HPEN gp = CreatePen(PS_DOT, 1, RGB(70, 170, 100));
        HGDIOBJ op = SelectObject(hdc, gp);
        MoveToEx(hdc, rc.left + 20, goal_y, nullptr); LineTo(hdc, rc.right - 20, goal_y);
        SelectObject(hdc, op); DeleteObject(gp);

        // Link 1 (theta_1 = 0 hangs straight down in Acrobot)
        int x1 = cx + static_cast<int>(l * std::sin(f.s[0]));
        int y1 = cy + static_cast<int>(l * std::cos(f.s[0]));
        // Link 2
        int x2 = x1 + static_cast<int>(l * std::sin(f.s[0] + f.s[1]));
        int y2 = y1 + static_cast<int>(l * std::cos(f.s[0] + f.s[1]));

        HPEN p1 = CreatePen(PS_SOLID, 6, accent);
        op = SelectObject(hdc, p1);
        MoveToEx(hdc, cx, cy, nullptr); LineTo(hdc, x1, y1);
        SelectObject(hdc, op); DeleteObject(p1);

        HPEN p2 = CreatePen(PS_SOLID, 5, RGB(250, 204, 21));
        op = SelectObject(hdc, p2);
        MoveToEx(hdc, x1, y1, nullptr); LineTo(hdc, x2, y2);
        SelectObject(hdc, op); DeleteObject(p2);

        HBRUSH jb = CreateSolidBrush(RGB(255, 255, 255));
        HGDIOBJ ob = SelectObject(hdc, jb);
        Ellipse(hdc, cx - 5, cy - 5, cx + 5, cy + 5);
        Ellipse(hdc, x1 - 5, y1 - 5, x1 + 5, y1 + 5);
        Ellipse(hdc, x2 - 6, y2 - 6, x2 + 6, y2 + 6);
        SelectObject(hdc, ob); DeleteObject(jb);
    };
    return std::make_unique<RLtoolsEnvSuite<ENV, decltype(extract), decltype(draw)>>(
        L"[3] Acrobot 2-Link Swing-Up (RLtools Built-in)", L"6D Obs | 1D Elbow Torque | Chaotic Double Pendulum", extract, draw);
}

std::unique_ptr<IEnvSuite> MakeMountainCarSuite() {
    using ENV = rlt::rl::environments::MountainCar<
        rlt::rl::environments::mountain_car::Specification<T, TI>>;
    auto extract = [](const typename ENV::State& st, const auto& act, bool term) {
        RenderFrame f;
        f.s[0] = st.position;
        f.s[1] = st.velocity;
        f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
        f.terminated = term;
        return f;
    };
    auto draw = [](HDC hdc, const RECT& rc, const RenderFrame& f, COLORREF accent) {
        int w = rc.right - rc.left - 32;
        int h = rc.bottom - rc.top - 24;
        auto to_pt = [&](double pos) {
            double nx = (pos + 1.2) / 1.8;
            double ny = (std::sin(3.0 * pos) + 1.0) * 0.5;
            return POINT{rc.left + 16 + static_cast<int>(nx * w),
                         rc.bottom - 12 - static_cast<int>(ny * h)};
        };

        // Hill curve
        HPEN hp = CreatePen(PS_SOLID, 2, RGB(115, 130, 155));
        HGDIOBJ op = SelectObject(hdc, hp);
        for (int i = 0; i <= 60; ++i) {
            double p = -1.2 + (1.8 * i) / 60.0;
            POINT pt = to_pt(p);
            if (i == 0) MoveToEx(hdc, pt.x, pt.y, nullptr);
            else LineTo(hdc, pt.x, pt.y);
        }
        SelectObject(hdc, op); DeleteObject(hp);

        // Goal flag at x = 0.45
        POINT gp = to_pt(0.45);
        HPEN fp = CreatePen(PS_SOLID, 3, RGB(74, 222, 128));
        op = SelectObject(hdc, fp);
        MoveToEx(hdc, gp.x, gp.y, nullptr); LineTo(hdc, gp.x, gp.y - 26);
        LineTo(hdc, gp.x + 14, gp.y - 20); LineTo(hdc, gp.x, gp.y - 14);
        SelectObject(hdc, op); DeleteObject(fp);

        // Car dot
        POINT cp = to_pt(f.s[0]);
        HBRUSH cb = CreateSolidBrush(accent);
        HGDIOBJ ob = SelectObject(hdc, cb);
        Ellipse(hdc, cp.x - 9, cp.y - 9, cp.x + 9, cp.y + 9);
        SelectObject(hdc, ob); DeleteObject(cb);
    };
    return std::make_unique<RLtoolsEnvSuite<ENV, decltype(extract), decltype(draw)>>(
        L"[4] MountainCar-Continuous", L"2D State | 1D Engine Force | Pump momentum to reach flag (x >= 0.45)", extract, draw);
}

std::unique_ptr<IEnvSuite> MakeDrone2DSuite() {
    using ENV = rlt::rl::environments::Drone2D<
        rlt::rl::environments::drone_2d::Specification<T, TI>>;
    auto extract = [](const typename ENV::State& st, const auto& act, bool term) {
        RenderFrame f;
        f.s[0] = st.x;
        f.s[1] = st.y;
        f.s[2] = st.theta;
        f.a[0] = std::clamp(rlt::get(act, 0, 0), -1.0f, 1.0f);
        f.a[1] = std::clamp(rlt::get(act, 0, 1), -1.0f, 1.0f);
        f.terminated = term;
        return f;
    };
    auto draw = [](HDC hdc, const RECT& rc, const RenderFrame& f, COLORREF accent) {
        int w = rc.right - rc.left;
        int h = rc.bottom - rc.top;
        auto to_px = [&](double wx, double wy) {
            return POINT{rc.left + static_cast<int>(((wx + 1.5) / 3.0) * w),
                         rc.bottom - static_cast<int>(((wy + 1.5) / 3.0) * h)};
        };

        // Target hover crosshair at (0, 0)
        POINT tgt = to_px(0.0, 0.0);
        HPEN tp = CreatePen(PS_DOT, 1, RGB(74, 222, 128));
        HGDIOBJ op = SelectObject(hdc, tp);
        MoveToEx(hdc, tgt.x - 14, tgt.y, nullptr); LineTo(hdc, tgt.x + 14, tgt.y);
        MoveToEx(hdc, tgt.x, tgt.y - 14, nullptr); LineTo(hdc, tgt.x, tgt.y + 14);
        SelectObject(hdc, op); DeleteObject(tp);

        // Drone body bar tilted by theta
        POINT c = to_px(f.s[0], f.s[1]);
        double half_arm = 24.0;
        double ct = std::cos(f.s[2]), st = std::sin(f.s[2]);
        int lx = c.x - static_cast<int>(half_arm * ct);
        int ly = c.y - static_cast<int>(half_arm * st);
        int rx = c.x + static_cast<int>(half_arm * ct);
        int ry = c.y + static_cast<int>(half_arm * st);

        // Thrust flames under each rotor
        HPEN flp = CreatePen(PS_SOLID, 4, RGB(250, 204, 21));
        op = SelectObject(hdc, flp);
        double fl_l = (f.a[0] + 1.0) * 12.0;
        double fl_r = (f.a[1] + 1.0) * 12.0;
        MoveToEx(hdc, lx, ly, nullptr); LineTo(hdc, lx - static_cast<int>(fl_l * st), ly + static_cast<int>(fl_l * ct));
        MoveToEx(hdc, rx, ry, nullptr); LineTo(hdc, rx - static_cast<int>(fl_r * st), ry + static_cast<int>(fl_r * ct));
        SelectObject(hdc, op); DeleteObject(flp);

        // Chassis beam
        HPEN bp = CreatePen(PS_SOLID, 6, f.terminated ? RGB(200, 55, 55) : accent);
        op = SelectObject(hdc, bp);
        MoveToEx(hdc, lx, ly, nullptr); LineTo(hdc, rx, ry);
        SelectObject(hdc, op); DeleteObject(bp);
    };
    return std::make_unique<RLtoolsEnvSuite<ENV, decltype(extract), decltype(draw)>>(
        L"[5] Drone2D Planar Quadrotor Hover", L"6D State | 2D Dual-Rotor Thrust | Goal: Hover at green crosshair (0, 0)", extract, draw);
}

// ============================================================================
// Global Suite Manager & Win32 GUI
// ============================================================================
std::vector<std::unique_ptr<IEnvSuite>> g_suites;
int g_active_env = 0;

enum class ArenaMode { Idle = 0, RaceAnim = 1, LiveLearning = 2 };
ArenaMode g_mode = ArenaMode::Idle;
std::size_t g_anim_frame = 0;

void SelectEnvironment(int idx) {
    if (idx < 0 || idx >= static_cast<int>(g_suites.size())) return;
    g_active_env = idx;
    g_anim_frame = 0;
    if (g_suites[g_active_env]->stats(0).hist_return.empty()) {
        g_suites[g_active_env]->reset_trainers();
    }
}

void DrawComparisonPlot(HDC hdc, const RECT& rc, const IEnvSuite& suite) {
    HBRUSH bg = CreateSolidBrush(RGB(28, 34, 46));
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
    DrawTextW(hdc, L"Evaluation Return per Training Round (SAC vs. PPO vs. TD3)", -1, &trc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    const auto& s0 = suite.stats(0);
    const auto& s1 = suite.stats(1);
    const auto& s2 = suite.stats(2);

    std::wstringstream leg;
    leg << std::fixed << std::setprecision(1)
        << L"SAC (Cyan): " << s0.last_return
        << L"  |  PPO (Magenta): " << s1.last_return
        << L"  |  TD3 (Amber): " << s2.last_return;
    RECT lrc{rc.left + 12, rc.top + 34, rc.right - 12, rc.top + 56};
    SetTextColor(hdc, RGB(190, 205, 225));
    DrawTextW(hdc, leg.str().c_str(), -1, &lrc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    double min_v = 1e9, max_v = -1e9;
    for (int k = 0; k < 3; ++k) {
        for (double v : suite.stats(k).hist_return) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (min_v > max_v) return;
    if (std::abs(max_v - min_v) < 1e-4) { max_v += 1.0; min_v -= 1.0; }

    int il = rc.left + 18, ir = rc.right - 18;
    int it = rc.top + 66, ib = rc.bottom - 20;
    int w = std::max(1, ir - il), h = std::max(1, ib - it);

    if (min_v < 0.0 && max_v > 0.0) {
        int zy = ib - static_cast<int>(((0.0 - min_v) / (max_v - min_v)) * h);
        HPEN zp = CreatePen(PS_DOT, 1, RGB(90, 100, 120));
        op = SelectObject(hdc, zp);
        MoveToEx(hdc, il, zy, nullptr); LineTo(hdc, ir, zy);
        SelectObject(hdc, op); DeleteObject(zp);
    }

    const COLORREF cols[3] = {RGB(56, 189, 248), RGB(232, 121, 249), RGB(251, 146, 60)};
    for (int k = 0; k < 3; ++k) {
        const auto& v = suite.stats(k).hist_return;
        if (v.empty()) continue;
        HPEN cp = CreatePen(PS_SOLID, 2, cols[k]);
        HGDIOBJ old = SelectObject(hdc, cp);
        for (std::size_t i = 0; i < v.size(); ++i) {
            int px = il + (v.size() > 1 ? static_cast<int>((static_cast<double>(i) / (v.size() - 1)) * w) : w / 2);
            int py = ib - static_cast<int>(((v[i] - min_v) / (max_v - min_v)) * h);
            if (i == 0) MoveToEx(hdc, px, py, nullptr);
            else LineTo(hdc, px, py);
        }
        SelectObject(hdc, old);
        DeleteObject(cp);
    }
}

void DrawDashboard(HDC hdc, const RECT& client_rect) {
    int win_w = client_rect.right - client_rect.left;
    int win_h = client_rect.bottom - client_rect.top;

    HBRUSH bg = CreateSolidBrush(RGB(18, 22, 30));
    FillRect(hdc, &client_rect, bg);
    DeleteObject(bg);
    SetBkMode(hdc, TRANSPARENT);

    const IEnvSuite& suite = *g_suites[g_active_env];
    int pad = 16, hud_h = 104;

    std::wstring mode_str = L"Idle";
    if (g_mode == ArenaMode::LiveLearning) mode_str = L"LIVE LEARNING (Press L to Pause)";
    else if (g_mode == ArenaMode::RaceAnim) mode_str = L"3-WAY LIVE PLAYBACK (50 Hz)";

    std::wstringstream h1;
    h1 << L"C++ RLtools Gym Suite   |   Active: " << suite.name() << L"   |   Mode: " << mode_str;
    RECT r1{pad, 6, win_w - pad, 32};
    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextW(hdc, h1.str().c_str(), -1, &r1, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring env_bar = L"Switch Env:   [1] CartPole   |   [2] Pendulum   |   [3] Acrobot   |   [4] MountainCar   |   [5] Drone2D";
    RECT r2{pad, 34, win_w - pad, 60};
    SetTextColor(hdc, RGB(250, 204, 21));
    DrawTextW(hdc, env_bar.c_str(), -1, &r2, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl = L"[L]: Live Learning   |   [T]: +1 Big Round   |   [Space]: Watch All 3 Play!   |   [C]: Reset Networks   |   [Esc]: Quit";
    RECT r3{pad, 62, win_w - pad, 88};
    SetTextColor(hdc, RGB(125, 211, 252));
    DrawTextW(hdc, ctrl.c_str(), -1, &r3, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    int ctop = hud_h, ch = win_h - ctop - pad;
    int left_w = (win_w - 3 * pad) * 56 / 100;
    int right_w = (win_w - 3 * pad) - left_w;
    int gap = 10, stage_h = (ch - 2 * gap) / 3;

    RECT s0{pad, ctop, pad + left_w, ctop + stage_h};
    RECT s1{pad, ctop + stage_h + gap, pad + left_w, ctop + 2 * stage_h + gap};
    RECT s2{pad, ctop + 2 * (stage_h + gap), pad + left_w, ctop + ch};

    suite.draw_stage(hdc, s0, 0, g_anim_frame, L"1. SAC (Off-Policy Stochastic)", RGB(56, 189, 248));
    suite.draw_stage(hdc, s1, 1, g_anim_frame, L"2. PPO (On-Policy Stochastic)", RGB(232, 121, 249));
    suite.draw_stage(hdc, s2, 2, g_anim_frame, L"3. TD3 (Off-Policy Deterministic)", RGB(251, 146, 60));

    RECT plot_rc{2 * pad + left_w, ctop, 2 * pad + left_w + right_w, ctop + ch};
    DrawComparisonPlot(hdc, plot_rc, suite);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN: {
            if (wParam >= '1' && wParam <= '5') {
                g_mode = ArenaMode::Idle;
                KillTimer(hwnd, 1);
                SelectEnvironment(static_cast<int>(wParam - '1'));
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            // Support Numpad 1-5 and French AZERTY top-row keys too!
            if (wParam >= VK_NUMPAD1 && wParam <= VK_NUMPAD5) {
                g_mode = ArenaMode::Idle;
                KillTimer(hwnd, 1);
                SelectEnvironment(static_cast<int>(wParam - VK_NUMPAD1));
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
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
                    g_suites[g_active_env]->train_step(500, 4, 500);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_SPACE:
                    g_anim_frame = 0;
                    g_mode = ArenaMode::RaceAnim;
                    SetTimer(hwnd, 1, 20, nullptr);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'C':
                    g_mode = ArenaMode::Idle;
                    KillTimer(hwnd, 1);
                    g_suites[g_active_env]->reset_trainers();
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
                g_suites[g_active_env]->train_step(150, 1, 150);
                InvalidateRect(hwnd, nullptr, FALSE);
            } else if (g_mode == ArenaMode::RaceAnim) {
                const auto& suite = *g_suites[g_active_env];
                std::size_t max_f = std::max({suite.stats(0).rollout.size(),
                                              suite.stats(1).rollout.size(),
                                              suite.stats(2).rollout.size()});
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
            g_suites.clear();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

} // namespace

int main() {
    g_suites.push_back(MakeCartPoleSuite());
    g_suites.push_back(MakePendulumSuite());
    g_suites.push_back(MakeAcrobotSuite());
    g_suites.push_back(MakeMountainCarSuite());
    g_suites.push_back(MakeDrone2DSuite());
    SelectEnvironment(0);

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    const wchar_t CLASS_NAME[] = L"RLtoolsGymSuiteArenaClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"C++ RLtools Gym Suite: 5 Environments x 3 Algorithms (SAC / PPO / TD3)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1320, 840,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hwnd) return 1;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return 0;
}
