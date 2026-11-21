/**
 * @file visualize_cartpole.cpp
 * @brief Native Windows GUI Dashboard for CartPole balancing with Discretized Tabular Q-Learning!
 *
 * Visualizes:
 * - Left Panel:  Animated CartPole physics (track, cart, tilting pole, push force arrow,
 *                angle gauge, and live state readout: x, x_dot, theta, theta_dot).
 * - Right Panel: Live Training Curves:
 *                1. Smoothed Episode Return / Steps Balanced (Moving Average toward 500!)
 *                2. Exploration Rate (epsilon) & Raw Episode Return
 *
 * Controls:
 * - [L]: Toggle LIVE Training (watch the pole fall over at first, then gradually learn to balance!)
 * - [T]: Fast-Train +500 Episodes instantly and update learning curves
 * - [Space]: Watch the trained Greedy Policy balance the pole in real time (60 FPS)!
 * - Left / Right Arrow Keys (or Q / D): Play CartPole manually yourself!
 * - [C]: Clear / Reset Q-table & plots
 * - [R]: Reset current episode
 * - [Esc]: Quit
 */

#ifndef UNICODE
#define UNICODE
#endif

#include "rl/cartpole.hpp"
#include "rl/cartpole_discretizer.hpp"
#include "rl/q_learning_agent.hpp"
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// @brief Global CartPole environment and Discretized Q-Learning agent (5184 states, 2 actions).
rl::CartPole g_env(42);
rl::QLearningAgent g_agent(rl::CartPoleDiscretizer::num_states(), 2, 0.15, 0.99, 1.0, 42);

rl::CartAction g_last_action = rl::CartAction::Right;
bool g_has_pushed = false;
double g_ep_return = 0.0;
bool g_terminated = false;
bool g_truncated = false;
int g_episodes_trained = 0;

std::vector<double> g_hist_smooth_return;
std::vector<double> g_hist_raw_return;

enum class PlayMode {
    Manual = 0,
    GreedyWatch = 1,
    LiveTraining = 2
};
PlayMode g_mode = PlayMode::Manual;

/**
 * @brief Helper to map GridAction (0 or 1) returned by QLearningAgent to CartAction (Left or Right).
 */
rl::CartAction ToCartAction(rl::GridAction a) noexcept {
    return (static_cast<int>(a) == 1) ? rl::CartAction::Right : rl::CartAction::Left;
}

/**
 * @brief Helper to map CartAction (Left or Right) to GridAction (0 or 1) for QLearningAgent::update().
 */
rl::GridAction ToGridAction(rl::CartAction a) noexcept {
    return static_cast<rl::GridAction>(static_cast<int>(a));
}

/**
 * @brief Resets the current CartPole episode.
 */
void ResetEpisode() {
    g_env.reset();
    g_ep_return = 0.0;
    g_terminated = false;
    g_truncated = false;
    g_has_pushed = false;
}

/**
 * @brief Records a completed training episode's return into the plot histories and decays epsilon.
 */
void RecordEpisodeAndDecay() {
    g_episodes_trained++;
    g_hist_raw_return.push_back(g_ep_return);
    double prev = g_hist_smooth_return.empty() ? g_ep_return : g_hist_smooth_return.back();
    double smooth = g_hist_smooth_return.empty() ? g_ep_return : (0.95 * prev + 0.05 * g_ep_return);
    g_hist_smooth_return.push_back(smooth);
    g_agent.decay_epsilon(0.998, 0.01);
}

/**
 * @brief Clears Q-table and training plots back to untrained state.
 */
void ResetAgentAndPlots() {
    g_agent = rl::QLearningAgent(rl::CartPoleDiscretizer::num_states(), 2, 0.15, 0.99, 1.0, 42);
    g_episodes_trained = 0;
    g_hist_smooth_return.clear();
    g_hist_raw_return.clear();
    ResetEpisode();
}

/**
 * @brief Fast-trains the QLearningAgent on CartPole for `num_episodes` episodes.
 */
void TrainBatch(int num_episodes) {
    for (int ep = 0; ep < num_episodes; ++ep) {
        rl::CartPoleState s_cont = g_env.reset();
        int s = rl::CartPoleDiscretizer::discretize(s_cont);
        g_ep_return = 0.0;

        while (true) {
            rl::GridAction ga = g_agent.select_action(s);
            rl::CartAction ca = ToCartAction(ga);
            rl::StepResult<rl::CartPoleState> res = g_env.step(ca);
            int s_next = rl::CartPoleDiscretizer::discretize(res.next_state);

            g_agent.update(s, ga, res.reward, s_next, res.terminated);
            g_ep_return += res.reward;
            s = s_next;
            if (res.done()) {
                break;
            }
        }
        RecordEpisodeAndDecay();
    }
    ResetEpisode();
}

/**
 * @brief Runs a single live training step (updates physics + Q-table for live visualization).
 */
void StepLiveTraining() {
    if (g_terminated || g_truncated) {
        RecordEpisodeAndDecay();
        ResetEpisode();
        return;
    }
    int s = rl::CartPoleDiscretizer::discretize(g_env.state());
    rl::GridAction ga = g_agent.select_action(s);
    rl::CartAction ca = ToCartAction(ga);
    g_last_action = ca;
    g_has_pushed = true;

    rl::StepResult<rl::CartPoleState> res = g_env.step(ca);
    int s_next = rl::CartPoleDiscretizer::discretize(res.next_state);
    g_agent.update(s, ga, res.reward, s_next, res.terminated);

    g_ep_return += res.reward;
    g_terminated = res.terminated;
    g_truncated = res.truncated;
}

/**
 * @brief Runs a single greedy evaluation or manual step.
 */
void ApplyManualOrGreedy(rl::CartAction action) {
    if (g_terminated || g_truncated) return;
    g_last_action = action;
    g_has_pushed = true;
    rl::StepResult<rl::CartPoleState> res = g_env.step(action);
    g_ep_return += res.reward;
    g_terminated = res.terminated;
    g_truncated = res.truncated;
}

/**
 * @brief Draws a 2D line chart inside `plot_rect`.
 */
void DrawLineChart(HDC hdc, const RECT& plot_rect, const std::vector<double>& data,
                   const std::wstring& title, COLORREF line_color, double fixed_max_y = 500.0) {
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

    if (data.empty()) {
        RECT erc = plot_rect;
        SetTextColor(hdc, RGB(120, 135, 160));
        DrawTextW(hdc, L"Press [T] (+500 Episodes) or [L] (Live Train) to watch CartPole learn!",
                  -1, &erc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    std::size_t start_i = data.size() > 600 ? (data.size() - 600) : 0;
    double min_v = 0.0;
    double max_v = fixed_max_y;
    for (std::size_t i = start_i; i < data.size(); ++i) {
        max_v = std::max(max_v, data[i]);
    }

    std::wstringstream leg;
    leg << std::fixed << std::setprecision(1) << L"Latest: " << data.back() << L" / 500.0";
    SetTextColor(hdc, RGB(200, 215, 235));
    DrawTextW(hdc, leg.str().c_str(), -1, &trc, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    int il = plot_rect.left + 14;
    int ir = plot_rect.right - 14;
    int it = plot_rect.top + 32;
    int ib = plot_rect.bottom - 14;
    int w = std::max(1, ir - il);
    int h = std::max(1, ib - it);

    HPEN cp = CreatePen(PS_SOLID, 2, line_color);
    op = SelectObject(hdc, cp);
    std::size_t count = data.size() - start_i;
    for (std::size_t i = 0; i < count; ++i) {
        int px = il + (count > 1 ? static_cast<int>((static_cast<double>(i) / (count - 1)) * w) : w / 2);
        int py = ib - static_cast<int>(((data[start_i + i] - min_v) / (max_v - min_v)) * h);
        if (i == 0) MoveToEx(hdc, px, py, nullptr);
        else LineTo(hdc, px, py);
    }
    SelectObject(hdc, op);
    DeleteObject(cp);
}

/**
 * @brief Renders the animated CartPole physics scene (left) and learning curves (right).
 */
void DrawDashboard(HDC hdc, const RECT& client_rect) {
    int win_w = client_rect.right - client_rect.left;
    int win_h = client_rect.bottom - client_rect.top;

    HBRUSH bg = CreateSolidBrush(RGB(20, 24, 32));
    FillRect(hdc, &client_rect, bg);
    DeleteObject(bg);
    SetBkMode(hdc, TRANSPARENT);

    int pad = 18;
    int hud_h = 104;

    // 1. Top HUD
    std::wstring mode_txt = L"Manual (Use Left/Right Arrows)";
    if (g_mode == PlayMode::LiveTraining) mode_txt = L"LIVE TRAINING (Watch Agent Learn!)";
    else if (g_mode == PlayMode::GreedyWatch) mode_txt = L"GREEDY POLICY DEMO (60 FPS)";

    std::wstringstream top_ss;
    top_ss << std::fixed << std::setprecision(2)
           << L"CartPole Discretized Q-Learning   |   Mode: " << mode_txt
           << L"   |   Episodes: " << g_episodes_trained
           << L"   |   eps: " << g_agent.epsilon()
           << std::setprecision(0)
           << L"   |   Steps Balanced: " << g_ep_return << L" / 500";
    RECT hud1{pad, 10, win_w - pad, 38};
    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextW(hdc, top_ss.str().c_str(), -1, &hud1, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl1 =
        L"[T]: Fast-Train +500 Episodes   |   [L]: Live Training   |   [Space]: Watch Trained Agent Balance!   |   [C]: Reset Q-Table";
    RECT hud2{pad, 42, win_w - pad, 68};
    SetTextColor(hdc, RGB(125, 211, 252));
    DrawTextW(hdc, ctrl1.c_str(), -1, &hud2, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl2 =
        L"Manual Play: Left/Right Arrows or Q/D   |   [R]: Reset Episode   |   [Esc]: Quit";
    RECT hud3{pad, 70, win_w - pad, 94};
    SetTextColor(hdc, RGB(170, 185, 205));
    DrawTextW(hdc, ctrl2.c_str(), -1, &hud3, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    int content_top = hud_h;
    int content_h = win_h - content_top - pad;
    int left_w = (win_w - 3 * pad) * 56 / 100;
    int right_w = (win_w - 3 * pad) - left_w;

    // 2. Left Panel: CartPole Physics Stage
    RECT stage_rc{pad, content_top, pad + left_w, content_top + content_h};
    HBRUSH stage_bg = CreateSolidBrush(RGB(28, 34, 46));
    FillRect(hdc, &stage_rc, stage_bg);
    DeleteObject(stage_bg);

    HPEN s_border = CreatePen(PS_SOLID, 1, RGB(65, 75, 95));
    HGDIOBJ op = SelectObject(hdc, s_border);
    HGDIOBJ ob = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, stage_rc.left, stage_rc.top, stage_rc.right, stage_rc.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(s_border);

    const auto& st = g_env.state();
    double x = st[0], x_dot = st[1], theta = st[2], theta_dot = st[3];

    int track_y = stage_rc.top + content_h * 68 / 100;
    int track_margin = 40;
    int track_left = stage_rc.left + track_margin;
    int track_right = stage_rc.right - track_margin;
    int track_w = track_right - track_left;

    // Draw horizontal track line and boundary markers at +/- 2.4m
    HPEN track_pen = CreatePen(PS_SOLID, 3, RGB(110, 125, 150));
    op = SelectObject(hdc, track_pen);
    MoveToEx(hdc, track_left, track_y, nullptr);
    LineTo(hdc, track_right, track_y);
    SelectObject(hdc, op);
    DeleteObject(track_pen);

    HPEN limit_pen = CreatePen(PS_DASH, 1, RGB(220, 75, 75));
    op = SelectObject(hdc, limit_pen);
    MoveToEx(hdc, track_left, track_y - 90, nullptr);
    LineTo(hdc, track_left, track_y + 30);
    MoveToEx(hdc, track_right, track_y - 90, nullptr);
    LineTo(hdc, track_right, track_y + 30);
    SelectObject(hdc, op);
    DeleteObject(limit_pen);

    // Map cart x in [-2.4, +2.4] to pixel X
    double norm_x = std::clamp((x + rl::CartPole::x_threshold) / (2.0 * rl::CartPole::x_threshold), 0.0, 1.0);
    int cart_cx = track_left + static_cast<int>(norm_x * track_w);
    int cart_w = 88;
    int cart_h = 44;
    int cart_top = track_y - cart_h;

    // Draw Cart body
    RECT cart_rc{cart_cx - cart_w / 2, cart_top, cart_cx + cart_w / 2, track_y};
    COLORREF cart_col = g_terminated ? RGB(200, 60, 60) : (g_truncated ? RGB(40, 180, 100) : RGB(56, 189, 248));
    HBRUSH cart_br = CreateSolidBrush(cart_col);
    FillRect(hdc, &cart_rc, cart_br);
    DeleteObject(cart_br);

    // Draw Cart wheels
    HBRUSH wheel_br = CreateSolidBrush(RGB(220, 230, 245));
    ob = SelectObject(hdc, wheel_br);
    Ellipse(hdc, cart_cx - 32, track_y - 10, cart_cx - 12, track_y + 10);
    Ellipse(hdc, cart_cx + 12, track_y - 10, cart_cx + 32, track_y + 10);
    SelectObject(hdc, ob);
    DeleteObject(wheel_br);

    // Draw Pole from top-center of cart (length = 175px)
    int pivot_x = cart_cx;
    int pivot_y = cart_top + 6;
    double pole_len = 175.0;
    int tip_x = pivot_x + static_cast<int>(pole_len * std::sin(theta));
    int tip_y = pivot_y - static_cast<int>(pole_len * std::cos(theta));

    // Faint +/- 12 deg failure cones from pivot
    HPEN cone_pen = CreatePen(PS_DOT, 1, RGB(120, 70, 75));
    op = SelectObject(hdc, cone_pen);
    MoveToEx(hdc, pivot_x, pivot_y, nullptr);
    LineTo(hdc, pivot_x + static_cast<int>(pole_len * std::sin(rl::CartPole::theta_threshold)),
           pivot_y - static_cast<int>(pole_len * std::cos(rl::CartPole::theta_threshold)));
    MoveToEx(hdc, pivot_x, pivot_y, nullptr);
    LineTo(hdc, pivot_x - static_cast<int>(pole_len * std::sin(rl::CartPole::theta_threshold)),
           pivot_y - static_cast<int>(pole_len * std::cos(rl::CartPole::theta_threshold)));
    SelectObject(hdc, op);
    DeleteObject(cone_pen);

    // Solid Pole beam
    HPEN pole_pen = CreatePen(PS_SOLID, 10, RGB(250, 204, 21));
    op = SelectObject(hdc, pole_pen);
    MoveToEx(hdc, pivot_x, pivot_y, nullptr);
    LineTo(hdc, tip_x, tip_y);
    SelectObject(hdc, op);
    DeleteObject(pole_pen);

    // Pivot pin
    HBRUSH pin_br = CreateSolidBrush(RGB(255, 255, 255));
    ob = SelectObject(hdc, pin_br);
    Ellipse(hdc, pivot_x - 6, pivot_y - 6, pivot_x + 6, pivot_y + 6);
    SelectObject(hdc, ob);
    DeleteObject(pin_br);

    // Draw push force indicator arrow beside the cart
    if (g_has_pushed && !g_terminated && !g_truncated) {
        HPEN fpen = CreatePen(PS_SOLID, 4, RGB(74, 222, 128));
        op = SelectObject(hdc, fpen);
        int ay = cart_top + cart_h / 2;
        if (g_last_action == rl::CartAction::Right) {
            MoveToEx(hdc, cart_rc.right + 8, ay, nullptr);
            LineTo(hdc, cart_rc.right + 42, ay);
        } else {
            MoveToEx(hdc, cart_rc.left - 8, ay, nullptr);
            LineTo(hdc, cart_rc.left - 42, ay);
        }
        SelectObject(hdc, op);
        DeleteObject(fpen);
    }

    // State telemetry readout at bottom of left panel
    double deg = theta * 180.0 / 3.14159265358979323846;
    std::wstringstream tele;
    tele << std::fixed << std::setprecision(2)
         << L"x = " << x << L" m   |   x_dot = " << x_dot
         << L" m/s   |   theta = " << deg << L" deg   |   theta_dot = " << theta_dot << L" rad/s";
    if (g_terminated) tele << L"   [FELL! Press R]";
    else if (g_truncated) tele << L"   [SOLVED 500/500!]";

    RECT tele_rc{stage_rc.left + 12, stage_rc.bottom - 44, stage_rc.right - 12, stage_rc.bottom - 12};
    SetTextColor(hdc, RGB(220, 230, 245));
    DrawTextW(hdc, tele.str().c_str(), -1, &tele_rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    // 3. Right Panel: 2 Live Learning Charts
    int rx = 2 * pad + left_w;
    int gap = 14;
    int chart_h = (content_h - gap) / 2;

    RECT p1{rx, content_top, rx + right_w, content_top + chart_h};
    RECT p2{rx, content_top + chart_h + gap, rx + right_w, content_top + content_h};

    DrawLineChart(hdc, p1, g_hist_smooth_return,
                  L"1. Smoothed Steps Balanced (Moving Avg toward 500)", RGB(74, 222, 128), 500.0);
    DrawLineChart(hdc, p2, g_hist_raw_return,
                  L"2. Raw Episode Return (Last 600 Episodes)", RGB(56, 189, 248), 500.0);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN: {
            switch (wParam) {
                case VK_LEFT:
                case 'Q':
                case 'A':
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    ApplyManualOrGreedy(rl::CartAction::Left);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_RIGHT:
                case 'D':
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    ApplyManualOrGreedy(rl::CartAction::Right);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'T':
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    TrainBatch(500);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'L':
                    if (g_mode == PlayMode::LiveTraining) {
                        g_mode = PlayMode::Manual;
                        KillTimer(hwnd, 1);
                    } else {
                        g_mode = PlayMode::LiveTraining;
                        ResetEpisode();
                        SetTimer(hwnd, 1, 16, nullptr); // ~60 Hz live steps
                    }
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_SPACE:
                    ResetEpisode();
                    g_mode = PlayMode::GreedyWatch;
                    SetTimer(hwnd, 1, 20, nullptr); // Real-time 50 Hz (tau = 0.02s!)
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'C':
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    ResetAgentAndPlots();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'R':
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    ResetEpisode();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_ESCAPE:
                    PostQuitMessage(0);
                    break;
            }
            return 0;
        }

        case WM_TIMER: {
            if (g_mode == PlayMode::LiveTraining) {
                StepLiveTraining();
                InvalidateRect(hwnd, nullptr, FALSE);
            } else if (g_mode == PlayMode::GreedyWatch) {
                if (!g_terminated && !g_truncated) {
                    int s = rl::CartPoleDiscretizer::discretize(g_env.state());
                    ApplyManualOrGreedy(ToCartAction(g_agent.greedy_action(s)));
                    InvalidateRect(hwnd, nullptr, FALSE);
                } else {
                    g_mode = PlayMode::Manual;
                    KillTimer(hwnd, 1);
                    InvalidateRect(hwnd, nullptr, FALSE);
                }
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
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

} // namespace

int main() {
    ResetAgentAndPlots();

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    const wchar_t CLASS_NAME[] = L"RLCartPoleDashboardClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"C++ RL CartPole Arena: Discretized Q-Learning & Live Physics",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1240, 740,
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
