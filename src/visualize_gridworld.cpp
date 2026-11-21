/**
 * @file visualize_gridworld.cpp
 * @brief Native Windows GUI Dashboard comparing Q-Learning (Off-Policy) vs. SARSA (On-Policy) in parallel!
 *
 * Layout:
 * - Top Left Panel:    Q-Learning GridWorld (Cyan agent, Q-value heatmap & greedy policy arrows)
 * - Bottom Left Panel: SARSA GridWorld (Magenta agent, Q-value heatmap & greedy policy arrows)
 * - Right Panel:       3 Dual-Curve Live Plots comparing Q-Learning (Cyan) vs. SARSA (Magenta):
 *                      1. Smoothed Episode Return (Moving Average)
 *                      2. Mean TD Error |Target - Q(s,a)| (Bellman Loss)
 *                      3. Mean Max Q-Value Across Grid
 *
 * Controls:
 * - [L]: Toggle Live Parallel Training (watch both agents explore & learn side by side!)
 * - [T]: Fast-Train BOTH agents +100 episodes in parallel
 * - [Space]: Watch BOTH agents race using their learned Greedy Policies side by side!
 * - [K]: Load Classic "Cliff Walking" Trap Map (8x4) — highlights Q-Learning (risky edge) vs. SARSA (safe path)!
 * - [M]: Generate New Random Map (random shape, walls & traps, guaranteed solvable via BFS)
 * - [C]: Clear / Reset Q-Tables & Plots
 * - [R]: Reset current episode
 * - [Esc]: Quit
 */

#ifndef UNICODE
#define UNICODE
#endif

#include "rl/gridworld.hpp"
#include "rl/q_learning_agent.hpp"
#include "rl/sarsa_agent.hpp"
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// @brief Two synchronized GridWorld instances (identical maps, independent agent positions).
rl::GridWorld g_env_q(5, 5);
rl::GridWorld g_env_s(5, 5);

/// @brief Both tabular agents trained with identical hyperparameters (alpha=0.2, gamma=0.95, epsilon=0.2).
rl::QLearningAgent g_q_agent(g_env_q.num_states(), g_env_q.num_actions(), 0.2, 0.95, 0.2, 42);
rl::SarsaAgent g_s_agent(g_env_s.num_states(), g_env_s.num_actions(), 0.2, 0.95, 0.2, 42);

unsigned int g_map_seed = 100;

/// @brief Per-episode runtime tracking for an agent-environment pair.
struct EpisodeTracker {
    double last_reward{0.0};
    double total_return{0.0};
    double td_loss_sum{0.0};
    int steps{0};
    bool terminated{false};
    bool truncated{false};
    int episodes_done{0};
    rl::GridAction pending_sarsa_action{rl::GridAction::Right};

    std::vector<double> history_return;
    std::vector<double> history_loss;
    std::vector<double> history_mean_q;
};

EpisodeTracker g_track_q;
EpisodeTracker g_track_s;

enum class PlayMode {
    Idle = 0,
    GreedyRace = 1,
    LiveTraining = 2
};
PlayMode g_mode = PlayMode::Idle;

/**
 * @brief Computes mean max_a Q(s, a) over all Empty tiles for either QLearningAgent or SarsaAgent.
 * @tparam AgentType Either rl::QLearningAgent or rl::SarsaAgent.
 * @param env Reference to the GridWorld environment.
 * @param agent Reference to the agent.
 * @return Average max Q-value.
 */
template <typename AgentType>
double ComputeMeanMaxQ(const rl::GridWorld& env, const AgentType& agent) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < env.height(); ++y) {
        for (int x = 0; x < env.width(); ++x) {
            if (env.cell_at(x, y) == rl::CellType::Empty) {
                int s = y * env.width() + x;
                int best_a = static_cast<int>(agent.greedy_action(s));
                sum += agent.q_value(s, best_a);
                count++;
            }
        }
    }
    return count > 0 ? (sum / count) : 0.0;
}

/**
 * @brief Resets a single agent's episode state.
 */
void ResetSingleEpisode(rl::GridWorld& env, EpisodeTracker& tr, bool is_sarsa) {
    int s0 = env.reset();
    tr.last_reward = 0.0;
    tr.total_return = 0.0;
    tr.td_loss_sum = 0.0;
    tr.steps = 0;
    tr.terminated = false;
    tr.truncated = false;
    if (is_sarsa) {
        tr.pending_sarsa_action = g_s_agent.select_action(s0);
    }
}

/**
 * @brief Resets both environments' current episodes (keeps Q-tables and plots).
 */
void ResetBothEpisodes() {
    ResetSingleEpisode(g_env_q, g_track_q, false);
    ResetSingleEpisode(g_env_s, g_track_s, true);
}

/**
 * @brief Re-initializes both agents and clears all plot histories for the current map.
 */
void ResetAgentsAndPlots() {
    g_q_agent = rl::QLearningAgent(g_env_q.num_states(), g_env_q.num_actions(), 0.2, 0.95, 0.2, 42);
    g_s_agent = rl::SarsaAgent(g_env_s.num_states(), g_env_s.num_actions(), 0.2, 0.95, 0.2, 42);

    g_track_q = EpisodeTracker{};
    g_track_s = EpisodeTracker{};
    ResetBothEpisodes();
}

/**
 * @brief Records metrics at the end of a completed episode for plotting.
 */
template <typename AgentType>
void RecordMetrics(const rl::GridWorld& env, const AgentType& agent, EpisodeTracker& tr) {
    tr.episodes_done++;
    // Store exponential moving average (EMA) of return so Q-Learning vs SARSA curves are super clear!
    double prev_ret = tr.history_return.empty() ? tr.total_return : tr.history_return.back();
    double smooth_ret = tr.history_return.empty() ? tr.total_return : (0.85 * prev_ret + 0.15 * tr.total_return);
    tr.history_return.push_back(smooth_ret);

    double mean_loss = (tr.steps > 0) ? (tr.td_loss_sum / tr.steps) : 0.0;
    tr.history_loss.push_back(mean_loss);
    tr.history_mean_q.push_back(ComputeMeanMaxQ(env, agent));
}

/**
 * @brief Runs one single training step for Q-Learning (used in Live Training mode).
 */
void StepLiveQLearning() {
    if (g_track_q.terminated || g_track_q.truncated) {
        RecordMetrics(g_env_q, g_q_agent, g_track_q);
        g_q_agent.decay_epsilon(0.995, 0.02);
        ResetSingleEpisode(g_env_q, g_track_q, false);
        return;
    }
    int s = g_env_q.agent_y() * g_env_q.width() + g_env_q.agent_x();
    rl::GridAction act = g_q_agent.select_action(s);
    rl::StepResult<int> res = g_env_q.step(act);
    double td_err = g_q_agent.update(s, act, res.reward, res.next_state, res.terminated);

    g_track_q.last_reward = res.reward;
    g_track_q.total_return += res.reward;
    g_track_q.td_loss_sum += td_err;
    g_track_q.steps++;
    g_track_q.terminated = res.terminated;
    g_track_q.truncated = res.truncated;
}

/**
 * @brief Runs one single training step for SARSA (used in Live Training mode).
 */
void StepLiveSarsa() {
    if (g_track_s.terminated || g_track_s.truncated) {
        RecordMetrics(g_env_s, g_s_agent, g_track_s);
        g_s_agent.decay_epsilon(0.995, 0.02);
        ResetSingleEpisode(g_env_s, g_track_s, true);
        return;
    }
    int s = g_env_s.agent_y() * g_env_s.width() + g_env_s.agent_x();
    rl::GridAction act = g_track_s.pending_sarsa_action;
    rl::StepResult<int> res = g_env_s.step(act);
    rl::GridAction next_act = g_s_agent.select_action(res.next_state);
    double td_err = g_s_agent.update(s, act, res.reward, res.next_state, next_act, res.terminated);

    g_track_s.pending_sarsa_action = next_act;
    g_track_s.last_reward = res.reward;
    g_track_s.total_return += res.reward;
    g_track_s.td_loss_sum += td_err;
    g_track_s.steps++;
    g_track_s.terminated = res.terminated;
    g_track_s.truncated = res.truncated;
}

/**
 * @brief Fast-trains BOTH Q-Learning and SARSA in parallel for `num_episodes` episodes each!
 * @param num_episodes Number of episodes to train each agent.
 */
void TrainBothBatch(int num_episodes) {
    for (int ep = 0; ep < num_episodes; ++ep) {
        // 1. Q-Learning episode
        ResetSingleEpisode(g_env_q, g_track_q, false);
        while (!g_track_q.terminated && !g_track_q.truncated) {
            StepLiveQLearning();
        }
        RecordMetrics(g_env_q, g_q_agent, g_track_q);
        g_q_agent.decay_epsilon(0.995, 0.02);

        // 2. SARSA episode
        ResetSingleEpisode(g_env_s, g_track_s, true);
        while (!g_track_s.terminated && !g_track_s.truncated) {
            StepLiveSarsa();
        }
        RecordMetrics(g_env_s, g_s_agent, g_track_s);
        g_s_agent.decay_epsilon(0.995, 0.02);
    }
    ResetBothEpisodes();
}

/**
 * @brief Returns a directional arrow for an agent's greedy action in `state`.
 */
template <typename AgentType>
const wchar_t* PolicyArrow(const AgentType& agent, int state) {
    switch (agent.greedy_action(state)) {
        case rl::GridAction::Up:    return L"\u2191";
        case rl::GridAction::Right: return L"\u2192";
        case rl::GridAction::Down:  return L"\u2193";
        case rl::GridAction::Left:  return L"\u2190";
    }
    return L"?";
}

/**
 * @brief Draws one GridWorld panel (title, cells, heatmap, policy arrows, and agent circle).
 */
template <typename AgentType>
void DrawGridPanel(HDC hdc, const RECT& panel_rc, const rl::GridWorld& env,
                   const AgentType& agent, const EpisodeTracker& tr,
                   const std::wstring& title, COLORREF accent_color) {
    // Panel border & title
    HBRUSH panel_bg = CreateSolidBrush(RGB(30, 35, 45));
    FillRect(hdc, &panel_rc, panel_bg);
    DeleteObject(panel_bg);

    HPEN border_pen = CreatePen(PS_SOLID, 1, accent_color);
    HGDIOBJ old_p = SelectObject(hdc, border_pen);
    HGDIOBJ old_b = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, panel_rc.left, panel_rc.top, panel_rc.right, panel_rc.bottom);
    SelectObject(hdc, old_b);
    SelectObject(hdc, old_p);
    DeleteObject(border_pen);

    std::wstringstream hdr;
    hdr << std::fixed << std::setprecision(1)
        << title << L"   |   Ep: " << tr.episodes_done
        << L"   |   eps: " << std::setprecision(2) << agent.epsilon()
        << std::setprecision(1)
        << L"   |   Return: " << tr.total_return;
    RECT hdr_rc{panel_rc.left + 10, panel_rc.top + 6, panel_rc.right - 10, panel_rc.top + 28};
    SetTextColor(hdc, accent_color);
    DrawTextW(hdc, hdr.str().c_str(), -1, &hdr_rc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    int pad = 10;
    int avail_w = (panel_rc.right - panel_rc.left) - 2 * pad;
    int avail_h = (panel_rc.bottom - panel_rc.top) - 34 - pad;
    int cell_size = std::min(avail_w / env.width(), avail_h / env.height());
    if (cell_size < 18) cell_size = 18;

    int grid_w = cell_size * env.width();
    int grid_h = cell_size * env.height();
    int start_x = panel_rc.left + pad + (avail_w - grid_w) / 2;
    int start_y = panel_rc.top + 32 + (avail_h - grid_h) / 2;

    for (int y = 0; y < env.height(); ++y) {
        for (int x = 0; x < env.width(); ++x) {
            int s = y * env.width() + x;
            int best_a = static_cast<int>(agent.greedy_action(s));
            double max_q = agent.q_value(s, best_a);
            rl::CellType ctype = env.cell_at(x, y);

            RECT cell_rc{
                start_x + x * cell_size + 2,
                start_y + y * cell_size + 2,
                start_x + (x + 1) * cell_size - 2,
                start_y + (y + 1) * cell_size - 2
            };

            COLORREF cell_color = RGB(45, 52, 66);
            if (ctype == rl::CellType::Goal) {
                cell_color = RGB(34, 139, 87);
            } else if (ctype == rl::CellType::Trap) {
                cell_color = RGB(185, 45, 45);
            } else if (ctype == rl::CellType::Wall) {
                cell_color = RGB(16, 18, 24);
            } else if (tr.episodes_done > 0) {
                double norm = std::clamp(max_q / 10.0, -1.0, 1.0);
                if (norm >= 0.0) {
                    int g = static_cast<int>(52 + norm * 80);
                    int b = static_cast<int>(66 + norm * 25);
                    cell_color = RGB(38, g, b);
                } else {
                    int r = static_cast<int>(45 + (-norm) * 95);
                    cell_color = RGB(r, 45, 55);
                }
            }

            HBRUSH c_brush = CreateSolidBrush(cell_color);
            FillRect(hdc, &cell_rc, c_brush);
            DeleteObject(c_brush);

            // Agent circle
            if (x == env.agent_x() && y == env.agent_y()) {
                int margin = cell_size / 4;
                HBRUSH a_brush = CreateSolidBrush(accent_color);
                HPEN a_pen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
                HGDIOBJ ob = SelectObject(hdc, a_brush);
                HGDIOBJ op = SelectObject(hdc, a_pen);

                Ellipse(hdc, cell_rc.left + margin, cell_rc.top + margin,
                        cell_rc.right - margin, cell_rc.bottom - margin);

                SelectObject(hdc, ob);
                SelectObject(hdc, op);
                DeleteObject(a_brush);
                DeleteObject(a_pen);
            }

            SetTextColor(hdc, RGB(230, 235, 245));
            if (ctype == rl::CellType::Goal) {
                DrawTextW(hdc, L"G", -1, &cell_rc, DT_CENTER | DT_BOTTOM | DT_SINGLELINE);
            } else if (ctype == rl::CellType::Trap) {
                DrawTextW(hdc, L"X", -1, &cell_rc, DT_CENTER | DT_BOTTOM | DT_SINGLELINE);
            } else if (ctype == rl::CellType::Wall) {
                SetTextColor(hdc, RGB(90, 100, 115));
                DrawTextW(hdc, L"#", -1, &cell_rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            } else {
                std::wstringstream ss;
                ss << std::fixed << std::setprecision(1) << PolicyArrow(agent, s) << L" " << max_q;
                RECT lbl = cell_rc;
                lbl.bottom -= 2;
                DrawTextW(hdc, ss.str().c_str(), -1, &lbl, DT_CENTER | DT_BOTTOM | DT_SINGLELINE);
            }
        }
    }
}

/**
 * @brief Draws a dual-series comparison line chart (Q-Learning in Cyan vs. SARSA in Magenta).
 */
void DrawComparisonChart(HDC hdc, const RECT& plot_rect,
                         const std::vector<double>& data_q,
                         const std::vector<double>& data_s,
                         const std::wstring& title) {
    HBRUSH box_brush = CreateSolidBrush(RGB(34, 40, 52));
    FillRect(hdc, &plot_rect, box_brush);
    DeleteObject(box_brush);

    HPEN border_pen = CreatePen(PS_SOLID, 1, RGB(65, 75, 95));
    HGDIOBJ old_p = SelectObject(hdc, border_pen);
    HGDIOBJ old_b = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, plot_rect.left, plot_rect.top, plot_rect.right, plot_rect.bottom);
    SelectObject(hdc, old_b);
    SelectObject(hdc, old_p);
    DeleteObject(border_pen);

    RECT title_rc{plot_rect.left + 10, plot_rect.top + 6, plot_rect.right - 10, plot_rect.top + 26};
    SetTextColor(hdc, RGB(225, 232, 245));
    DrawTextW(hdc, title.c_str(), -1, &title_rc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    if (data_q.empty() && data_s.empty()) {
        RECT empty_rc = plot_rect;
        SetTextColor(hdc, RGB(120, 135, 160));
        DrawTextW(hdc, L"Press [L] (Live Parallel Train) or [T] (+100 Episodes) to compare Q-Learning vs SARSA",
                  -1, &empty_rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    auto calc_window = [](const std::vector<double>& v) {
        return v.size() > 300 ? (v.size() - 300) : 0;
    };
    std::size_t sq = calc_window(data_q);
    std::size_t ss = calc_window(data_s);

    double min_v = 1e9, max_v = -1e9;
    for (std::size_t i = sq; i < data_q.size(); ++i) {
        min_v = std::min(min_v, data_q[i]);
        max_v = std::max(max_v, data_q[i]);
    }
    for (std::size_t i = ss; i < data_s.size(); ++i) {
        min_v = std::min(min_v, data_s[i]);
        max_v = std::max(max_v, data_s[i]);
    }
    if (std::abs(max_v - min_v) < 1e-5) {
        max_v += 1.0;
        min_v -= 1.0;
    }

    // Latest values legend on top-right
    std::wstringstream leg;
    leg << std::fixed << std::setprecision(2);
    if (!data_q.empty()) leg << L"Q-Learn: " << data_q.back();
    if (!data_s.empty()) leg << L"   |   SARSA: " << data_s.back();
    SetTextColor(hdc, RGB(200, 215, 235));
    DrawTextW(hdc, leg.str().c_str(), -1, &title_rc, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    int inner_l = plot_rect.left + 14;
    int inner_r = plot_rect.right - 14;
    int inner_t = plot_rect.top + 32;
    int inner_b = plot_rect.bottom - 12;
    int w = std::max(1, inner_r - inner_l);
    int h = std::max(1, inner_b - inner_t);

    if (min_v < 0.0 && max_v > 0.0) {
        int zero_y = inner_b - static_cast<int>(((0.0 - min_v) / (max_v - min_v)) * h);
        HPEN zp = CreatePen(PS_DOT, 1, RGB(90, 100, 120));
        old_p = SelectObject(hdc, zp);
        MoveToEx(hdc, inner_l, zero_y, nullptr);
        LineTo(hdc, inner_r, zero_y);
        SelectObject(hdc, old_p);
        DeleteObject(zp);
    }

    auto plot_series = [&](const std::vector<double>& v, std::size_t start_i, COLORREF col) {
        if (v.size() <= start_i) return;
        HPEN cp = CreatePen(PS_SOLID, 2, col);
        HGDIOBJ op = SelectObject(hdc, cp);
        std::size_t count = v.size() - start_i;
        for (std::size_t i = 0; i < count; ++i) {
            int px = inner_l + (count > 1 ? static_cast<int>((static_cast<double>(i) / (count - 1)) * w) : w / 2);
            int py = inner_b - static_cast<int>(((v[start_i + i] - min_v) / (max_v - min_v)) * h);
            if (i == 0) MoveToEx(hdc, px, py, nullptr);
            else LineTo(hdc, px, py);
        }
        SelectObject(hdc, op);
        DeleteObject(cp);
    };

    plot_series(data_q, sq, RGB(56, 189, 248));  // Cyan = Q-Learning
    plot_series(data_s, ss, RGB(232, 121, 249)); // Magenta = SARSA
}

/**
 * @brief Renders the full parallel comparison dashboard.
 */
void DrawDashboard(HDC hdc, const RECT& client_rect) {
    int win_w = client_rect.right - client_rect.left;
    int win_h = client_rect.bottom - client_rect.top;

    HBRUSH bg = CreateSolidBrush(RGB(20, 24, 32));
    FillRect(hdc, &client_rect, bg);
    DeleteObject(bg);
    SetBkMode(hdc, TRANSPARENT);

    int pad = 16;
    int hud_h = 95;

    // Top HUD
    std::wstring mode_txt = L"Idle";
    if (g_mode == PlayMode::LiveTraining) mode_txt = L"LIVE PARALLEL TRAINING";
    else if (g_mode == PlayMode::GreedyRace) mode_txt = L"GREEDY POLICY RACE";

    std::wstringstream top_ss;
    top_ss << L"PARALLEL ARENA: Q-Learning (Cyan, Off-Policy) vs. SARSA (Magenta, On-Policy)   |   Mode: "
           << mode_txt << L"   |   Map: " << g_env_q.width() << L"x" << g_env_q.height();
    RECT hud1{pad, 8, win_w - pad, 36};
    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextW(hdc, top_ss.str().c_str(), -1, &hud1, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl1 =
        L"[L]: Toggle Live Parallel Training   |   [T]: Fast-Train Both +100 Ep   |   [Space]: Greedy Policy Race!";
    RECT hud2{pad, 38, win_w - pad, 64};
    SetTextColor(hdc, RGB(125, 211, 252));
    DrawTextW(hdc, ctrl1.c_str(), -1, &hud2, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    std::wstring ctrl2 =
        L"[K]: Load Cliff-Walking Map (8x4)   |   [M]: Random Maze Map   |   [C]: Clear Q-Tables   |   [R]: Reset Ep   |   [Esc]: Quit";
    RECT hud3{pad, 66, win_w - pad, 90};
    SetTextColor(hdc, RGB(180, 195, 215));
    DrawTextW(hdc, ctrl2.c_str(), -1, &hud3, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    int content_top = hud_h;
    int content_h = win_h - content_top - pad;
    int left_w = (win_w - 3 * pad) * 52 / 100;
    int right_w = (win_w - 3 * pad) - left_w;

    // Left Side: Stacked GridWorlds (Top = Q-Learning, Bottom = SARSA)
    int gap = 12;
    int grid_panel_h = (content_h - gap) / 2;
    RECT q_rc{pad, content_top, pad + left_w, content_top + grid_panel_h};
    RECT s_rc{pad, content_top + grid_panel_h + gap, pad + left_w, content_top + content_h};

    DrawGridPanel(hdc, q_rc, g_env_q, g_q_agent, g_track_q, L"Q-LEARNING (Off-Policy)", RGB(56, 189, 248));
    DrawGridPanel(hdc, s_rc, g_env_s, g_s_agent, g_track_s, L"SARSA (On-Policy)", RGB(232, 121, 249));

    // Right Side: 3 Dual-Curve Plots
    int rx = 2 * pad + left_w;
    int plot_h = (content_h - 2 * gap) / 3;
    RECT p1{rx, content_top, rx + right_w, content_top + plot_h};
    RECT p2{rx, content_top + plot_h + gap, rx + right_w, content_top + 2 * plot_h + gap};
    RECT p3{rx, content_top + 2 * (plot_h + gap), rx + right_w, content_top + content_h};

    DrawComparisonChart(hdc, p1, g_track_q.history_return, g_track_s.history_return,
                        L"1. Smoothed Training Return (Cyan=Q-Learn, Magenta=SARSA)");
    DrawComparisonChart(hdc, p2, g_track_q.history_loss, g_track_s.history_loss,
                        L"2. Mean TD Error / Bellman Loss (Cyan=Q-Learn, Magenta=SARSA)");
    DrawComparisonChart(hdc, p3, g_track_q.history_mean_q, g_track_s.history_mean_q,
                        L"3. Mean Max Q-Value Across Grid (Cyan=Q-Learn, Magenta=SARSA)");
}

/**
 * @brief Win32 Window Procedure handling keyboard commands and parallel animation timer.
 */
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN: {
            switch (wParam) {
                case 'L':
                    if (g_mode == PlayMode::LiveTraining) {
                        g_mode = PlayMode::Idle;
                        KillTimer(hwnd, 1);
                    } else {
                        g_mode = PlayMode::LiveTraining;
                        ResetBothEpisodes();
                        SetTimer(hwnd, 1, 30, nullptr);
                    }
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'T':
                    g_mode = PlayMode::Idle;
                    KillTimer(hwnd, 1);
                    TrainBothBatch(100);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'K':
                    // Classic Cliff Walking layout (8x4) to see Q-Learning vs SARSA difference!
                    g_mode = PlayMode::Idle;
                    KillTimer(hwnd, 1);
                    g_env_q.set_cliff_layout();
                    g_env_s.set_cliff_layout();
                    ResetAgentsAndPlots();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'M': {
                    g_mode = PlayMode::Idle;
                    KillTimer(hwnd, 1);
                    g_map_seed += 19;
                    std::mt19937 rng(g_map_seed);
                    std::uniform_int_distribution<int> w_dist(5, 9);
                    std::uniform_int_distribution<int> h_dist(4, 7);
                    int w = w_dist(rng);
                    int h = h_dist(rng);
                    g_env_q.randomize_layout(w, h, 0.14, 0.15, g_map_seed);
                    g_env_s.randomize_layout(w, h, 0.14, 0.15, g_map_seed);
                    ResetAgentsAndPlots();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                }
                case 'C':
                    g_mode = PlayMode::Idle;
                    KillTimer(hwnd, 1);
                    ResetAgentsAndPlots();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case 'R':
                    g_mode = PlayMode::Idle;
                    KillTimer(hwnd, 1);
                    ResetBothEpisodes();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    break;
                case VK_SPACE:
                    ResetBothEpisodes();
                    g_mode = PlayMode::GreedyRace;
                    SetTimer(hwnd, 1, 220, nullptr);
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
                StepLiveQLearning();
                StepLiveSarsa();
                InvalidateRect(hwnd, nullptr, FALSE);
            } else if (g_mode == PlayMode::GreedyRace) {
                bool q_active = !g_track_q.terminated && !g_track_q.truncated;
                bool s_active = !g_track_s.terminated && !g_track_s.truncated;
                if (q_active) {
                    int s = g_env_q.agent_y() * g_env_q.width() + g_env_q.agent_x();
                    rl::StepResult<int> res = g_env_q.step(g_q_agent.greedy_action(s));
                    g_track_q.total_return += res.reward;
                    g_track_q.terminated = res.terminated;
                    g_track_q.truncated = res.truncated;
                }
                if (s_active) {
                    int s = g_env_s.agent_y() * g_env_s.width() + g_env_s.agent_x();
                    rl::StepResult<int> res = g_env_s.step(g_s_agent.greedy_action(s));
                    g_track_s.total_return += res.reward;
                    g_track_s.terminated = res.terminated;
                    g_track_s.truncated = res.truncated;
                }
                if (!q_active && !s_active) {
                    g_mode = PlayMode::Idle;
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
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

} // namespace

int main() {
    // Start directly on the Cliff-Walking map so the Q-Learning vs SARSA contrast is immediately visible!
    g_env_q.set_cliff_layout();
    g_env_s.set_cliff_layout();
    ResetAgentsAndPlots();

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    const wchar_t CLASS_NAME[] = L"RLParallelComparisonClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"C++ RL Parallel Arena: Q-Learning vs. SARSA",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 820,
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
