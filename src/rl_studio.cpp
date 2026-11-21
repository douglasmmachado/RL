 /**
 * @file rl_studio.cpp
 * @brief Complete C++ Interactive RL Studio built with Dear ImGui + ImPlot + DirectX11 + RLtools!
 *
 * Unifies ALL 7 environments into one interactive desktop studio:
 *   0. GridWorld (Tabular Q-Learning vs. SARSA — Cliff Walking & Random BFS Mazes)
 *   1. CartPole-Continuous (RLtools SAC vs. PPO vs. TD3)
 *   2. Pendulum-v1 Swing-Up (RLtools Built-in — SAC vs. PPO vs. TD3)
 *   3. Acrobot 2-Link Swing-Up (RLtools Built-in — SAC vs. PPO vs. TD3)
 *   4. MountainCar-Continuous (RLtools SAC vs. PPO vs. TD3)
 *   5. Drone2D Quadrotor Hover (RLtools SAC vs. PPO vs. TD3)
 *   6. ContinuousGrid2D Trap Arena (RLtools SAC vs. PPO vs. TD3 with Random Trap Maps)
 *
 * Educational Features:
 *   - Interactive Hyperparameter Sliders with hover tooltips explaining math & failure modes
 *   - "Snapshot Run" button to save any learning curve and overlay different hyperparameter runs!
 *   - Toggleable X-Axis: Rounds vs. Environment Steps (Sample Efficiency) vs. Wall-Clock Time (Speed)
 *   - Zoomable ImPlot charts for Return, Episode Length, Action Timeline, and Tabular TD Loss / Mean Q
 *   - Interactive Rollout Scrubber (Play/Pause, 0.25x-2x Speed, frame-by-frame slider)
 */

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "implot.h"

#include "rl/studio_engines.hpp"

#include <d3d11.h>
#include <tchar.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Forward declare ImGui Win32 message handler
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

// DirectX 11 globals
ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;
UINT g_ResizeWidth = 0, g_ResizeHeight = 0;

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (pBackBuffer) {
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
        pBackBuffer->Release();
    }
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    HRESULT res = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, featureLevelArray, 2,
        D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED) {
        res = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, featureLevelArray, 2,
            D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    }
    if (res != S_OK) return false;
    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    switch (msg) {
        case WM_SIZE:
            if (wParam == SIZE_MINIMIZED) return 0;
            g_ResizeWidth = (UINT)LOWORD(lParam);
            g_ResizeHeight = (UINT)HIWORD(lParam);
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

/**
 * @brief Renders an educational tooltip when hovering over a hyperparameter or control.
 */
void HelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/**
 * @brief Draws a continuous environment frame inside an ImGui DrawList canvas!
 */
void DrawContinuousCanvas(ImDrawList* dl, ImVec2 p0, ImVec2 sz, int env_kind,
                          const std::vector<studio::RenderFrame>& rollout, int frame_idx, ImU32 accent) {
    ImVec2 p1(p0.x + sz.x, p0.y + sz.y);
    dl->AddRectFilled(p0, p1, IM_COL32(22, 26, 36, 255), 6.0f);
    dl->AddRect(p0, p1, accent, 6.0f, 0, 1.5f);

    if (rollout.empty()) return;
    int idx = std::clamp(frame_idx, 0, static_cast<int>(rollout.size()) - 1);
    const auto& f = rollout[idx];

    if (env_kind == 0) { // CartPole
        float ty = p0.y + sz.y * 0.76f;
        float tl = p0.x + 24.0f, tr = p1.x - 24.0f;
        dl->AddLine(ImVec2(tl, ty), ImVec2(tr, ty), IM_COL32(110, 125, 150, 255), 2.5f);
        float nx = std::clamp((f.s[0] + 2.4f) / 4.8f, 0.0f, 1.0f);
        float cx = tl + nx * (tr - tl);
        float cw = 48.0f, ch = 22.0f;
        ImU32 ccol = f.terminated ? IM_COL32(210, 60, 60, 255) : accent;
        dl->AddRectFilled(ImVec2(cx - cw * 0.5f, ty - ch), ImVec2(cx + cw * 0.5f, ty), ccol, 4.0f);
        float plen = std::max(36.0f, sz.y * 0.52f);
        ImVec2 piv(cx, ty - ch + 3.0f);
        ImVec2 tip(piv.x + plen * std::sin(f.s[1]), piv.y - plen * std::cos(f.s[1]));
        dl->AddLine(piv, tip, IM_COL32(250, 204, 21, 255), 5.5f);
        dl->AddCircleFilled(piv, 4.0f, IM_COL32(255, 255, 255, 255));
        if (std::abs(f.a[0]) > 0.05f && !f.terminated) {
            float ay = ty - ch * 0.5f;
            float sx = (f.a[0] > 0) ? (cx + cw * 0.5f + 4.0f) : (cx - cw * 0.5f - 4.0f);
            dl->AddLine(ImVec2(sx, ay), ImVec2(sx + f.a[0] * 36.0f, ay), IM_COL32(74, 222, 128, 255), 3.0f);
        }
    } else if (env_kind == 1) { // Pendulum
        ImVec2 c(p0.x + sz.x * 0.5f, p0.y + sz.y * 0.52f);
        float len = std::min(sz.x, sz.y) * 0.36f;
        dl->AddLine(c, ImVec2(c.x, c.y - len * 1.15f), IM_COL32(60, 150, 95, 160), 1.5f);
        ImVec2 bob(c.x + len * std::sin(f.s[0]), c.y - len * std::cos(f.s[0]));
        dl->AddLine(c, bob, IM_COL32(250, 204, 21, 255), 5.5f);
        dl->AddCircleFilled(bob, 9.0f, accent);
        dl->AddCircleFilled(c, 4.5f, IM_COL32(255, 255, 255, 255));
    } else if (env_kind == 2) { // Acrobot
        ImVec2 c(p0.x + sz.x * 0.5f, p0.y + sz.y * 0.52f);
        float l = std::min(sz.x, sz.y) * 0.20f;
        float gy = c.y - l * 1.15f;
        dl->AddLine(ImVec2(p0.x + 18, gy), ImVec2(p1.x - 18, gy), IM_COL32(70, 175, 105, 160), 1.5f);
        ImVec2 j1(c.x + l * std::sin(f.s[0]), c.y + l * std::cos(f.s[0]));
        ImVec2 j2(j1.x + l * std::sin(f.s[0] + f.s[1]), j1.y + l * std::cos(f.s[0] + f.s[1]));
        dl->AddLine(c, j1, accent, 5.5f);
        dl->AddLine(j1, j2, IM_COL32(250, 204, 21, 255), 4.5f);
        dl->AddCircleFilled(c, 4.5f, IM_COL32(255, 255, 255, 255));
        dl->AddCircleFilled(j1, 4.5f, IM_COL32(255, 255, 255, 255));
        dl->AddCircleFilled(j2, 5.5f, IM_COL32(255, 255, 255, 255));
    } else if (env_kind == 3) { // MountainCar
        auto to_pt = [&](float pos) {
            float nx = (pos + 1.2f) / 1.8f;
            float ny = (std::sin(3.0f * pos) + 1.0f) * 0.5f;
            return ImVec2(p0.x + 16.0f + nx * (sz.x - 32.0f), p1.y - 14.0f - ny * (sz.y - 28.0f));
        };
        for (int i = 0; i < 60; ++i) {
            float x0 = -1.2f + (1.8f * i) / 60.0f;
            float x1 = -1.2f + (1.8f * (i + 1)) / 60.0f;
            dl->AddLine(to_pt(x0), to_pt(x1), IM_COL32(115, 130, 155, 255), 2.0f);
        }
        ImVec2 gp = to_pt(0.45f);
        dl->AddLine(gp, ImVec2(gp.x, gp.y - 22.0f), IM_COL32(74, 222, 128, 255), 2.5f);
        dl->AddTriangleFilled(ImVec2(gp.x, gp.y - 22.0f), ImVec2(gp.x + 14.0f, gp.y - 16.0f),
                              ImVec2(gp.x, gp.y - 10.0f), IM_COL32(74, 222, 128, 255));
        dl->AddCircleFilled(to_pt(f.s[0]), 8.0f, accent);
    } else if (env_kind == 4) { // Drone2D
        auto to_px = [&](float wx, float wy) {
            return ImVec2(p0.x + ((wx + 1.5f) / 3.0f) * sz.x, p1.y - ((wy + 1.5f) / 3.0f) * sz.y);
        };
        ImVec2 tgt = to_px(0.0f, 0.0f);
        dl->AddLine(ImVec2(tgt.x - 12, tgt.y), ImVec2(tgt.x + 12, tgt.y), IM_COL32(74, 222, 128, 200), 1.5f);
        dl->AddLine(ImVec2(tgt.x, tgt.y - 12), ImVec2(tgt.x, tgt.y + 12), IM_COL32(74, 222, 128, 200), 1.5f);
        ImVec2 c = to_px(f.s[0], f.s[1]);
        float arm = 22.0f, ct = std::cos(f.s[2]), st = std::sin(f.s[2]);
        ImVec2 lp(c.x - arm * ct, c.y - arm * st);
        ImVec2 rp(c.x + arm * ct, c.y + arm * st);
        float fl = (f.a[0] + 1.0f) * 11.0f, fr = (f.a[1] + 1.0f) * 11.0f;
        dl->AddLine(lp, ImVec2(lp.x - fl * st, lp.y + fl * ct), IM_COL32(250, 204, 21, 255), 3.5f);
        dl->AddLine(rp, ImVec2(rp.x - fr * st, rp.y + fr * ct), IM_COL32(250, 204, 21, 255), 3.5f);
        dl->AddLine(lp, rp, f.terminated ? IM_COL32(210, 55, 55, 255) : accent, 5.5f);
    } else if (env_kind == 5) { // ContinuousGrid2D (Trap Arena)
        float side = std::min(sz.x, sz.y) - 12.0f;
        ImVec2 a0(p0.x + (sz.x - side) * 0.5f, p0.y + (sz.y - side) * 0.5f);
        auto w2s = [&](float wx, float wy) {
            return ImVec2(a0.x + (wx + 1.0f) * 0.5f * side, a0.y + side - (wy + 1.0f) * 0.5f * side);
        };
        const auto& m = rlt::rl::environments::continuous_grid_2d::active_map();
        dl->AddCircleFilled(w2s(m.start_x, m.start_y), 0.07f * 0.5f * side, IM_COL32(90, 105, 130, 220));
        for (int i = 0; i < m.num_traps; ++i) {
            dl->AddCircleFilled(w2s(m.trap_x[i], m.trap_y[i]), m.trap_r[i] * 0.5f * side, IM_COL32(190, 48, 48, 220));
        }
        dl->AddCircleFilled(w2s(m.target_x, m.target_y), m.goal_radius * 0.5f * side, IM_COL32(34, 155, 95, 230));
        for (int i = 0; i < idx; ++i) {
            dl->AddLine(w2s(rollout[i].s[0], rollout[i].s[1]),
                        w2s(rollout[i + 1].s[0], rollout[i + 1].s[1]), accent, 2.5f);
        }
        dl->AddCircleFilled(w2s(f.s[0], f.s[1]), 6.5f, accent);
    } else if (env_kind == 6) { // LunarLander (Moon Landing)
        auto to_px = [&](float wx, float wy) {
            return ImVec2(p0.x + ((wx + 1.3f) / 2.6f) * sz.x,
                          p1.y - 18.0f - (wy / 1.6f) * (sz.y - 26.0f));
        };
        // Starfield dots
        for (int i = 0; i < 18; ++i) {
            float sx = p0.x + std::fmod(i * 73.1f + 19.0f, sz.x - 12.0f) + 6.0f;
            float sy = p0.y + std::fmod(i * 41.7f + 11.0f, sz.y * 0.65f) + 6.0f;
            dl->AddCircleFilled(ImVec2(sx, sy), 1.3f, IM_COL32(210, 220, 245, 160));
        }
        // Lunar gray surface + craters
        float ground_y = to_px(0.0f, 0.0f).y;
        dl->AddRectFilled(ImVec2(p0.x + 2, ground_y), ImVec2(p1.x - 2, p1.y - 2), IM_COL32(68, 72, 84, 255));
        // Landing pad with yellow flags at x in [-0.28, +0.28]
        ImVec2 pad_l = to_px(-0.28f, 0.0f);
        ImVec2 pad_r = to_px(0.28f, 0.0f);
        dl->AddLine(pad_l, pad_r, IM_COL32(74, 222, 128, 255), 4.0f);
        dl->AddLine(pad_l, ImVec2(pad_l.x, pad_l.y - 14.0f), IM_COL32(250, 204, 21, 255), 2.0f);
        dl->AddLine(pad_r, ImVec2(pad_r.x, pad_r.y - 14.0f), IM_COL32(250, 204, 21, 255), 2.0f);

        // Descent trajectory trail
        for (int i = 0; i < idx; ++i) {
            dl->AddLine(to_px(rollout[i].s[0], rollout[i].s[1]),
                        to_px(rollout[i + 1].s[0], rollout[i + 1].s[1]),
                        IM_COL32(200, 210, 230, 70), 1.5f);
        }

        // Lunar Module sprite (body + landing legs + main/RCS engine flames)
        ImVec2 c = to_px(f.s[0], f.s[1]);
        float th = f.s[2], ct = std::cos(th), st = std::sin(th);
        auto rot = [&](float lx, float ly) {
            return ImVec2(c.x + lx * ct - ly * st, c.y - (lx * st + ly * ct));
        };

        // Main engine flame
        float throttle = (f.a[0] + 1.0f) * 0.5f;
        if (throttle > 0.08f && !f.terminated) {
            float flen = 8.0f + throttle * 26.0f;
            dl->AddTriangleFilled(rot(-6.0f, -8.0f), rot(6.0f, -8.0f), rot(0.0f, -8.0f - flen),
                                  IM_COL32(255, 145, 30, 240));
            dl->AddTriangleFilled(rot(-3.0f, -8.0f), rot(3.0f, -8.0f), rot(0.0f, -8.0f - flen * 0.6f),
                                  IM_COL32(255, 240, 120, 255));
        }
        // Side RCS thruster puff
        if (std::abs(f.a[1]) > 0.15f && !f.terminated) {
            float sgn = (f.a[1] > 0) ? -1.0f : 1.0f;
            dl->AddLine(rot(sgn * 10.0f, 3.0f), rot(sgn * (10.0f + std::abs(f.a[1]) * 14.0f), 3.0f),
                        IM_COL32(180, 235, 255, 220), 2.5f);
        }
        // Landing legs
        dl->AddLine(rot(-7.0f, -4.0f), rot(-14.0f, -13.0f), IM_COL32(210, 220, 235, 255), 2.0f);
        dl->AddLine(rot(7.0f, -4.0f), rot(14.0f, -13.0f), IM_COL32(210, 220, 235, 255), 2.0f);
        // Capsule polygon
        ImU32 body_col = (f.s[3] > 0.5f) ? IM_COL32(74, 222, 128, 255)
                       : ((f.s[3] < -0.5f) ? IM_COL32(220, 55, 55, 255) : accent);
        ImVec2 pts[4] = {rot(-10.0f, -7.0f), rot(10.0f, -7.0f), rot(7.0f, 9.0f), rot(-7.0f, 9.0f)};
        dl->AddConvexPolyFilled(pts, 4, body_col);
        dl->AddPolyline(pts, 4, IM_COL32(255, 255, 255, 220), ImDrawFlags_Closed, 1.5f);
    } else if (env_kind == 7) { // Reacher-2DOF Robotic Arm
        ImVec2 c(p0.x + sz.x * 0.5f, p0.y + sz.y * 0.5f);
        float scale = std::min(sz.x, sz.y) * 0.42f;
        // Workspace boundary circle
        dl->AddCircle(c, scale, IM_COL32(60, 72, 95, 160), 48, 1.2f);
        // Glowing Target Dot
        ImVec2 tgt(c.x + f.s[2] * scale, c.y - f.s[3] * scale);
        dl->AddCircleFilled(tgt, 10.0f, IM_COL32(74, 222, 128, 90));
        dl->AddCircleFilled(tgt, 6.0f, IM_COL32(74, 222, 128, 255));
        // 2-Link Robotic Arm (l1 = 0.5, l2 = 0.5)
        ImVec2 j1(c.x + 0.5f * scale * std::cos(f.s[0]), c.y - 0.5f * scale * std::sin(f.s[0]));
        ImVec2 tip(j1.x + 0.5f * scale * std::cos(f.s[0] + f.s[1]), j1.y - 0.5f * scale * std::sin(f.s[0] + f.s[1]));
        dl->AddLine(c, j1, accent, 7.0f);
        dl->AddLine(j1, tip, IM_COL32(250, 204, 21, 255), 5.5f);
        dl->AddCircleFilled(c, 6.0f, IM_COL32(255, 255, 255, 255));
        dl->AddCircleFilled(j1, 5.0f, IM_COL32(255, 255, 255, 255));
        dl->AddCircleFilled(tip, 5.5f, accent);
    } else if (env_kind == 8) { // Ball-and-Beam
        ImVec2 c(p0.x + sz.x * 0.5f, p0.y + sz.y * 0.58f);
        float half_b = std::min(sz.x * 0.42f, 140.0f);
        float th = f.s[2], ct = std::cos(th), st = std::sin(th);
        // Pivot stand triangle
        dl->AddTriangleFilled(ImVec2(c.x, c.y + 4), ImVec2(c.x - 16, c.y + 34), ImVec2(c.x + 16, c.y + 34),
                              IM_COL32(95, 110, 135, 255));
        // Tilting Beam
        ImVec2 bl(c.x - half_b * ct, c.y + half_b * st);
        ImVec2 br(c.x + half_b * ct, c.y - half_b * st);
        dl->AddLine(bl, br, IM_COL32(180, 195, 220, 255), 7.0f);
        // Center target tick on beam
        dl->AddCircleFilled(c, 4.5f, IM_COL32(74, 222, 128, 255));
        // Rolling Steel Ball at position x in [-1, 1]
        float bx = c.x + (f.s[0] * half_b) * ct - 12.0f * st;
        float by = c.y - (f.s[0] * half_b) * st - 12.0f * ct;
        dl->AddCircleFilled(ImVec2(bx, by), 10.0f, f.terminated ? IM_COL32(215, 55, 55, 255) : accent);
        dl->AddCircle(ImVec2(bx, by), 10.0f, IM_COL32(255, 255, 255, 220), 24, 1.5f);
    } else if (env_kind == 9) { // Inverted Double CartPole
        float ty = p0.y + sz.y * 0.80f;
        float tl = p0.x + 24.0f, tr = p1.x - 24.0f;
        dl->AddLine(ImVec2(tl, ty), ImVec2(tr, ty), IM_COL32(110, 125, 150, 255), 2.5f);
        float nx = std::clamp((f.s[0] + 2.4f) / 4.8f, 0.0f, 1.0f);
        float cx = tl + nx * (tr - tl);
        float cw = 46.0f, ch = 20.0f;
        dl->AddRectFilled(ImVec2(cx - cw * 0.5f, ty - ch), ImVec2(cx + cw * 0.5f, ty),
                          f.terminated ? IM_COL32(210, 55, 55, 255) : accent, 4.0f);
        float plen = std::max(28.0f, sz.y * 0.30f);
        ImVec2 piv(cx, ty - ch + 2.0f);
        ImVec2 mid(piv.x + plen * std::sin(f.s[1]), piv.y - plen * std::cos(f.s[1]));
        ImVec2 tip(mid.x + plen * std::sin(f.s[2]), mid.y - plen * std::cos(f.s[2]));
        dl->AddLine(piv, mid, IM_COL32(250, 204, 21, 255), 5.5f);
        dl->AddLine(mid, tip, IM_COL32(74, 222, 128, 255), 4.5f);
        dl->AddCircleFilled(piv, 4.0f, IM_COL32(255, 255, 255, 255));
        dl->AddCircleFilled(mid, 4.0f, IM_COL32(255, 255, 255, 255));
    }
}

/**
 * @brief Draws the 5x5 Taxi-v3 grid with colored landmarks (R, G, Y, B), walls, passenger, and Taxi!
 */
void DrawTaxiCanvas(ImDrawList* dl, ImVec2 p0, ImVec2 sz, const std::vector<studio::TaxiLab::TaxiState>& path,
                    int anim_step, ImU32 accent) {
    ImVec2 p1(p0.x + sz.x, p0.y + sz.y);
    dl->AddRectFilled(p0, p1, IM_COL32(22, 26, 36, 255), 6.0f);
    dl->AddRect(p0, p1, accent, 6.0f, 0, 1.5f);

    float pad = 12.0f;
    float cs = std::min((sz.x - 2 * pad) / 5.0f, (sz.y - 2 * pad) / 5.0f);
    float ox = p0.x + (sz.x - cs * 5.0f) * 0.5f;
    float oy = p0.y + (sz.y - cs * 5.0f) * 0.5f;

    studio::TaxiLab::TaxiState st;
    if (!path.empty()) st = path[std::clamp(anim_step, 0, (int)path.size() - 1)];

    const ImU32 lcols[4] = {IM_COL32(190, 55, 55, 200), IM_COL32(45, 165, 95, 200),
                            IM_COL32(215, 175, 40, 200), IM_COL32(55, 125, 220, 200)};
    const char* lnames[4] = {"R", "G", "Y", "B"};

    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            ImVec2 c0(ox + x * cs + 2, oy + y * cs + 2);
            ImVec2 c1(ox + (x + 1) * cs - 2, oy + (y + 1) * cs - 2);
            ImU32 bg = IM_COL32(42, 48, 62, 255);
            for (int l = 0; l < 4; ++l) {
                if (studio::TaxiLab::LANDMARKS[l].first == x && studio::TaxiLab::LANDMARKS[l].second == y) {
                    bg = lcols[l];
                }
            }
            dl->AddRectFilled(c0, c1, bg, 4.0f);
            for (int l = 0; l < 4; ++l) {
                if (studio::TaxiLab::LANDMARKS[l].first == x && studio::TaxiLab::LANDMARKS[l].second == y) {
                    dl->AddText(ImVec2(c0.x + 5, c0.y + 3), IM_COL32(255, 255, 255, 230), lnames[l]);
                    if (st.dest == l) {
                        dl->AddRect(c0, c1, IM_COL32(255, 255, 255, 255), 4.0f, 0, 2.5f);
                    }
                }
            }
        }
    }

    // Internal Taxi-v3 walls
    auto draw_wall = [&](int col_left, int row) {
        float wx = ox + (col_left + 1) * cs;
        dl->AddLine(ImVec2(wx, oy + row * cs), ImVec2(wx, oy + (row + 1) * cs), IM_COL32(245, 245, 250, 240), 5.0f);
    };
    draw_wall(1, 0); draw_wall(1, 1);
    draw_wall(0, 3); draw_wall(0, 4);
    draw_wall(2, 3); draw_wall(2, 4);

    // Passenger indicator (if not yet inside taxi)
    if (st.pass_loc < 4) {
        auto [px, py] = studio::TaxiLab::LANDMARKS[st.pass_loc];
        ImVec2 pc(ox + (px + 0.5f) * cs, oy + (py + 0.5f) * cs);
        dl->AddCircleFilled(pc, cs * 0.22f, IM_COL32(255, 255, 255, 255));
        dl->AddText(ImVec2(pc.x - 4, pc.y - 7), IM_COL32(20, 25, 35, 255), "P");
    }

    // Taxi cab (turns bright yellow-green when carrying passenger!)
    ImVec2 tc(ox + (st.x + 0.5f) * cs, oy + (st.y + 0.5f) * cs);
    ImU32 cab_col = (st.pass_loc == 4) ? IM_COL32(74, 222, 128, 255) : IM_COL32(250, 204, 21, 255);
    dl->AddRectFilled(ImVec2(tc.x - cs * 0.30f, tc.y - cs * 0.22f),
                      ImVec2(tc.x + cs * 0.30f, tc.y + cs * 0.22f), cab_col, 6.0f);
    dl->AddRect(ImVec2(tc.x - cs * 0.30f, tc.y - cs * 0.22f),
                ImVec2(tc.x + cs * 0.30f, tc.y + cs * 0.22f), accent, 6.0f, 0, 2.5f);
    const char* tlbl = (st.pass_loc == 4) ? "TAXI+P" : "TAXI";
    ImVec2 tsz = ImGui::CalcTextSize(tlbl);
    dl->AddText(ImVec2(tc.x - tsz.x * 0.5f, tc.y - tsz.y * 0.5f), IM_COL32(15, 20, 30, 255), tlbl);
}

/**
 * @brief Draws a cute vector Robot sprite centered at `c` with radius `r`.
 */
void DrawRobotSprite(ImDrawList* dl, ImVec2 c, float r, ImU32 accent) {
    // Antenna
    dl->AddLine(ImVec2(c.x, c.y - r * 0.75f), ImVec2(c.x, c.y - r * 1.15f), IM_COL32(220, 230, 245, 255), 2.0f);
    dl->AddCircleFilled(ImVec2(c.x, c.y - r * 1.18f), r * 0.16f, IM_COL32(250, 204, 21, 255));
    // Robot Head (rounded metal box)
    ImVec2 h0(c.x - r * 0.78f, c.y - r * 0.72f);
    ImVec2 h1(c.x + r * 0.78f, c.y + r * 0.65f);
    dl->AddRectFilled(h0, h1, accent, r * 0.28f);
    dl->AddRect(h0, h1, IM_COL32(255, 255, 255, 230), r * 0.28f, 0, 1.8f);
    // Dark visor
    dl->AddRectFilled(ImVec2(c.x - r * 0.56f, c.y - r * 0.42f),
                      ImVec2(c.x + r * 0.56f, c.y + r * 0.08f), IM_COL32(18, 22, 32, 255), r * 0.16f);
    // Glowing eyes
    dl->AddCircleFilled(ImVec2(c.x - r * 0.26f, c.y - r * 0.17f), r * 0.14f, IM_COL32(125, 245, 255, 255));
    dl->AddCircleFilled(ImVec2(c.x + r * 0.26f, c.y - r * 0.17f), r * 0.14f, IM_COL32(125, 245, 255, 255));
    // Cute little smile line
    dl->AddLine(ImVec2(c.x - r * 0.28f, c.y + r * 0.34f), ImVec2(c.x + r * 0.28f, c.y + r * 0.34f),
                IM_COL32(20, 28, 40, 255), 2.0f);
}

/**
 * @brief Draws an animated flickering Fire Flame sprite for Trap cells!
 */
void DrawFireSprite(ImDrawList* dl, ImVec2 c, float r, float time_sec) {
    float flicker = std::sin(time_sec * 12.0f + c.x * 0.1f) * r * 0.12f;
    // Outer red-orange flame
    dl->AddTriangleFilled(ImVec2(c.x, c.y - r * 0.95f + flicker),
                          ImVec2(c.x - r * 0.68f, c.y + r * 0.58f),
                          ImVec2(c.x + r * 0.68f, c.y + r * 0.58f),
                          IM_COL32(235, 65, 35, 240));
    dl->AddCircleFilled(ImVec2(c.x, c.y + r * 0.32f), r * 0.52f, IM_COL32(235, 65, 35, 240));
    // Inner bright orange flame
    dl->AddTriangleFilled(ImVec2(c.x + flicker * 0.5f, c.y - r * 0.62f - flicker * 0.5f),
                          ImVec2(c.x - r * 0.44f, c.y + r * 0.52f),
                          ImVec2(c.x + r * 0.44f, c.y + r * 0.52f),
                          IM_COL32(255, 145, 25, 250));
    // Hot yellow core
    dl->AddCircleFilled(ImVec2(c.x, c.y + r * 0.32f), r * 0.26f, IM_COL32(255, 235, 90, 255));
}

/**
 * @brief Draws a glowing Flag/Trophy icon for the Goal cell!
 */
void DrawGoalFlag(ImDrawList* dl, ImVec2 c, float r) {
    dl->AddLine(ImVec2(c.x - r * 0.35f, c.y + r * 0.65f), ImVec2(c.x - r * 0.35f, c.y - r * 0.72f),
                IM_COL32(240, 245, 255, 255), 2.5f);
    dl->AddTriangleFilled(ImVec2(c.x - r * 0.35f, c.y - r * 0.72f),
                          ImVec2(c.x + r * 0.65f, c.y - r * 0.32f),
                          ImVec2(c.x - r * 0.35f, c.y + r * 0.08f),
                          IM_COL32(74, 222, 128, 255));
}

/**
 * @brief Draws a Tabular GridWorld heatmap + greedy policy arrows + flickering Fire traps + moving Robot
 *        + Wind column badges + Interactive Click-to-Paint + Hover 4-action Q-value Inspector!
 */
template <typename AgentT>
void DrawGridWorldCanvas(ImDrawList* dl, ImVec2 p0, ImVec2 sz, const rl::GridWorld& env,
                         const AgentT& agent, int episodes,
                         const std::vector<std::pair<int, int>>& path, int anim_step,
                         float time_sec, ImU32 accent,
                         const studio::ValueIterationSolver* dp_opt,
                         int& out_clicked_x, int& out_clicked_y) {
    out_clicked_x = -1;
    out_clicked_y = -1;

    ImVec2 p1(p0.x + sz.x, p0.y + sz.y);
    dl->AddRectFilled(p0, p1, IM_COL32(22, 26, 36, 255), 6.0f);
    dl->AddRect(p0, p1, accent, 6.0f, 0, 1.5f);

    float pad = 10.0f;
    float cs = std::min((sz.x - 2 * pad) / env.width(), (sz.y - 2 * pad) / env.height());
    float ox = p0.x + (sz.x - cs * env.width()) * 0.5f;
    float oy = p0.y + (sz.y - cs * env.height()) * 0.5f;

    int rob_x = 0, rob_y = 0;
    if (!path.empty()) {
        int idx = std::clamp(anim_step, 0, static_cast<int>(path.size()) - 1);
        rob_x = path[idx].first;
        rob_y = path[idx].second;
    }

    ImVec2 mpos = ImGui::GetMousePos();
    bool mouse_in_grid = (mpos.x >= ox && mpos.x < ox + cs * env.width() &&
                          mpos.y >= oy && mpos.y < oy + cs * env.height());
    int hov_x = mouse_in_grid ? static_cast<int>((mpos.x - ox) / cs) : -1;
    int hov_y = mouse_in_grid ? static_cast<int>((mpos.y - oy) / cs) : -1;

    if (mouse_in_grid && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        out_clicked_x = hov_x;
        out_clicked_y = hov_y;
    }

    const char* arrows[4] = {"^", ">", "v", "<"};
    for (int y = 0; y < env.height(); ++y) {
        for (int x = 0; x < env.width(); ++x) {
            int s = y * env.width() + x;
            int ba = dp_opt ? dp_opt->policy[s] : static_cast<int>(agent.greedy_action(s));
            double mq = dp_opt ? dp_opt->V[s] : agent.q_value(s, ba);
            auto ct = env.cell_at(x, y);
            bool has_vals = dp_opt ? (dp_opt->iterations > 0) : (episodes > 0);

            ImU32 col = IM_COL32(45, 52, 68, 255);
            if (ct == rl::CellType::Goal) col = IM_COL32(24, 92, 58, 255);
            else if (ct == rl::CellType::Trap) col = IM_COL32(85, 28, 28, 255);
            else if (ct == rl::CellType::Wall) col = IM_COL32(16, 18, 24, 255);
            else if (has_vals) {
                double n = std::clamp(mq / 10.0, -1.0, 1.0);
                col = (n >= 0.0) ? IM_COL32(38, static_cast<int>(52 + n * 85), static_cast<int>(68 + n * 25), 255)
                                 : IM_COL32(static_cast<int>(45 - n * 95), 45, 55, 255);
            }

            ImVec2 c0(ox + x * cs + 2, oy + y * cs + 2);
            ImVec2 c1(ox + (x + 1) * cs - 2, oy + (y + 1) * cs - 2);
            ImVec2 center((c0.x + c1.x) * 0.5f, (c0.y + c1.y) * 0.5f);
            dl->AddRectFilled(c0, c1, col, 4.0f);

            // Upward Wind tint/indicator for Windy GridWorld columns
            int wind = env.wind_at(x);
            if (wind > 0 && y == env.height() - 1) {
                char wbuf[12];
                std::snprintf(wbuf, sizeof(wbuf), "W%d^", wind);
                dl->AddText(ImVec2(c0.x + 3, c0.y + 2), IM_COL32(140, 210, 255, 200), wbuf);
            }

            if (ct == rl::CellType::Goal) {
                DrawGoalFlag(dl, center, cs * 0.34f);
            } else if (ct == rl::CellType::Trap) {
                DrawFireSprite(dl, center, cs * 0.34f, time_sec);
            } else if (ct == rl::CellType::Wall) {
                dl->AddRectFilled(ImVec2(c0.x + 4, c0.y + 4), ImVec2(c1.x - 4, c1.y - 4), IM_COL32(30, 34, 44, 255), 3.0f);
            } else {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%s %.1f", arrows[ba], mq);
                ImVec2 tsz = ImGui::CalcTextSize(buf);
                dl->AddText(ImVec2(center.x - tsz.x * 0.5f, c1.y - tsz.y - 3.0f),
                            IM_COL32(225, 232, 245, 220), buf);
            }

            if (x == rob_x && y == rob_y && !dp_opt) {
                DrawRobotSprite(dl, ImVec2(center.x, center.y - cs * 0.05f), cs * 0.28f, accent);
            }

            // Hover highlight + 4-action Q-value tooltip inspector!
            if (x == hov_x && y == hov_y) {
                dl->AddRect(c0, c1, IM_COL32(255, 255, 255, 220), 4.0f, 0, 2.0f);
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("Cell (%d, %d) | State ID: %d | Wind: +%d", x, y, s, wind);
                    ImGui::Separator();
                    if (dp_opt) {
                        ImGui::Text("Exact Value Iteration V*(s) = %.3f", dp_opt->V[s]);
                        ImGui::Text("Optimal Policy: %s", arrows[dp_opt->policy[s]]);
                    } else {
                        const char* anames[4] = {"Up   (^)", "Right(>)", "Down (v)", "Left (<)"};
                        for (int a = 0; a < 4; ++a) {
                            double qv = agent.q_value(s, a);
                            ImGui::Text("%s: %6.2f", anames[a], qv);
                            ImGui::SameLine();
                            float frac = static_cast<float>(std::clamp((qv + 10.0) / 20.0, 0.0, 1.0));
                            ImGui::ProgressBar(frac, ImVec2(90, 12), "");
                        }
                    }
                    ImGui::TextDisabled("Click to paint selected brush!");
                    ImGui::EndTooltip();
                }
            }
        }
    }
}

void RandomizeContinuousGrid2D(int& seed) {
    seed += 23;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pd(-0.6f, 0.6f);
    std::uniform_real_distribution<float> rd(0.16f, 0.25f);
    int c = std::uniform_int_distribution<int>(0, 3)(rng);
    auto& m = rlt::rl::environments::continuous_grid_2d::active_map();
    m.start_x = (c == 0 || c == 2) ? -0.8f : 0.8f;
    m.start_y = (c == 0 || c == 1) ? -0.8f : 0.8f;
    m.target_x = -m.start_x;
    m.target_y = -m.start_y;
    m.num_traps = std::uniform_int_distribution<int>(2, 4)(rng);
    for (int i = 0; i < m.num_traps; ++i) {
        for (int a = 0; a < 50; ++a) {
            float tx = pd(rng), ty = pd(rng), tr = rd(rng);
            if (std::hypot(tx - m.start_x, ty - m.start_y) > tr + 0.35f &&
                std::hypot(tx - m.target_x, ty - m.target_y) > tr + 0.35f) {
                m.trap_x[i] = tx; m.trap_y[i] = ty; m.trap_r[i] = tr; break;
            }
        }
    }
}

} // namespace

int main() {
    // 1. Create Win32 Window + DirectX11 Device
    WNDCLASSEXW wc{sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr),
                   nullptr, nullptr, nullptr, nullptr, L"RLStudioClass", nullptr};
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"C++ RL Studio — Interactive Algorithm & Hyperparameter Lab (Dear ImGui + ImPlot + RLtools)",
                              WS_OVERLAPPEDWINDOW, 60, 40, 1520, 920, nullptr, nullptr, wc.hInstance, nullptr);
    if (!CreateDeviceD3D(hwnd)) { CleanupDeviceD3D(); UnregisterClassW(wc.lpszClassName, wc.hInstance); return 1; }
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    // 2. Initialize Dear ImGui + ImPlot with unified color palette (Cyan, Magenta, Amber, Emerald)!
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;

    // Match ImPlot curve colors 1-to-1 with the top viewport accents:
    // 0: Cyan (SAC / Q-Learning), 1: Magenta (PPO / SARSA), 2: Amber (TD3), 3: Green (Target Threshold)
    static const ImVec4 kStudioPalette[8] = {
        ImVec4(0.22f, 0.74f, 0.97f, 1.0f), // Cyan    -> SAC / Q-Learning
        ImVec4(0.91f, 0.47f, 0.98f, 1.0f), // Magenta -> PPO / SARSA
        ImVec4(0.98f, 0.57f, 0.24f, 1.0f), // Amber   -> TD3
        ImVec4(0.29f, 0.87f, 0.50f, 1.0f), // Green   -> Target Threshold
        ImVec4(0.98f, 0.80f, 0.08f, 1.0f), // Yellow  -> Snapshot 1
        ImVec4(0.97f, 0.44f, 0.44f, 1.0f), // Coral   -> Snapshot 2
        ImVec4(0.58f, 0.64f, 0.72f, 1.0f), // Slate   -> Snapshot 3
        ImVec4(0.95f, 0.95f, 0.98f, 1.0f)  // White   -> Snapshot 4
    };
    ImPlotColormap studio_cmap = ImPlot::AddColormap("StudioPalette", kStudioPalette, 8);
    ImPlot::GetStyle().Colormap = studio_cmap;

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // 3. Initialize All 13 Labs (Bandits + GridWorld + Taxi-v3 + 10 RLtools Continuous Suites!)
    studio::BanditsLab bandits_lab;
    studio::GridWorldStudio grid_studio;
    studio::TaxiLab taxi_lab;
    auto cont_suites = studio::create_all_continuous_suites();
    studio::ContinuousHyperparams hp;
    cont_suites[0]->reset(hp);

    const char* env_names[] = {
        "0. Multi-Armed Bandits (Eps-Greedy vs. UCB1 vs. Thompson)",
        "1. GridWorld Lab (Q-Learning, SARSA, Windy, Slip & Value Iter)",
        "2. Taxi-v3 (Hierarchical Pickup/Dropoff: Q-Learning vs. SARSA)",
        "3. CartPole-Continuous (SAC / PPO / TD3)",
        "4. Pendulum-v1 Swing-Up (RLtools Built-in)",
        "5. Acrobot 2-Link Swing-Up (RLtools Built-in)",
        "6. MountainCar-Continuous (SAC / PPO / TD3)",
        "7. Drone2D Quadrotor Hover (SAC / PPO / TD3)",
        "8. ContinuousGrid2D Trap Arena (SAC / PPO / TD3)",
        "9. LunarLander / Moon Landing (SAC / PPO / TD3)",
        "10. Reacher-2DOF Robotic Arm (SAC / PPO / TD3)",
        "11. Ball-and-Beam Control (SAC / PPO / TD3)",
        "12. Inverted Double CartPole (SAC / PPO / TD3)"
    };
    int selected_env = 3; // Start on CartPole-Continuous
    int cont_map_seed = 77;
    int heatmap_algo = 1; // Default PPO for Neural Policy Heatmap
    std::vector<double> cached_heatmap;
    int cached_heatmap_env = -1;
    int cached_heatmap_steps = -1;
    int cached_heatmap_algo = -1;

    bool live_training = false;
    bool race_to_target = false; // "Train Until Target / Budget" benchmark mode
    float target_return = cont_suites[0]->default_target_return();
    int max_budget_rounds = 120;
    int race_rounds_run = 0;

    bool playing_anim = false; // Start paused until the user clicks Play or drags the Frame scrubber!
    bool loop_anim = false;
    float anim_accum = 0.0f;
    float anim_speed = 1.0f;
    int anim_frame = 0;
    int x_axis_mode = 0; // 0 = Rounds, 1 = Env Steps, 2 = Wall Time (s)
    int sac_steps_per_tick = 150;
    int ppo_iters_per_tick = 1;
    int td3_steps_per_tick = 150;
    std::vector<studio::SavedRun> saved_runs;

    // 4. Main 60 FPS Render & Training Loop
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Advance live training or "Train Until Target / Budget" race
        if (live_training || race_to_target) {
            if (selected_env == 0) {
                bandits_lab.step_n(20);
                if (race_to_target) {
                    race_rounds_run += 20;
                    if (race_rounds_run >= max_budget_rounds * 20) race_to_target = false;
                }
            } else if (selected_env == 1) {
                grid_studio.train_episodes(10);
                if (race_to_target) {
                    race_rounds_run += 10;
                    bool q_ok = !grid_studio.q_ret.empty() && grid_studio.q_ret.back() >= target_return;
                    bool s_ok = !grid_studio.s_ret.empty() && grid_studio.s_ret.back() >= target_return;
                    if ((q_ok && s_ok) || race_rounds_run >= max_budget_rounds * 10) {
                        race_to_target = false;
                    }
                }
            } else if (selected_env == 2) {
                taxi_lab.train_episodes(25);
                if (race_to_target) {
                    race_rounds_run += 25;
                    bool q_ok = !taxi_lab.ret_q.empty() && taxi_lab.ret_q.back() >= target_return;
                    bool s_ok = !taxi_lab.ret_s.empty() && taxi_lab.ret_s.back() >= target_return;
                    if ((q_ok && s_ok) || race_rounds_run >= max_budget_rounds * 25) race_to_target = false;
                }
            } else {
                auto& suite = *cont_suites[selected_env - 3];
                suite.train_step(sac_steps_per_tick, ppo_iters_per_tick, td3_steps_per_tick, hp);
                if (race_to_target) {
                    suite.check_threshold(target_return);
                    race_rounds_run++;
                    bool all_done = true;
                    for (int k = 0; k < 3; ++k) {
                        if (suite.telemetry(k).enabled && !suite.telemetry(k).solved) {
                            all_done = false;
                        }
                    }
                    if (all_done || race_rounds_run >= max_budget_rounds) {
                        race_to_target = false;
                    }
                }
            }
        }

        // Advance playback animation for BOTH GridWorld (5 Hz Robot walk) and Continuous envs (50 Hz)
        if (playing_anim && selected_env >= 1) {
            int max_f = 1;
            float base_fps = 50.0f;
            if (selected_env == 1) {
                max_f = std::max((int)grid_studio.q_path.size(), (int)grid_studio.s_path.size());
                base_fps = 5.0f; // Crisp 5 tiles/sec robot walk!
            } else if (selected_env == 2) {
                max_f = std::max((int)taxi_lab.path_q.size(), (int)taxi_lab.path_s.size());
                base_fps = 4.0f;
            } else {
                const auto& suite = *cont_suites[selected_env - 3];
                max_f = std::max({(int)suite.telemetry(0).rollout.size(),
                                  (int)suite.telemetry(1).rollout.size(),
                                  (int)suite.telemetry(2).rollout.size()});
            }
            if (max_f > 1) {
                anim_accum += ImGui::GetIO().DeltaTime * base_fps * anim_speed;
                if (anim_accum >= 1.0f) {
                    int next_f = anim_frame + static_cast<int>(anim_accum);
                    anim_accum -= std::floor(anim_accum);
                    if (next_f >= max_f) {
                        if (loop_anim) {
                            anim_frame = next_f % max_f;
                        } else {
                            anim_frame = max_f - 1;
                            playing_anim = false;
                        }
                    } else {
                        anim_frame = next_f;
                    }
                }
            }
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Full-viewport Studio Layout
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGuiWindowFlags root_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
        ImGui::Begin("RLStudioRoot", nullptr, root_flags);

        // ====================================================================
        // LEFT SIDEBAR: Environment Picker, Hyperparameters & Algorithm Lab
        // ====================================================================
        ImGui::BeginChild("LeftControls", ImVec2(360, 0), true);
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "C++ RL STUDIO");
        ImGui::Separator();

        ImGui::Text("1. Environment Selection");
        if (ImGui::Combo("##EnvCombo", &selected_env, env_names, IM_ARRAYSIZE(env_names))) {
            live_training = false;
            race_to_target = false;
            playing_anim = false;
            anim_frame = 0;
            saved_runs.clear();
            cached_heatmap_env = -1;
            if (selected_env == 0) {
                target_return = 1.0f;
                max_budget_rounds = 100;
            } else if (selected_env == 1) {
                target_return = 8.0f;
                max_budget_rounds = 100;
            } else if (selected_env == 2) {
                target_return = 6.0f;
                max_budget_rounds = 120;
            } else {
                target_return = cont_suites[selected_env - 3]->default_target_return();
                max_budget_rounds = 120;
                if (cont_suites[selected_env - 3]->telemetry(0).y_return.empty()) {
                    cont_suites[selected_env - 3]->reset(hp);
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("2. Training Controls");

        if (ImGui::Button(live_training ? "PAUSE Live Learning" : "START Live Learning", ImVec2(-1, 32))) {
            race_to_target = false;
            live_training = !live_training;
        }

        // Train-Until-Threshold / Budget Benchmark Box (for GridWorld & Continuous Suites)
        if (selected_env >= 1) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.65f, 1.0f), "Benchmark: Train Until Threshold");
            ImGui::SetNextItemWidth(110);
            ImGui::InputFloat("Target Return", &target_return, 10.0f, 50.0f, "%.1f");
            HelpMarker("Each algorithm stops as soon as its evaluation return reaches Target Return, recording exact steps & time!");
            ImGui::SetNextItemWidth(110);
            ImGui::InputInt("Max Rounds", &max_budget_rounds, 10, 50);
            max_budget_rounds = std::clamp(max_budget_rounds, 5, 2000);
            if (ImGui::Button(race_to_target ? "STOP Threshold Race" : "RUN Benchmark (Reset & Race!)", ImVec2(-1, 30))) {
                if (race_to_target) {
                    race_to_target = false;
                } else {
                    live_training = false;
                    race_rounds_run = 0;
                    if (selected_env == 1) grid_studio.reset_agents();
                    else if (selected_env == 2) taxi_lab.reset();
                    else cont_suites[selected_env - 3]->reset(hp);
                    race_to_target = true;
                }
            }
            if (race_to_target) {
                ImGui::ProgressBar(static_cast<float>(race_rounds_run) / static_cast<float>(max_budget_rounds), ImVec2(-1, 14));
            }
        }

        if (selected_env == 0) {
            if (ImGui::Button("Pull +200 Steps", ImVec2(-1, 28))) bandits_lab.step_n(200);
            if (ImGui::Button("Reset 10-Armed Bandits", ImVec2(-1, 26))) { live_training = false; bandits_lab.reset(); }
            ImGui::Separator();
            ImGui::Text("3. Bandit Exploration Params");
            ImGui::SliderFloat("Eps-Greedy (eps)", &bandits_lab.epsilon, 0.0f, 0.50f, "%.2f");
            HelpMarker("Probability of pulling a random arm. eps=0 is purely greedy (can get stuck on a suboptimal arm!).");
            ImGui::SliderFloat("UCB1 Bonus (c)", &bandits_lab.ucb_c, 0.1f, 5.0f, "%.2f");
            HelpMarker("Upper Confidence Bound exploration weight c in Q(a) + c*sqrt(ln(t)/N(a)).");
            ImGui::SliderFloat("Reward Noise Std", &bandits_lab.reward_std, 0.1f, 3.0f, "%.2f");
            if (ImGui::InputInt("Bandit Seed", &bandits_lab.seed)) {
                bandits_lab.seed = std::max(0, bandits_lab.seed);
                bandits_lab.reset();
            }
            ImGui::Separator();
            ImGui::TextWrapped("Total Pulls: %d | Best Arm: #%d (mean %.2f)",
                               bandits_lab.total_steps, bandits_lab.optimal_arm, bandits_lab.true_means[bandits_lab.optimal_arm]);
        } else if (selected_env == 1) {
            if (ImGui::Button("Train +100 Episodes", ImVec2(-1, 28))) grid_studio.train_episodes(100);
            if (ImGui::Button("Reset Q-Tables & Plots", ImVec2(-1, 26))) grid_studio.reset_agents();
            ImGui::Separator();
            ImGui::Text("Map Presets & Live Editor");
            if (ImGui::Button("Cliff (8x4)", ImVec2(106, 26))) grid_studio.load_cliff();
            ImGui::SameLine();
            if (ImGui::Button("Windy (10x7)", ImVec2(106, 26))) grid_studio.load_windy();
            ImGui::SameLine();
            if (ImGui::Button("Random", ImVec2(-1, 26))) grid_studio.load_random_map();

            ImGui::Text("Click-to-Paint Brush:");
            ImGui::RadioButton("Empty", &grid_studio.paint_brush, 0); ImGui::SameLine();
            ImGui::RadioButton("Wall", &grid_studio.paint_brush, 1); ImGui::SameLine();
            ImGui::RadioButton("Fire Trap", &grid_studio.paint_brush, 2);
            HelpMarker("Click ANY cell in the grid (even mid-training!) to add/remove Walls or Fire Traps and watch agents adapt!");

            ImGui::Separator();
            ImGui::Text("3. Tabular & Stochastic Params");
            bool changed = false;
            changed |= ImGui::SliderFloat("Learning Rate (alpha)", &grid_studio.alpha, 0.02f, 0.80f, "%.2f");
            changed |= ImGui::SliderFloat("Discount (gamma)", &grid_studio.gamma, 0.50f, 0.99f, "%.2f");
            changed |= ImGui::SliderFloat("Initial Epsilon", &grid_studio.init_epsilon, 0.01f, 0.60f, "%.2f");
            if (ImGui::SliderFloat("Slip Prob (Ice!)", &grid_studio.slip_prob, 0.0f, 0.40f, "%.2f")) {
                grid_studio.env_q.set_slip_prob(grid_studio.slip_prob);
                grid_studio.env_s.set_slip_prob(grid_studio.slip_prob);
                grid_studio.dp_solver.reset(grid_studio.env_q);
            }
            HelpMarker("FrozenLake stochasticity! Probability the robot slips sideways perpendicular to its intended move.");
            ImGui::Checkbox("GLIE Epsilon Decay", &grid_studio.use_glie_decay);
            if (grid_studio.use_glie_decay) {
                ImGui::SliderFloat("Decay / Ep", &grid_studio.decay_rate, 0.980f, 0.999f, "%.3f");
            }
            if (changed) grid_studio.reset_agents();

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.98f, 0.80f, 0.35f, 1.0f), "4. Dynamic Programming (Value Iter)");
            ImGui::Checkbox("Show Exact Value Iteration", &grid_studio.show_dp_overlay);
            HelpMarker("Replaces the right panel with Model-Based Exact Value Iteration V*(s) so you can step 1 Bellman sweep at a time!");
            if (grid_studio.show_dp_overlay) {
                if (ImGui::Button("+1 Bellman Sweep", ImVec2(165, 24))) grid_studio.dp_solver.sweep(grid_studio.env_q, grid_studio.gamma, 1);
                ImGui::SameLine();
                if (ImGui::Button("Solve (50 Sweeps)", ImVec2(-1, 24))) grid_studio.dp_solver.sweep(grid_studio.env_q, grid_studio.gamma, 50);
                ImGui::TextDisabled("Sweeps: %d | Max Delta: %.4f", grid_studio.dp_solver.iterations, grid_studio.dp_solver.max_delta);
            }

            ImGui::Separator();
            ImGui::Text("5. Robot Playback");
            if (ImGui::Button(playing_anim ? "Pause Robots" : "Watch Robots Race!", ImVec2(165, 26))) {
                if (!playing_anim) {
                    grid_studio.record_greedy_paths();
                    anim_frame = 0;
                    anim_accum = 0.0f;
                }
                playing_anim = !playing_anim;
            }
            ImGui::SameLine();
            ImGui::Checkbox("Loop", &loop_anim);
            int max_gf = std::max({1, (int)grid_studio.q_path.size() - 1, (int)grid_studio.s_path.size() - 1});
            if (ImGui::SliderInt("Step", &anim_frame, 0, max_gf)) playing_anim = false;
        } else if (selected_env == 2) {
            if (ImGui::Button("Train +500 Episodes", ImVec2(-1, 28))) taxi_lab.train_episodes(500);
            if (ImGui::Button("Reset Taxi Q-Tables", ImVec2(-1, 26))) taxi_lab.reset();
            ImGui::Separator();
            ImGui::Text("3. Taxi-v3 Hyperparameters");
            ImGui::SliderFloat("Alpha", &taxi_lab.alpha, 0.05f, 0.80f, "%.2f");
            ImGui::SliderFloat("Gamma", &taxi_lab.gamma, 0.70f, 0.99f, "%.2f");
            ImGui::SliderFloat("Epsilon", &taxi_lab.epsilon, 0.02f, 0.50f, "%.2f");
            ImGui::Separator();
            ImGui::Text("4. Taxi Demo Playback");
            if (ImGui::Button(playing_anim ? "Pause Taxis" : "Watch Taxis Deliver!", ImVec2(165, 26))) {
                if (!playing_anim) { taxi_lab.record_demo_paths(); anim_frame = 0; anim_accum = 0.0f; }
                playing_anim = !playing_anim;
            }
            ImGui::SameLine();
            ImGui::Checkbox("Loop", &loop_anim);
            int max_tf = std::max({1, (int)taxi_lab.path_q.size() - 1, (int)taxi_lab.path_s.size() - 1});
            if (ImGui::SliderInt("Step", &anim_frame, 0, max_tf)) playing_anim = false;
            ImGui::TextWrapped("White border = Destination. Taxi turns green after picking up Passenger (P)!");
        } else {
            auto& suite = *cont_suites[selected_env - 3];
            if (ImGui::Button("Train +1 Big Round", ImVec2(-1, 28))) {
                suite.train_step(500, 4, 500, hp);
            }
            if (ImGui::Button("Reset Neural Networks", ImVec2(-1, 26))) {
                live_training = false;
                suite.reset(hp);
            }
            if (suite.env_kind() == 5) {
                if (ImGui::Button("Randomize 2D Trap Map", ImVec2(-1, 26))) {
                    live_training = false;
                    RandomizeContinuousGrid2D(cont_map_seed);
                    suite.reset(hp);
                }
            }

            ImGui::Separator();
            ImGui::Text("3. Active Algorithms (Toggle to Isolate)");
            ImGui::Checkbox("SAC (Off-Policy Stochastic)", &suite.telemetry(0).enabled);
            HelpMarker("Soft Actor-Critic: Maximizes reward + entropy using a Replay Buffer. High sample efficiency!");
            ImGui::Checkbox("PPO (On-Policy Stochastic)", &suite.telemetry(1).enabled);
            HelpMarker("Proximal Policy Optimization: Collects fresh parallel rollouts (4 envs) and clips policy updates. Blazing fast wall-clock!");
            ImGui::Checkbox("TD3 (Off-Policy Deterministic)", &suite.telemetry(2).enabled);
            HelpMarker("Twin Delayed DDPG: Deterministic actor + twin critics to prevent Q-overestimation bias.");

            ImGui::Separator();
            ImGui::Text("4. Live Hyperparameters (RLtools)");
            if (ImGui::SliderFloat("Adam LR (alpha)", &hp.lr, 0.0001f, 0.01f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
                suite.apply_hyperparams(hp);
            }
            HelpMarker("Live Adam learning rate for Actor & Critic networks! Try 0.0003 (stable) vs 0.005 (fast/unstable) and snapshot runs to compare.");
            ImGui::SliderInt("SAC Steps/Tick", &sac_steps_per_tick, 50, 500);
            ImGui::SliderInt("PPO Iters/Tick", &ppo_iters_per_tick, 1, 5);
            ImGui::SliderInt("TD3 Steps/Tick", &td3_steps_per_tick, 50, 500);
            if (ImGui::InputInt("Random Seed", &hp.seed)) {
                hp.seed = std::max(0, hp.seed);
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.98f, 0.80f, 0.35f, 1.0f), "5. Reward Shaping Lab");
            auto& rs = studio::reward_shaping();
            int ek = suite.env_kind();
            if (ek == 0) { // CartPole
                ImGui::SliderFloat("Center Penalty |x|", &rs.cartpole_center_penalty, 0.0f, 0.5f, "%.2f");
                HelpMarker("Penalizes drifting away from x=0. Try 0.25 to force the cart to stay centered!");
                ImGui::SliderFloat("Angle Penalty |th|", &rs.cartpole_angle_penalty, 0.0f, 2.0f, "%.2f");
                ImGui::SliderFloat("Effort Penalty u^2", &rs.cartpole_effort_penalty, 0.0f, 0.2f, "%.3f");
            } else if (ek == 1) {
                ImGui::TextWrapped("Pendulum-v1 uses RLtools' built-in reward: -(theta^2 + 0.1*vel^2 + 0.001*u^2).");
            } else if (ek == 2) { // Acrobot
                ImGui::Checkbox("Velocity Penalty (Anti-Spin!)", &rs.acrobot_vel_penalty_on);
                HelpMarker("Penalizes fast joint angular velocities so the arm learns to balance upright instead of spinning wildly!");
                if (rs.acrobot_vel_penalty_on) {
                    ImGui::SliderFloat("Vel Weight", &rs.acrobot_vel_weight, 0.001f, 0.10f, "%.3f");
                }
                ImGui::SliderFloat("Torque Weight", &rs.acrobot_torque_weight, 0.0f, 0.10f, "%.3f");
            } else if (ek == 3) { // MountainCar
                ImGui::Checkbox("Dense Height/Vel Shaping", &rs.mcar_dense_shaping);
                HelpMarker("Uncheck for SPARSE reward (+100 at flag only) to see how hard exploration becomes without shaping!");
                ImGui::SliderFloat("Fuel Penalty u^2", &rs.mcar_effort_weight, 0.0f, 0.20f, "%.3f");
            } else if (ek == 4) { // Drone2D
                ImGui::SliderFloat("Tilt Penalty |th|", &rs.drone_tilt_weight, 0.0f, 2.0f, "%.2f");
                ImGui::SliderFloat("Vel Damping", &rs.drone_vel_weight, 0.0f, 0.30f, "%.2f");
                ImGui::SliderFloat("Rotor Effort", &rs.drone_effort_weight, 0.0f, 0.15f, "%.3f");
            } else if (ek == 5) { // ContinuousGrid2D
                ImGui::Checkbox("Dense Distance Shaping", &rs.cgrid_dense_shaping);
                HelpMarker("Uncheck for SPARSE reward (+10 Goal / -10 Trap only)!");
                ImGui::SliderFloat("Velocity Effort", &rs.cgrid_effort_weight, 0.0f, 0.20f, "%.3f");
            } else if (ek == 6) { // LunarLander
                ImGui::SliderFloat("Fuel Penalty", &rs.lander_fuel_weight, 0.0f, 1.50f, "%.2f");
                HelpMarker("Cost of firing main & RCS thrusters. High fuel penalty encourages free-fall + last-second suicide burn!");
            }

            ImGui::Separator();
            ImGui::Text("6. Experiment Comparison");
            if (ImGui::Button("Snapshot Current Curves", ImVec2(-1, 26))) {
                for (int k = 0; k < 3; ++k) {
                    const auto& t = suite.telemetry(k);
                    if (t.enabled && !t.y_return.empty()) {
                        char lbl[96];
                        const char* short_n[3] = {"SAC", "PPO", "TD3"};
                        std::snprintf(lbl, sizeof(lbl), "%s (lr=%.4f, seed=%d)", short_n[k], hp.lr, hp.seed);
                        saved_runs.push_back({lbl, t.x_steps, t.x_time_s, t.y_return});
                    }
                }
            }
            HelpMarker("Freezes the current learning curves onto the plot so you can change Learning Rate or Seed, click Reset, and directly compare!");
            if (!saved_runs.empty()) {
                if (ImGui::Button("Clear Snapshots", ImVec2(-1, 24))) saved_runs.clear();
                ImGui::TextDisabled("%d saved curve(s) overlaid", (int)saved_runs.size());
            }

            ImGui::Separator();
            ImGui::Text("6. Rollout Scrubber");
            if (ImGui::Button(playing_anim ? "Pause" : "Play Rollout", ImVec2(110, 24))) {
                if (!playing_anim) {
                    int max_f_chk = std::max({1, (int)suite.telemetry(0).rollout.size() - 1,
                                              (int)suite.telemetry(1).rollout.size() - 1,
                                              (int)suite.telemetry(2).rollout.size() - 1});
                    if (anim_frame >= max_f_chk) anim_frame = 0;
                    anim_accum = 0.0f;
                }
                playing_anim = !playing_anim;
            }
            ImGui::SameLine();
            ImGui::Checkbox("Loop", &loop_anim);
            ImGui::SetNextItemWidth(160);
            ImGui::SliderFloat("Speed", &anim_speed, 0.2f, 2.5f, "%.1fx");
            int max_f = std::max({1, (int)suite.telemetry(0).rollout.size() - 1,
                                  (int)suite.telemetry(1).rollout.size() - 1,
                                  (int)suite.telemetry(2).rollout.size() - 1});
            if (ImGui::SliderInt("Frame", &anim_frame, 0, max_f)) {
                playing_anim = false; // Dragging the scrubber pauses playback cleanly
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // ====================================================================
        // MAIN AREA: Top Live Simulation Viewports + Bottom ImPlot Analytics
        // ====================================================================
        ImGui::BeginChild("MainWorkspace", ImVec2(0, 0), false);

        if (selected_env == 0) {
            // Multi-Armed Bandits Lab: Top Arm Distribution Bar Chart + Bottom Regret & Optimal % Plots
            float top_h = ImGui::GetContentRegionAvail().y * 0.45f;
            if (ImPlot::BeginPlot("10-Armed Bandit: True Mean vs. Agent Q-Estimates", ImVec2(-1, top_h))) {
                ImPlot::SetupAxes("Arm Index (0..9)", "Expected Reward", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                double arm_x[studio::BanditsLab::K];
                for (int i = 0; i < studio::BanditsLab::K; ++i) arm_x[i] = static_cast<double>(i);
                ImPlot::PlotBars("Eps-Greedy Q(a)", arm_x, bandits_lab.agents[0].q_est.data(), studio::BanditsLab::K, 0.22, ImPlotSpec(ImPlotProp_Offset, 0));
                ImPlot::PlotBars("UCB1 Q(a)", arm_x, bandits_lab.agents[1].q_est.data(), studio::BanditsLab::K, 0.22);
                ImPlot::PlotBars("Thompson Q(a)", arm_x, bandits_lab.agents[2].q_est.data(), studio::BanditsLab::K, 0.22);
                ImPlot::PlotScatter("True Mean q*(a)", arm_x, bandits_lab.true_means.data(), studio::BanditsLab::K);
                ImPlot::EndPlot();
            }

            if (ImGui::BeginTabBar("BanditPlots")) {
                if (ImGui::BeginTabItem("Cumulative Regret (Lower is Better!)")) {
                    if (ImPlot::BeginPlot("Cumulative Regret vs. Pulls", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Pulls (t)", "Cumulative Regret", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!bandits_lab.x_step.empty()) {
                            int n = (int)bandits_lab.x_step.size();
                            ImPlot::PlotLine("Eps-Greedy", bandits_lab.x_step.data(), bandits_lab.hist_cum_regret[0].data(), n);
                            ImPlot::PlotLine("UCB1", bandits_lab.x_step.data(), bandits_lab.hist_cum_regret[1].data(), n);
                            ImPlot::PlotLine("Thompson Sampling", bandits_lab.x_step.data(), bandits_lab.hist_cum_regret[2].data(), n);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("% Optimal Arm Pulls")) {
                    if (ImPlot::BeginPlot("Optimal Arm Selection (%)", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Pulls (t)", "% Optimal Action", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!bandits_lab.x_step.empty()) {
                            int n = (int)bandits_lab.x_step.size();
                            ImPlot::PlotLine("Eps-Greedy", bandits_lab.x_step.data(), bandits_lab.hist_opt_pct[0].data(), n);
                            ImPlot::PlotLine("UCB1", bandits_lab.x_step.data(), bandits_lab.hist_opt_pct[1].data(), n);
                            ImPlot::PlotLine("Thompson Sampling", bandits_lab.x_step.data(), bandits_lab.hist_opt_pct[2].data(), n);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Average Reward")) {
                    if (ImPlot::BeginPlot("Average Reward Over Time", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Pulls (t)", "Avg Reward", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!bandits_lab.x_step.empty()) {
                            int n = (int)bandits_lab.x_step.size();
                            ImPlot::PlotLine("Eps-Greedy", bandits_lab.x_step.data(), bandits_lab.hist_avg_reward[0].data(), n);
                            ImPlot::PlotLine("UCB1", bandits_lab.x_step.data(), bandits_lab.hist_avg_reward[1].data(), n);
                            ImPlot::PlotLine("Thompson Sampling", bandits_lab.x_step.data(), bandits_lab.hist_avg_reward[2].data(), n);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else if (selected_env == 1) {
            // GridWorld Tabular View: 2 Side-by-Side Interactive Grids (Click-to-Paint + Hover Inspector + DP Value Iteration!)
            float top_h = ImGui::GetContentRegionAvail().y * 0.48f;
            ImGui::BeginChild("GridViewports", ImVec2(0, top_h), false);
            float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;

            float t_sec = static_cast<float>(ImGui::GetTime());
            int cx1 = -1, cy1 = -1, cx2 = -1, cy2 = -1;

            ImGui::BeginChild("QGrid", ImVec2(half_w, 0), false);
            ImGui::TextColored(ImVec4(0.22f, 0.74f, 0.97f, 1.0f),
                               "Q-Learning Robot (Off-Policy) | Ep: %d | Slip: %.0f%%",
                               grid_studio.episodes, grid_studio.slip_prob * 100.0f);
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            DrawGridWorldCanvas(ImGui::GetWindowDrawList(), p0, sz, grid_studio.env_q, grid_studio.q_agent,
                                grid_studio.episodes, grid_studio.q_path, anim_frame, t_sec,
                                IM_COL32(56, 189, 248, 255), nullptr, cx1, cy1);
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("SGrid", ImVec2(0, 0), false);
            if (grid_studio.show_dp_overlay) {
                ImGui::TextColored(ImVec4(0.98f, 0.80f, 0.35f, 1.0f),
                                   "Exact Value Iteration (Model-Based DP) | Sweeps: %d",
                                   grid_studio.dp_solver.iterations);
                p0 = ImGui::GetCursorScreenPos();
                sz = ImGui::GetContentRegionAvail();
                DrawGridWorldCanvas(ImGui::GetWindowDrawList(), p0, sz, grid_studio.env_q, grid_studio.s_agent,
                                    grid_studio.episodes, grid_studio.s_path, anim_frame, t_sec,
                                    IM_COL32(250, 204, 21, 255), &grid_studio.dp_solver, cx2, cy2);
            } else {
                ImGui::TextColored(ImVec4(0.91f, 0.47f, 0.98f, 1.0f),
                                   "SARSA Robot (On-Policy) | Ep: %d | Slip: %.0f%%",
                                   grid_studio.episodes, grid_studio.slip_prob * 100.0f);
                p0 = ImGui::GetCursorScreenPos();
                sz = ImGui::GetContentRegionAvail();
                DrawGridWorldCanvas(ImGui::GetWindowDrawList(), p0, sz, grid_studio.env_s, grid_studio.s_agent,
                                    grid_studio.episodes, grid_studio.s_path, anim_frame, t_sec,
                                    IM_COL32(232, 121, 249, 255), nullptr, cx2, cy2);
            }
            ImGui::EndChild();
            ImGui::EndChild();

            if (cx1 >= 0 && cy1 >= 0) grid_studio.paint_cell(cx1, cy1);
            if (cx2 >= 0 && cy2 >= 0) grid_studio.paint_cell(cx2, cy2);

            // ImPlot Charts for GridWorld
            if (ImGui::BeginTabBar("GridPlots")) {
                if (ImGui::BeginTabItem("Smoothed Episode Return")) {
                    if (ImPlot::BeginPlot("Episode Return (Q-Learning vs. SARSA)", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Episode", "Smoothed Return", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!grid_studio.x_ep.empty()) {
                            ImPlot::PlotLine("Q-Learning", grid_studio.x_ep.data(), grid_studio.q_ret.data(), (int)grid_studio.x_ep.size());
                            ImPlot::PlotLine("SARSA", grid_studio.x_ep.data(), grid_studio.s_ret.data(), (int)grid_studio.x_ep.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Bellman TD Loss & Mean Q")) {
                    float pw = (ImGui::GetContentRegionAvail().x - 8.0f) * 0.5f;
                    if (ImPlot::BeginPlot("Mean TD Error |Target - Q|", ImVec2(pw, -1))) {
                        ImPlot::SetupAxes("Episode", "TD Error", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!grid_studio.x_ep.empty()) {
                            ImPlot::PlotLine("Q-Learning", grid_studio.x_ep.data(), grid_studio.q_loss.data(), (int)grid_studio.x_ep.size());
                            ImPlot::PlotLine("SARSA", grid_studio.x_ep.data(), grid_studio.s_loss.data(), (int)grid_studio.x_ep.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();
                    if (ImPlot::BeginPlot("Mean Max Q-Value", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Episode", "Mean Max Q", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        if (!grid_studio.x_ep.empty()) {
                            ImPlot::PlotLine("Q-Learning", grid_studio.x_ep.data(), grid_studio.q_mean_q.data(), (int)grid_studio.x_ep.size());
                            ImPlot::PlotLine("SARSA", grid_studio.x_ep.data(), grid_studio.s_mean_q.data(), (int)grid_studio.x_ep.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else if (selected_env == 2) {
            // Taxi-v3 View: Side-by-Side Q-Learning vs. SARSA Taxis + Return Plot
            float top_h = ImGui::GetContentRegionAvail().y * 0.52f;
            ImGui::BeginChild("TaxiViewports", ImVec2(0, top_h), false);
            float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;

            ImGui::BeginChild("QTaxi", ImVec2(half_w, 0), false);
            ImGui::TextColored(ImVec4(0.22f, 0.74f, 0.97f, 1.0f), "Q-Learning Taxi (500 States) | Ep: %d", taxi_lab.episodes);
            DrawTaxiCanvas(ImGui::GetWindowDrawList(), ImGui::GetCursorScreenPos(), ImGui::GetContentRegionAvail(),
                           taxi_lab.path_q, anim_frame, IM_COL32(56, 189, 248, 255));
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("STaxi", ImVec2(0, 0), false);
            ImGui::TextColored(ImVec4(0.91f, 0.47f, 0.98f, 1.0f), "SARSA Taxi (500 States) | Ep: %d", taxi_lab.episodes);
            DrawTaxiCanvas(ImGui::GetWindowDrawList(), ImGui::GetCursorScreenPos(), ImGui::GetContentRegionAvail(),
                           taxi_lab.path_s, anim_frame, IM_COL32(232, 121, 249, 255));
            ImGui::EndChild();
            ImGui::EndChild();

            if (ImPlot::BeginPlot("Taxi-v3 Smoothed Episode Return (Q-Learning vs. SARSA)", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("Episode", "Smoothed Return", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                if (!taxi_lab.x_ep.empty()) {
                    int n = (int)taxi_lab.x_ep.size();
                    ImPlot::PlotLine("Q-Learning", taxi_lab.x_ep.data(), taxi_lab.ret_q.data(), n);
                    ImPlot::PlotLine("SARSA", taxi_lab.x_ep.data(), taxi_lab.ret_s.data(), n);
                }
                ImPlot::EndPlot();
            }
        } else {
            // Continuous RLtools Suite View: 3 Side-by-Side Live Viewports + Zoomable ImPlot Tabs
            auto& suite = *cont_suites[selected_env - 3];
            ImGui::TextColored(ImVec4(0.98f, 0.80f, 0.08f, 1.0f), "%s", suite.title().c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("- %s", suite.description().c_str());

            float top_h = ImGui::GetContentRegionAvail().y * 0.44f;
            ImGui::BeginChild("ContViewports", ImVec2(0, top_h), false);
            float col_w = (ImGui::GetContentRegionAvail().x - 16.0f) / 3.0f;
            const ImU32 cols[3] = {IM_COL32(56, 189, 248, 255), IM_COL32(232, 121, 249, 255), IM_COL32(251, 146, 60, 255)};
            const ImVec4 vcols[3] = {{0.22f, 0.74f, 0.97f, 1.0f}, {0.91f, 0.47f, 0.98f, 1.0f}, {0.98f, 0.57f, 0.24f, 1.0f}};

            for (int k = 0; k < 3; ++k) {
                if (k > 0) ImGui::SameLine();
                ImGui::BeginChild(k + 100, ImVec2(k < 2 ? col_w : 0, 0), false);
                const auto& t = suite.telemetry(k);
                ImGui::TextColored(vcols[k], "%s", t.name.c_str());
                if (t.solved) {
                    ImGui::TextColored(ImVec4(0.30f, 0.95f, 0.55f, 1.0f),
                                       "SOLVED! Rd %d | %d steps | %.0f ms",
                                       t.solved_at_round, t.solved_at_steps, t.solved_at_ms);
                } else {
                    ImGui::TextDisabled("Steps: %d | Time: %.0f ms | Ret: %.1f", t.env_steps, t.wall_ms, t.last_return);
                }
                ImVec2 p0 = ImGui::GetCursorScreenPos();
                ImVec2 sz = ImGui::GetContentRegionAvail();
                DrawContinuousCanvas(ImGui::GetWindowDrawList(), p0, sz, suite.env_kind(), t.rollout, anim_frame, cols[k]);
                ImGui::EndChild();
            }
            ImGui::EndChild();

            // X-Axis Switcher (Rounds vs. Sample Efficiency vs. Wall-Clock Speed!)
            ImGui::Text("Plot X-Axis:");
            ImGui::SameLine();
            ImGui::RadioButton("Training Rounds", &x_axis_mode, 0); ImGui::SameLine();
            ImGui::RadioButton("Env Steps (Sample Efficiency)", &x_axis_mode, 1); ImGui::SameLine();
            ImGui::RadioButton("Wall-Clock Seconds (Compute Speed)", &x_axis_mode, 2);

            const char* x_label = (x_axis_mode == 0) ? "Round" : ((x_axis_mode == 1) ? "Environment Steps" : "Wall-Clock Time (s)");

            if (ImGui::BeginTabBar("ContPlots")) {
                if (ImGui::BeginTabItem("Eval Return (SAC vs. PPO vs. TD3)")) {
                    if (ImPlot::BeginPlot("Evaluation Return Comparison", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes(x_label, "Evaluation Return", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        for (int k = 0; k < 3; ++k) {
                            const auto& t = suite.telemetry(k);
                            if (t.y_return.empty()) continue;
                            const auto& xs = (x_axis_mode == 0) ? t.x_round : ((x_axis_mode == 1) ? t.x_steps : t.x_time_s);
                            ImPlot::PlotLine(t.name.c_str(), xs.data(), t.y_return.data(), (int)t.y_return.size());
                        }
                        for (const auto& sr : saved_runs) {
                            if (x_axis_mode == 1) ImPlot::PlotLine(sr.label.c_str(), sr.x_steps.data(), sr.y_return.data(), (int)sr.y_return.size());
                            else if (x_axis_mode == 2) ImPlot::PlotLine(sr.label.c_str(), sr.x_time_s.data(), sr.y_return.data(), (int)sr.y_return.size());
                        }
                        // Draw horizontal Target Return threshold line!
                        double thresh_y = static_cast<double>(target_return);
                        ImPlotSpec thresh_spec;
                        thresh_spec.Flags = ImPlotInfLinesFlags_Horizontal;
                        ImPlot::PlotInfLines("Target Threshold", &thresh_y, 1, thresh_spec);
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Episode Length")) {
                    if (ImPlot::BeginPlot("Episode Steps", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes(x_label, "Episode Length (Steps)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        for (int k = 0; k < 3; ++k) {
                            const auto& t = suite.telemetry(k);
                            if (t.y_ep_len.empty()) continue;
                            const auto& xs = (x_axis_mode == 0) ? t.x_round : ((x_axis_mode == 1) ? t.x_steps : t.x_time_s);
                            ImPlot::PlotLine(t.name.c_str(), xs.data(), t.y_ep_len.data(), (int)t.y_ep_len.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Threshold Benchmark Table")) {
                    ImGui::Spacing();
                    ImGui::Text("Comparison of Sample Efficiency & Wall-Clock Speed to reach Target Return >= %.1f:", target_return);
                    ImGui::Spacing();
                    if (ImGui::BeginTable("BenchTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Algorithm");
                        ImGui::TableSetupColumn("Status");
                        ImGui::TableSetupColumn("Rounds to Solve");
                        ImGui::TableSetupColumn("Env Steps to Solve");
                        ImGui::TableSetupColumn("Wall Time to Solve");
                        ImGui::TableSetupColumn("Latest Eval Return");
                        ImGui::TableHeadersRow();
                        for (int k = 0; k < 3; ++k) {
                            const auto& t = suite.telemetry(k);
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(t.name.c_str());
                            ImGui::TableSetColumnIndex(1);
                            if (t.solved) ImGui::TextColored(ImVec4(0.3f, 0.95f, 0.55f, 1.0f), "SOLVED");
                            else ImGui::TextDisabled(race_to_target ? "Training..." : "Not Reached");
                            ImGui::TableSetColumnIndex(2);
                            if (t.solved) ImGui::Text("%d", t.solved_at_round); else ImGui::Text("-");
                            ImGui::TableSetColumnIndex(3);
                            if (t.solved) ImGui::Text("%d", t.solved_at_steps); else ImGui::Text("%d (ongoing)", t.env_steps);
                            ImGui::TableSetColumnIndex(4);
                            if (t.solved) ImGui::Text("%.1f ms", t.solved_at_ms); else ImGui::Text("%.1f ms", t.wall_ms);
                            ImGui::TableSetColumnIndex(5); ImGui::Text("%.2f", t.last_return);
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Rollout Action & Reward Timeline")) {
                    if (ImPlot::BeginPlot("Applied Action u(t) During Latest Rollout", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Rollout Timestep", "Action u in [-1, 1]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        for (int k = 0; k < 3; ++k) {
                            const auto& r = suite.telemetry(k).rollout;
                            if (r.empty()) continue;
                            std::vector<double> ts(r.size()), acts(r.size());
                            for (size_t i = 0; i < r.size(); ++i) { ts[i] = (double)i; acts[i] = r[i].a[0]; }
                            ImPlot::PlotLine(suite.telemetry(k).name.c_str(), ts.data(), acts.data(), (int)ts.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Phase Space Portrait")) {
                    const char* px_lbl = (suite.env_kind() == 0) ? "Pole Angle theta (rad)"
                                       : ((suite.env_kind() == 1) ? "Angle theta (rad)"
                                       : ((suite.env_kind() == 3) ? "Car Position x"
                                       : "State s[0] (x / theta1)"));
                    const char* py_lbl = (suite.env_kind() == 0) ? "Angular Velocity theta_dot"
                                       : ((suite.env_kind() == 1) ? "Angular Velocity theta_dot"
                                       : ((suite.env_kind() == 3) ? "Car Velocity v"
                                       : "State s[1] (y / theta2)"));
                    if (ImPlot::BeginPlot("State Phase-Space Trajectory", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes(px_lbl, py_lbl, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        for (int k = 0; k < 3; ++k) {
                            const auto& r = suite.telemetry(k).rollout;
                            if (r.empty()) continue;
                            std::vector<double> xs(r.size()), ys(r.size());
                            for (size_t i = 0; i < r.size(); ++i) {
                                if (suite.env_kind() == 0) { xs[i] = r[i].s[1]; ys[i] = r[i].s[3]; }
                                else { xs[i] = r[i].s[0]; ys[i] = r[i].s[1]; }
                            }
                            ImPlot::PlotLine(suite.telemetry(k).name.c_str(), xs.data(), ys.data(), (int)xs.size());
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Neural Policy Heatmap")) {
                    ImGui::Text("Inspect Neural Actor Policy u(s) across 2D State Slice:");
                    ImGui::SameLine();
                    ImGui::RadioButton("SAC##hm", &heatmap_algo, 0); ImGui::SameLine();
                    ImGui::RadioButton("PPO##hm", &heatmap_algo, 1); ImGui::SameLine();
                    ImGui::RadioButton("TD3##hm", &heatmap_algo, 2);

                    constexpr int GRID_N = 25;
                    int cur_steps = suite.telemetry(heatmap_algo).env_steps;
                    if (cached_heatmap_env != selected_env || cached_heatmap_algo != heatmap_algo ||
                        cached_heatmap_steps != cur_steps || cached_heatmap.empty()) {
                        cached_heatmap = suite.compute_policy_heatmap(heatmap_algo, GRID_N);
                        cached_heatmap_env = selected_env;
                        cached_heatmap_algo = heatmap_algo;
                        cached_heatmap_steps = cur_steps;
                    }

                    ImPlot::PushColormap(ImPlotColormap_RdBu);
                    if (ImPlot::BeginPlot("Actor Policy Action Heatmap u in [-1 (Red), +1 (Blue)]", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("State Dim 1 (Normalized -1..+1)", "State Dim 2 (Normalized -1..+1)");
                        ImPlot::PlotHeatmap("Policy u(s)", cached_heatmap.data(), GRID_N, GRID_N, -1.0, 1.0,
                                            nullptr, ImPlotPoint(-1.0, -1.0), ImPlotPoint(1.0, 1.0));
                        ImPlot::EndPlot();
                    }
                    ImPlot::PopColormap();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }

        ImGui::EndChild();
        ImGui::End();

        // Render DirectX11 Frame
        ImGui::Render();
        const float clear_col[4] = {0.07f, 0.08f, 0.11f, 1.0f};
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_col);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0); // VSync 60 FPS
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

