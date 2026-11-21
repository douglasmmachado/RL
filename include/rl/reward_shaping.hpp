/**
 * @file reward_shaping.hpp
 * @brief Runtime-tunable Reward Shaping weights shared across all RL Studio environments
 *        so users can toggle velocity/action/centering penalties live and observe reward shaping effects!
 */

#pragma once

namespace studio {

/**
 * @brief Live reward shaping configuration controlled by sliders & checkboxes in RL Studio.
 */
struct RewardShapingConfig {
    // 1. CartPole-Continuous
    float cartpole_center_penalty = 0.0f;  ///< Penalizes |x| distance from track center (0 = classic +1 only)
    float cartpole_angle_penalty = 0.0f;   ///< Penalizes |theta| tilt from vertical
    float cartpole_effort_penalty = 0.0f;  ///< Penalizes u^2 control force effort

    // 2. Acrobot Swing-Up
    bool acrobot_vel_penalty_on = true;    ///< Toggle angular velocity penalty (stops crazy spinning!)
    float acrobot_vel_weight = 0.02f;      ///< Weight on (theta1_dot^2 + theta2_dot^2)
    float acrobot_torque_weight = 0.01f;   ///< Weight on u^2 torque effort

    // 3. MountainCar-Continuous
    bool mcar_dense_shaping = true;        ///< Toggle dense height/velocity shaping vs sparse (+100 at flag only!)
    float mcar_effort_weight = 0.05f;      ///< Weight on u^2 engine energy penalty

    // 4. Drone2D Hover
    float drone_vel_weight = 0.05f;        ///< Penalty on linear velocity (smooth braking vs aggressive flight)
    float drone_tilt_weight = 0.50f;       ///< Penalty on pitch angle |theta|
    float drone_effort_weight = 0.02f;     ///< Penalty on rotor throttle effort

    // 5. ContinuousGrid2D Trap Arena
    bool cgrid_dense_shaping = true;       ///< Toggle dense distance-to-goal shaping vs sparse (+10 / -10 only!)
    float cgrid_effort_weight = 0.01f;     ///< Penalty on velocity effort

    // 6. Tabular GridWorld
    float grid_step_penalty = -0.1f;       ///< Step penalty per move (0.0 vs -0.1 vs -0.5)

    // 7. LunarLander (Moon Landing)
    float lander_fuel_weight = 0.30f;      ///< Main & side thruster fuel cost per step
};

/// @brief Global singleton RewardShapingConfig read by all custom RL Studio environments.
inline RewardShapingConfig& reward_shaping() {
    static RewardShapingConfig cfg;
    return cfg;
}

} // namespace studio
