import os
import time
import numpy as np
import yaml
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import Optional

# Isaac Gym before torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, export_policy_as_jit

import torch

import viser
import viser.transforms as tf
import viser.uplot
from viser.extras import ViserUrdf
import yourdfpy

EXPORT_POLICY = True
RECORD_FRAMES = False
LOG_IMITATION_ERROR = False
VISUALIZE_IMITATION = True  # keep feature; can toggle at runtime

# ---------- Config / assets ----------
with open("legged_gym/envs/param_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path_to_imit = config["path_to_imitation_data"]

imit_np = None
if LOG_IMITATION_ERROR or VISUALIZE_IMITATION:
    import pandas as pd
    # Preload only needed cols to NumPy for speed: feet EE (22:34)
    df_imit = pd.read_csv(path_to_imit, parse_dates=False)
    imit_np = df_imit.iloc[:, 22:34].to_numpy().reshape(-1, 4, 3)

# ---------- Helpers ----------
PINK_GRID = dict(
    cell_color=(235, 190, 235),
    cell_thickness=0.5,
    section_color=(245, 160, 245),
    section_thickness=0.8,
)

FOOT_COLORS = np.array([
    [250, 120, 120],  # FL
    [120, 250, 160],  # FR
    [120, 140, 255],  # RL
    [255, 230, 120],  # RR
], dtype=np.uint8)

def cam_offset(distance: float, height: float, angle_deg: float, up="+z") -> np.ndarray:
    ang = np.radians(angle_deg)
    x = -distance * np.cos(ang)
    y = -distance * np.sin(ang)
    if up == "+z":
        return np.array([x, y, height])
    # Y-up fallback
    return np.array([x, height, y])

def setup_scene(server: viser.ViserServer):
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(True, True)

    # Ground mat 
    # server.scene.add_mesh_simple(
    #     name="/ground_mat",
    #     vertices=np.array([[-20, -20, -0.002], [20, -20, -0.002], [20, 20, -0.002], [-20, 20, -0.002]]),
    #     faces=np.array([[0, 1, 2], [0, 2, 3]]),
    #     color=(255, 230, 240),
    # )

    server.scene.add_grid(
        name="/grid_floor",
        width=100.0,
        height=100.0,
        width_segments=100,
        height_segments=100,
        plane="xy",
        cell_size=1.0,
        section_size=1.0,
        shadow_opacity=0.10,
        position=(0.0, 0.0, 0.0),
        visible=True,
        **PINK_GRID,
    )

def setup_real_time_plots(server: viser.ViserServer, max_points: int = 300):
    """Setup real-time plotting system using viser uplot - replaces matplotlib"""
    
    plot_state = {
        'time_data': np.array([]),
        'max_points': max_points,
        'plots': {},
        'data_buffers': {
            'dof_pos': [],
            'dof_pos_target': [],
            'dof_vel': [],
            'dof_torque': [],
            'base_vel_x': [],
            'base_vel_y': [],
            'base_vel_z': [],
            'base_vel_yaw': [],
            'command_x': [],
            'command_y': [],
            'command_yaw': [],
            'contact_forces_z': []
        }
    }
    
    # Create plots in organized GUI folders
    with server.gui.add_folder("📊 Real-time Plots"):
        
        # with server.gui.add_folder("Joint Control"):
        #     # DOF Position plot
        #     plot_state['plots']['dof_pos'] = server.gui.add_uplot(
        #         data=(np.array([0]), np.array([0]), np.array([0])),
        #         series=(
        #             viser.uplot.Series(label="time"),
        #             viser.uplot.Series(label="measured", stroke="blue", width=2),
        #             viser.uplot.Series(label="target", stroke="red", width=2, points_show=False)
        #         ),
        #         scales={
        #             "x": viser.uplot.Scale(time=False, auto=True),
        #             "y": viser.uplot.Scale(auto=True),
        #         },
        #         legend=viser.uplot.Legend(show=True),
        #         aspect=2.0,
        #     )
            
        #     # DOF Velocity plot
        #     plot_state['plots']['dof_vel'] = server.gui.add_uplot(
        #         data=(np.array([0]), np.array([0])),
        #         series=(
        #             viser.uplot.Series(label="time"),
        #             viser.uplot.Series(label="velocity", stroke="green", width=2)
        #         ),
        #         scales={
        #             "x": viser.uplot.Scale(time=False, auto=True),
        #             "y": viser.uplot.Scale(auto=True),
        #         },
        #         legend=viser.uplot.Legend(show=True),
        #         aspect=2.0,
        #     )
        
        with server.gui.add_folder("Base Velocity"):
            # Base velocity X
            plot_state['plots']['base_vel_x'] = server.gui.add_uplot(
                data=(np.array([0]), np.array([0]), np.array([0])),
                series=(
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="measured", stroke="blue", width=2),
                    viser.uplot.Series(label="commanded", stroke="orange", width=2, points_show=False)
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(auto=True),
                },
                legend=viser.uplot.Legend(show=True),
                aspect=2.0,
            )
            
            # Base velocity Y
            plot_state['plots']['base_vel_y'] = server.gui.add_uplot(
                data=(np.array([0]), np.array([0]), np.array([0])),
                series=(
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="measured", stroke="blue", width=2),
                    viser.uplot.Series(label="commanded", stroke="orange", width=2, points_show=False)
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(auto=True),
                },
                legend=viser.uplot.Legend(show=True),
                aspect=2.0,
            )
        
        with server.gui.add_folder("Forces & Torques"):
            # Contact forces
            plot_state['plots']['contact_forces'] = server.gui.add_uplot(
                data=(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])),
                series=(
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="FL", stroke="red", width=1.5),
                    viser.uplot.Series(label="FR", stroke="green", width=1.5),
                    viser.uplot.Series(label="RL", stroke="blue", width=1.5),
                    viser.uplot.Series(label="RR", stroke="orange", width=1.5)
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(auto=True),
                },
                legend=viser.uplot.Legend(show=True),
                aspect=2.0,
            )
            
            # Torques
            plot_state['plots']['torques'] = server.gui.add_uplot(
                data=(np.array([0]), np.array([0])),
                series=(
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(label="torque", stroke="purple", width=2)
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(auto=True),
                },
                legend=viser.uplot.Legend(show=True),
                aspect=2.0,
            )
    
    return plot_state

def update_real_time_plots(plot_state: dict, log_data: dict, current_time: float):
    """Update all real-time plots with new data - replaces matplotlib logger.plot_states()"""
    
    # Add current time
    plot_state['time_data'] = np.append(plot_state['time_data'], current_time)
    
    # Update data buffers
    for key, value in log_data.items():
        if key in plot_state['data_buffers']:
            if key == 'contact_forces_z':
                # Special handling for contact forces array
                plot_state['data_buffers'][key].append(value)
            else:
                plot_state['data_buffers'][key].append(value)
    
    # Limit data to max_points for performance
    max_pts = plot_state['max_points']
    if len(plot_state['time_data']) > max_pts:
        plot_state['time_data'] = plot_state['time_data'][-max_pts:]
        for key in plot_state['data_buffers']:
            if plot_state['data_buffers'][key]:
                plot_state['data_buffers'][key] = plot_state['data_buffers'][key][-max_pts:]
    
    time_data = plot_state['time_data']
    
    # Update DOF position plot
    if 'dof_pos' in plot_state['plots'] and plot_state['data_buffers']['dof_pos']:
        measured = np.array(plot_state['data_buffers']['dof_pos'])
        target = np.array(plot_state['data_buffers']['dof_pos_target']) if plot_state['data_buffers']['dof_pos_target'] else np.zeros_like(measured)
        plot_state['plots']['dof_pos'].data = (time_data, measured, target)
    
    # Update DOF velocity plot
    if 'dof_vel' in plot_state['plots'] and plot_state['data_buffers']['dof_vel']:
        velocity = np.array(plot_state['data_buffers']['dof_vel'])
        plot_state['plots']['dof_vel'].data = (time_data, velocity)
    
    # Update base velocity X plot
    if 'base_vel_x' in plot_state['plots'] and plot_state['data_buffers']['base_vel_x']:
        measured = np.array(plot_state['data_buffers']['base_vel_x'])
        commanded = np.array(plot_state['data_buffers']['command_x']) if plot_state['data_buffers']['command_x'] else np.zeros_like(measured)
        plot_state['plots']['base_vel_x'].data = (time_data, measured, commanded)
    
    # Update base velocity Y plot
    if 'base_vel_y' in plot_state['plots'] and plot_state['data_buffers']['base_vel_y']:
        measured = np.array(plot_state['data_buffers']['base_vel_y'])
        commanded = np.array(plot_state['data_buffers']['command_y']) if plot_state['data_buffers']['command_y'] else np.zeros_like(measured)
        plot_state['plots']['base_vel_y'].data = (time_data, measured, commanded)
    
    # Update contact forces plot
    if 'contact_forces' in plot_state['plots'] and plot_state['data_buffers']['contact_forces_z']:
        forces_data = plot_state['data_buffers']['contact_forces_z']
        if forces_data and len(forces_data) > 0:
            # Convert list of arrays to proper format
            forces_array = np.array(forces_data)
            if forces_array.ndim == 2 and forces_array.shape[1] >= 4:
                plot_state['plots']['contact_forces'].data = (
                    time_data,
                    forces_array[:, 0],  # FL
                    forces_array[:, 1],  # FR  
                    forces_array[:, 2],  # RL
                    forces_array[:, 3]   # RR
                )
    
    # Update torques plot
    if 'torques' in plot_state['plots'] and plot_state['data_buffers']['dof_torque']:
        torques = np.array(plot_state['data_buffers']['dof_torque'])
        plot_state['plots']['torques'].data = (time_data, torques)

def setup_camera_gui(server: viser.ViserServer, connected):
    state = dict(
        follow_enabled=True,
        mode="side",  # behind | side | top | cinematic | orbit | auto_angle
        distance=1.6,
        height=0.4,
        angle=-28.0,
        orbit_speed=0.5,
        orbit_theta=0.0,
        
        # --- NEW: dt-aware smoothing params (seconds) ---
        tau_pos=0.25,     # target position smoothing time-constant (~ responsiveness)
        tau_vel=0.35,     # velocity smoothing time-constant
        tau_z=0.60,       # vertical (z) follow time-constant (bigger = steadier horizon)
        tau_cam=0.20,     # camera pose interpolation time-constant (final ease)

        # --- NEW: physical limits ---
        max_speed=6.0,    # m/s cap for filtered target motion
        max_accel=30.0,   # m/s^2 cap for filtered acceleration

        # --- NEW: filter state (initialized lazily) ---
        pos_est=None,     # 3D filtered estimate
        vel_est=None,     # 3D filtered velocity
        z_follow=None,    # filtered z (soft horizon)

        # --- Yaw tracking / smoothing ---
        tau_yaw=0.25,      # time-constant for yaw smoothing
        yaw_est=None,      # filtered yaw estimate (radians, unwrapped)
        yaw_vel=0.0,       # filtered yaw rate (rad/s)
        yaw_look_ahead=0.8,# meters to look ahead along heading when following
        _yaw_last=None,    # for unwrapping continuity

        # legacy scalar smoothing removed from use in follow path
        smooth=0.3,
        z_fixed=None,
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        connected[client.client_id] = client
        client.camera.near = 0.05
        client.camera.far = 80.0
        client.camera.up_direction = (0.0, 0.0, 1.0)

        with client.gui.add_folder("Camera"):
            follow = client.gui.add_checkbox("Follow", initial_value=state["follow_enabled"])
            mode = client.gui.add_dropdown("Mode", ("behind","side","top","cinematic","orbit","auto_angle"), initial_value=state["mode"])
            dist = client.gui.add_slider("Distance", 0.6, 5.0, 0.1, state["distance"])
            height = client.gui.add_slider("Height", 0.3, 3.0, 0.1, state["height"])
            angle = client.gui.add_slider("Angle", -90.0, 90.0, 1.0, state["angle"])
            smooth = client.gui.add_slider("Smooth", 0.05, 0.5, 0.01, state["smooth"])
            orbit = client.gui.add_slider("Orbit Speed", 0.1, 3.0, 0.1, state["orbit_speed"])
            yaw_lookahead = client.gui.add_slider("Yaw Look-ahead", 0.0, 2.0, 0.1, state["yaw_look_ahead"])
            preset_front = client.gui.add_button("Front")
            preset_side  = client.gui.add_button("Side")
            preset_top   = client.gui.add_button("Top")
            reset_btn    = client.gui.add_button("Reset")

        @follow.on_update
        def _(_):
            state["follow_enabled"] = follow.value

        @mode.on_update
        def _(_):
            state["mode"] = mode.value
            if state["mode"] == "orbit":
                state["orbit_theta"] = 0.0

        @dist.on_update
        def _(_): state["distance"] = dist.value
        @height.on_update
        def _(_): state["height"] = height.value
        @angle.on_update
        def _(_): state["angle"] = angle.value
        @smooth.on_update
        def _(_): state["smooth"] = smooth.value
        @orbit.on_update
        def _(_): state["orbit_speed"] = orbit.value
        @yaw_lookahead.on_update
        def _(_): state["yaw_look_ahead"] = yaw_lookahead.value

        @preset_front.on_click
        def _(_):
            state["follow_enabled"] = False
            follow.value = False
            with client.atomic():
                client.camera.position = (0.0, -4.0, 1.5)
                client.camera.look_at  = (0.0,  0.0, 0.5)

        @preset_side.on_click
        def _(_):
            state["follow_enabled"] = False
            follow.value = False
            with client.atomic():
                client.camera.position = (-4.0,  0.0, 1.5)  # Changed from +4.0 to -4.0
                client.camera.look_at  = (0.0,  0.0, 0.5)

        @preset_top.on_click
        def _(_):
            state["follow_enabled"] = False
            follow.value = False
            with client.atomic():
                client.camera.position = (0.0,  0.0, 8.0)
                client.camera.look_at  = (0.0,  0.0, 0.0)

        @reset_btn.on_click
        def _(_):
            state["follow_enabled"] = True
            follow.value = True

    @server.on_client_disconnect
    def _(client: viser.ClientHandle):
        connected.pop(client.client_id, None)

    return state

def _exp_smooth_factor(dt: float, tau: float) -> float:
    """Continuous-time to discrete EMA. Larger tau -> more smoothing."""
    if tau <= 0.0:
        return 1.0
    return 1.0 - np.exp(-dt / max(1e-6, tau))

def _alpha_beta_update(pos_est, vel_est, meas, dt, tau_pos, tau_vel):
    """
    α–β filter tuned via time-constants so it's framerate-independent.
    """
    # Predict
    pred = pos_est + vel_est * dt
    residual = meas - pred

    # Convert time-constants to gains
    alpha = _exp_smooth_factor(dt, tau_pos)
    beta  = _exp_smooth_factor(dt, tau_vel)

    # Correct
    pos_est = pred + alpha * residual
    vel_est = vel_est + (beta / max(1e-6, dt)) * residual
    return pos_est, vel_est

def _limit_vec(vec, max_norm):
    n = np.linalg.norm(vec)
    if n > max_norm and n > 0:
        return vec * (max_norm / n)
    return vec

def _quat_xyzw_to_yaw(q_xyzw: np.ndarray) -> float:
    """Extract yaw from quaternion (x,y,z,w)"""
    x, y, z, w = q_xyzw
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(s, c))

def _unwrap_angle(curr: float, last_unwrapped: Optional[float]) -> float:
    """Keep yaw continuous across ±π boundaries."""
    if last_unwrapped is None:
        return curr
    delta = curr - (last_unwrapped % (2.0 * np.pi))
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return last_unwrapped + delta

def _Rz(theta: float) -> np.ndarray:
    """Rotation matrix around Z-axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def update_camera(connected, state, robot_pos, dt, robot_quat_xyzw=None):
    if not state["follow_enabled"] or not connected:
        return

    # --- Lazy init of filter state ---
    if state.get("pos_est") is None:
        state["pos_est"]  = np.array(robot_pos, dtype=np.float64)
        state["vel_est"]  = np.zeros(3, dtype=np.float64)
        state["z_follow"] = float(robot_pos[2])
        state["z_fixed"]  = float(robot_pos[2])
        state["_sm_cam_pos"]  = np.array(robot_pos, dtype=np.float64)
        state["_sm_cam_look"] = np.array(robot_pos, dtype=np.float64)
        # yaw init
        if robot_quat_xyzw is not None:
            yaw0 = _quat_xyzw_to_yaw(np.asarray(robot_quat_xyzw, dtype=np.float64))
        else:
            yaw0 = 0.0
        state["yaw_est"] = yaw0
        state["yaw_vel"] = 0.0
        state["_yaw_last"] = yaw0

    # --- Position α–β filter (already smoothed in your previous version) ---
    meas_pos = np.array(robot_pos, dtype=np.float64)
    pos_est  = state["pos_est"].copy()
    vel_est  = state["vel_est"].copy()

    pos_est, vel_est = _alpha_beta_update(
        pos_est, vel_est, meas_pos, dt, state["tau_pos"], state["tau_vel"]
    )
    # Limits
    vel_est = _limit_vec(vel_est, state["max_speed"])
    max_dv = state["max_accel"] * dt
    dv = vel_est - state["vel_est"]
    vel_est = state["vel_est"] + _limit_vec(dv, max_dv)

    # --- Yaw measurement (radians), unwrap, and α–β filter ---
    if robot_quat_xyzw is not None:
        yaw_meas_wrapped = _quat_xyzw_to_yaw(np.asarray(robot_quat_xyzw, dtype=np.float64))
    else:
        yaw_meas_wrapped = 0.0  # fallback if quat not provided this frame

    yaw_meas = _unwrap_angle(yaw_meas_wrapped, state["_yaw_last"])
    state["_yaw_last"] = yaw_meas

    yaw_est = state["yaw_est"]
    yaw_vel = state["yaw_vel"]

    yaw_est, yaw_vel = _alpha_beta_update(
        yaw_est, yaw_vel, yaw_meas, dt, state["tau_yaw"], state["tau_yaw"] * 1.25
    )
    # (Optional) clamp yaw rate if you like:
    # yaw_vel = float(np.clip(yaw_vel, -4*np.pi, 4*np.pi))

    # --- Soft Z follow to keep horizon steady ---
    az = _exp_smooth_factor(dt, state["tau_z"])
    z_follow = (1.0 - az) * state["z_follow"] + az * pos_est[2]

    # Compose smoothed follow position
    follow_pos = np.array([pos_est[0], pos_est[1], z_follow], dtype=np.float64)

    # --- Build heading-aligned camera target ---
    d, h, mode = state["distance"], state["height"], state["mode"]
    Rz = _Rz(yaw_est)

    if mode == "behind":
        rel = np.array([-d, 0.0, h])
        cam_target  = follow_pos + Rz @ rel
        look_target = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.3])
    elif mode == "side":
        rel = np.array([0.0, -d, h])   # left/right relative to heading
        cam_target  = follow_pos + Rz @ rel
        look_target = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.3])
    elif mode == "cinematic":
        rel = np.array([-0.7 * d, 0.5 * d, h])
        cam_target  = follow_pos + Rz @ rel
        look_target = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.3])
    elif mode == "top":
        cam_target  = follow_pos + np.array([0.0, 0.0, max(d * 1.5, h + 1.0)])
        look_target = follow_pos
    elif mode == "orbit":
        # Orbit around heading-aligned forward axis
        state["orbit_theta"] += state["orbit_speed"] * dt
        r = d
        rel_orbit = np.array([np.cos(state["orbit_theta"]) * r,
                              np.sin(state["orbit_theta"]) * r,
                              h])
        cam_target  = follow_pos + Rz @ rel_orbit
        look_target = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.3])
    else:  # auto_angle (angle in degrees, rotated around heading)
        ang = np.radians(state["angle"])
        rel = np.array([-d * np.cos(ang), -d * np.sin(ang), h])
        cam_target  = follow_pos + Rz @ rel
        look_target = follow_pos + Rz @ np.array([state["yaw_look_ahead"], 0.0, 0.0])

    # --- Final easing for micro-jitter ---
    ac = _exp_smooth_factor(dt, state["tau_cam"])
    state["_sm_cam_pos"]  = (1.0 - ac) * state["_sm_cam_pos"]  + ac * cam_target
    state["_sm_cam_look"] = (1.0 - ac) * state["_sm_cam_look"] + ac * look_target

    # --- Commit state ---
    state["pos_est"]  = pos_est
    state["vel_est"]  = vel_est
    state["z_follow"] = z_follow
    state["yaw_est"]  = float(yaw_est)
    state["yaw_vel"]  = float(yaw_vel)

    # --- Push to clients ---
    for c in connected.values():
        try:
            with c.atomic():
                c.camera.position = tuple(state["_sm_cam_pos"])
                c.camera.look_at  = tuple(state["_sm_cam_look"])
                c.camera.up_direction = (0.0, 0.0, 1.0)
        except Exception:
            pass

def overwrite_point_cloud(server, name: str, points: np.ndarray, colors, point_size: float):
    # Viser overwrites by name; single node reused each frame
    server.scene.add_point_cloud(name, points=points, colors=colors, point_size=point_size)

# ---------- Main ----------
def play(args):
    # Env + policy
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy:", path)

    # Viser with real-time plotting (replaces matplotlib)
    server = viser.ViserServer()
    setup_scene(server)

    connected = {}
    cam_state = setup_camera_gui(server, connected)
    
    # Setup real-time plots instead of matplotlib logger
    plot_state = setup_real_time_plots(server)

    # URDF
    go1_urdf_path = "resources/robots/go1/urdf/go1.urdf"
    urdf = yourdfpy.URDF.load(
        go1_urdf_path, load_meshes=True, build_scene_graph=True,
        load_collision_meshes=False, build_collision_scene_graph=False
    )
    root = server.scene.add_frame("/robot", axes_length=0.0, axes_radius=0.0)
    vbot = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot", load_meshes=True, load_collision_meshes=False)

    # GUI Perf + toggles
    with server.gui.add_folder("Viz"):
        show_trail = server.gui.add_checkbox("Motion Trail", initial_value=True)
        show_foot  = server.gui.add_checkbox("Foot Trails", initial_value=True)
        show_imit  = server.gui.add_checkbox("Imitation Targets", initial_value=False)
    with server.gui.add_folder("Perf"):
        fps_txt  = server.gui.add_text("FPS", "—", disabled=True)
        step_txt = server.gui.add_text("Step", "0", disabled=True)
        pos_txt  = server.gui.add_text("Base Pos", "—", disabled=True)
        vel_txt  = server.gui.add_text("Base Vel", "—", disabled=True)

    # Replace matplotlib logger with real-time viser plotting
    robot_i = 0
    joint_i = 2
    stop_state_log = 1000
    stop_rew_log = env.max_episode_length + 1

    # Histories (deque avoids slicing)
    path_pts = deque(maxlen=600)
    foot_hist = deque(maxlen=240)

    # Timing
    target_hz = 60.0
    target_dt = 1.0 / target_hz
    t_prev = time.perf_counter()
    fps_alpha = 0.1
    fps_est = None

    # Log helpers
    FF_counter = 0
    err_angles, err_vel, err_h = [], [], []

    for i in range(10 * int(env.max_episode_length)):
        loop_t0 = time.perf_counter()

        with torch.no_grad():
            act = policy(obs)
            obs, _, rews, dones, infos = env.step(act)

        # Pull once per item
        dof_pos = env.dof_pos[robot_i].detach().cpu().numpy()
        base_state = env.root_states[robot_i].detach().cpu().numpy()
        base_pos = base_state[:3]
        base_quat_xyzw = base_state[3:7]  # x y z w
        base_vel = env.base_lin_vel[robot_i].detach().cpu().numpy()

        # Foot positions
        foot_pos = None
        try:
            fp = env.foot_positions[robot_i].detach().cpu().numpy()
            if not np.allclose(fp, 0.0):
                foot_pos = fp
        except AttributeError:
            pass
        if foot_pos is None and hasattr(env, "rigid_body_state") and hasattr(env, "feet_indices"):
            rbs = env.rigid_body_state
            feet_idx = env.feet_indices.detach().cpu().numpy()
            num_rb = rbs.shape[0] // env.num_envs
            base_idx = robot_i * num_rb
            foot_pos = np.stack([rbs[base_idx + fi, :3].detach().cpu().numpy() for fi in feet_idx], axis=0)

        # Update URDF pose
        vbot.update_cfg(dof_pos)
        root.position = base_pos
        # Viser expects wxyz; convert xyzw -> wxyz
        root.wxyz = np.array([base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]])

        # Camera
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        update_camera(connected, cam_state, base_pos, dt, base_quat_xyzw)

        # Trails
        path_pts.append(base_pos.copy())
        if show_trail.value and len(path_pts) >= 2:
            overwrite_point_cloud(server, "/trail", np.array(path_pts), colors=(0, 220, 200), point_size=0.014)

        if foot_pos is not None:
            foot_hist.append(foot_pos.copy())
            if show_foot.value and len(foot_hist) >= 2:
                # stack recent foot points per foot
                recent = np.stack(foot_hist, axis=0)  # [T, 4, 3]
                for k in range(4):
                    pts = recent[:, k, :]
                    overwrite_point_cloud(server, f"/foot_trail_{k}", pts, colors=FOOT_COLORS[k], point_size=0.011)

        # Imitation target feet (world frame)
        if show_imit.value and (imit_np is not None):
            if hasattr(env, "imitation_index"):
                idx = int(env.imitation_index[robot_i].detach().cpu().numpy())
                idx = max(0, min(idx, len(imit_np) - 1))
            else:
                idx = i % len(imit_np)
            ee_body = imit_np[idx]  # [4,3] body frame
            r = R.from_quat(base_quat_xyzw)  # xyzw
            ee_world = r.apply(ee_body) + base_pos
            overwrite_point_cloud(server, "/imit_targets", ee_world, colors=(0, 255, 0), point_size=0.02)

        # Perf HUD (update ~10 Hz)
        if i % 6 == 0:
            inst_fps = 1.0 / max(1e-6, (time.perf_counter() - loop_t0))
            fps_est = inst_fps if fps_est is None else (1 - fps_alpha) * fps_est + fps_alpha * inst_fps
            fps_txt.value = f"{fps_est:5.1f}"
            step_txt.value = f"{i}"
            pos_txt.value = f"{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}"
            vel_txt.value = f"{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}"

        # Original logging bits (kept)
        if LOG_IMITATION_ERROR and i > 100 and (df_imit is not None):
            dof_pos_imit = df_imit.iloc[FF_counter, 6:18].values
            err_angles.append(np.square(np.abs(dof_pos - dof_pos_imit)))
            base_vx_cmd = obs[robot_i, 3].detach().cpu().numpy() / 2.0
            err_vel.append(np.square(np.abs(base_vel[0] - base_vx_cmd)))
            height_imit = df_imit.iloc[FF_counter, 21]
            err_h.append(np.square(np.abs(base_pos[2] - height_imit)))
            FF_counter += 1

        if i % env.max_episode_length == 0 and i > 0:
            path_pts.clear()
            foot_hist.clear()
            if LOG_IMITATION_ERROR and err_angles:
                print("Avg Err Angles", float(np.sqrt(np.mean(err_angles))))
                print("Avg Err Vel   ", float(np.sqrt(np.mean(err_vel))))
                print("Avg Err Height", float(np.sqrt(np.mean(err_h))))
                err_angles.clear(); err_vel.clear(); err_h.clear(); FF_counter = 0

        if i < stop_state_log:
            # Real-time plotting instead of logger.log_states()
            log_data = {
                # 'dof_pos_target': float(np.clip(act[robot_i, joint_i].item() * env.cfg.control.action_scale, -100, 100)),
                # 'dof_pos': float(env.dof_pos[robot_i, joint_i].item()),
                'dof_vel': float(env.dof_vel[robot_i, joint_i].item()),
                'dof_torque': float(env.torques[robot_i, joint_i].item()),
                'command_x': float(env.commands[robot_i, 0].item()),
                'command_y': float(env.commands[robot_i, 1].item()),
                'command_yaw': float(env.commands[robot_i, 2].item()),
                'base_vel_x': float(env.base_lin_vel[robot_i, 0].item()),
                'base_vel_y': float(env.base_lin_vel[robot_i, 1].item()),
                'base_vel_z': float(env.base_lin_vel[robot_i, 2].item()),
                'base_vel_yaw': float(env.base_ang_vel[robot_i, 2].item()),
                'contact_forces_z': env.contact_forces[robot_i, env.feet_indices, 2].detach().cpu().numpy(),
            }
            # Update real-time plots every few steps for performance
            if i % 3 == 0:  # Update plots every 3 steps
                current_time = i * env.dt
                update_real_time_plots(plot_state, log_data, current_time)
        elif i == stop_state_log:
            print("📊 Real-time plotting complete - all plots updated in browser")

        if 0 < i < stop_rew_log:
            if infos and "episode" in infos:
                nreset = torch.sum(env.reset_buf).item()
                if nreset > 0:
                    # Simple reward logging without matplotlib
                    for key, value in infos["episode"].items():
                        if 'rew' in key:
                            print(f"Episode {key}: {value.item():.3f}")
        elif i == stop_rew_log:
            print("📊 Episode reward logging complete")

        # frame pacing
        elapsed = time.perf_counter() - loop_t0
        sleep = target_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

if __name__ == "__main__":
    args = get_args()
    play(args)
