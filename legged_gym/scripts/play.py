# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pandas as pd

# index_csv = [['base1','shoulder1', 'elbow1', 'base2', 'shoulder2', 'elbow2', 'base3', 'shoulder3', 'elbow3', 'base4', 'shoulder4', 'elbow4'
#              , 'base5', 'shoulder5', 'elbow5', 'base6', 'shoulder6', 'elbow6']]
index_csv = [['dof_target','dof_pos']]

rest_position_gym = np.array([0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0])

a1_rest_torques = np.array([ 0.8347,  0.2094,  3.8837, -0.1135, -0.8283,  4.6491, -0.2000, -0.5681,
         4.2520, -0.6055,  0.1871,  3.5612])

df_imit = pd.read_csv('imitation_data/imitation_data_yuna_torques.csv', parse_dates=False)
# df_imit = pd.read_csv('imitation_data/imitation_data_wtw.csv', parse_dates=False)
# df_imit = pd.read_csv('imitation_data/imitation_cassie.csv', parse_dates=False)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    FF_counter = 1
    avg_torque = 0
    error_array = []
    error_v_array = []
    arror_w_array = []
    # actions = np.zeros(18)
    for i in range(10*int(env.max_episode_length)):
    # for i in range(450):
        # print("FF_counter", FF_counter)
        actions = policy(obs.detach())
        # print("OBS_Grav_vec",np.round(obs.detach().cpu().numpy()[:,0:3], decimals=2))
        # print("OBS_Cmd", np.round(obs.detach().cpu().numpy()[:,3:6], decimals=2))
        # print("OBS_DOF_Pos", np.round(obs.detach().cpu().numpy()[:,6:18], decimals=2))
        # print("OBS_DOF_Vel", np.round(obs.detach().cpu().numpy()[:,18:30], decimals=2))
        # print("OBS_Prev_Act", np.round(obs.detach().cpu().numpy()[:,30:42], decimals=2))
        # print("ACT",np.round(actions.detach().cpu().numpy()*1.0, decimals=2))
        # print("....................................................")
        # print("ACT",np.round(actions.detach().cpu().numpy(), decimals=2))
        # print("PID TORQUES", env.torques[robot_index, :])
        # for j in range(env.num_actions):
        #     actions[:,j] = 0.0
        obs, _, rews, dones, infos = env.step(actions.detach())
        avg_torque += env.torques[robot_index, :]
        FF_counter += 1

        dof_target =  np.clip(actions[robot_index, joint_index].item() * env.cfg.control.action_scale, -100,100)
        dof_pos =  env.dof_pos[robot_index, joint_index].item()
        # df = pd.DataFrame([[dof_target, dof_pos]])
        # df.to_csv("testfile.csv",index=False, mode='a', header=False) #save to file

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                filename = '/home/marmot/Sood/yuna_legged_gym/videos/' + f"{img_idx}.png"
                # print("FILENAME", filename)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if LOG_IMITATION_ERROR:
            #print the average error between imitation angles and actual angles
            dof_pos = env.dof_pos[robot_index, :].detach().cpu().numpy()
            dof_pos_imit = df_imit.iloc[i, :18].values
            error = np.square(np.abs(dof_pos - dof_pos_imit))
            error_array.append(error)

            #print the average error between imitation velocities and actual velocities
            # base_lin_vel = env.base_lin_vel[robot_index, :].detach().cpu().numpy()
            # base_lin_vel_imit = df_imit.iloc[i, 18:21].values
            # error_v = np.square(np.abs(base_lin_vel - base_lin_vel_imit))
            # error_v_array.append(error_v)

            # #print the average error between imitation angular velocities and actual angular velocities
            # base_ang_vel = env.base_ang_vel[robot_index, :].detach().cpu().numpy()
            # base_ang_vel_imit = df_imit.iloc[i, 20:21].values
            # error_w = np.square(np.abs(base_ang_vel - base_ang_vel_imit))
            # arror_w_array.append(error_w)
            
            if i == 1000:
                error_avg = np.sqrt(np.mean(error_array))
                print("Average error between imitation angles and actual angles", error_avg)

                # error_v_avg = np.sqrt(np.mean(error_v_array))
                # print("Average error between imitation velocities and actual velocities", error_v_avg)

                # error_w_avg = np.sqrt(np.mean(arror_w_array))
                # print("Average error between imitation angular velocities and actual angular velocities", error_w_avg)
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': np.clip(actions[robot_index, joint_index].item() * env.cfg.control.action_scale, -100,100),
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        #         }
        #     )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()



if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    LOG_IMITATION_ERROR = False
    args = get_args()
    df = pd.DataFrame(index_csv) #convert to a dataframe
    df.to_csv("testfile.csv",index=False, mode='w', header=False) #save to file
    play(args)
