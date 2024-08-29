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
from legged_gym.utils import quat_apply_yaw, get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *
import numpy as np
import torch
import pandas as pd

commands_for_train = np.array([0.4, 0.0, 0.0])

index_csv_imit = [['base1','shoulder1', 'elbow1', 'base2', 'shoulder2', 'elbow2', 'base3', 'shoulder3', 'elbow3', 'base4', 'shoulder4', 'elbow4'
             , 'base5', 'shoulder5', 'elbow5', 'base6', 'shoulder6', 'elbow6', 'com_vx','com_vy','com_wz', 'height','e1x','e1y','e1z','e2x','e2y','e2z','e3x','e3y','e3z','e4x','e4y','e4z',"e5x","e5y","e5z","e6x","e6y","e6z"]]


rest_position_gym = np.array([0.1, 0.0, 1.0, -1.8, 1.57, 
                              -1.57, -0.1, 0.0, 1.0, -1.8, 1.57, -1.57])


def play(args):
    global dof_pos
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

    # df = pd.read_csv('imitation_data.csv', parse_dates=False)

    # print(df.iloc[1000].to_numpy())
    # print(df.iloc[1001].to_numpy())
    #An infinite loop since the commands would be changed according to the training
    #TODO Assuming the robot doesn't fall down the terrain

    for i in range(int(env.max_episode_length)):
    # while (1):
        #Get the same commands being used in torque training
        # obs[:,9], obs[:,10], obs[:,11] = commands_for_train[0], commands_for_train[1], commands_for_train[2]
        # print(obs[:,9], obs[:,10], obs[:,11])
        actions = policy(obs.detach())

        # actions_imit = df.iloc[i].to_numpy() - rest_position_gym    
        # for j in range(18):
        #     actions[:,j] = actions_imit[j]

        obs, _, rews, dones, infos = env.step(actions.detach())
        # print(actions)
        #Get the joint angles executed by the robot
        # lin_vel_base = env.base_lin_vel
        # lin_vel_base = lin_vel_base.detach().cpu().numpy()

        # ang_vel_base = env.base_ang_vel
        # ang_vel_base = ang_vel_base.detach().cpu().numpy()

        dof_pos = env.dof_pos
        dof_pos = dof_pos.detach().cpu().numpy()

        commands = env.commands[:,0:3]
        commands = commands.detach().cpu().numpy()
        print("Commands", commands)
        commands = commands.reshape(1,3)

        height = env.base_pos[:, 2]
        height = height.detach().cpu().numpy()
        height = height.reshape(1,1)
        # print(dof_pos.shape, lin_vel_base.shape, ang_vel_base.shape)
        # print(dof_pos)
        

        #set z component of init_pos to 0 for translation
        # init_pos[:,2] = 0
        # #Repeat the array 4 times
        # init_pos_footsteps_translation = np.concatenate((init_pos,init_pos,init_pos,init_pos), axis=0)
        # # print("INIT POS", init_pos_footsteps_translation)
        # end_effector_pos_world = env.foot_positions.detach().cpu().numpy() - init_pos_footsteps_translation
        # # print("INIT POS",  np.around(end_effector_pos_world,decimals=2))
        # end_effector_pos_world = end_effector_pos_world.reshape(1,12)

        # base_pos_x_y = env.base_pos[:, 0:2] 
        # base_pos_x_y = base_pos_x_y.detach().cpu().numpy() - init_pos[:, 0:2]
        # base_pos_x_y = base_pos_x_y.reshape(1,2)
        # print("FOOT_HEIGHT", end_effector_pos_world[:,2])
        cur_footsteps_translated = env.foot_positions - env.base_pos.unsqueeze(1)
        # print("CUR POS",  np.around(cur_footsteps_translated.detach().cpu().numpy(),decimals=2))
        footsteps_in_body_frame = torch.zeros(env.num_envs, 2 , 3, device=env.device)
        for i in range(2):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(env.base_quat),
                                                              cur_footsteps_translated[:, i, :])
            
        footsteps_in_body_frame = footsteps_in_body_frame.detach().cpu().numpy()
        end_effector_pos = footsteps_in_body_frame.reshape(1,6)

        # quats = env.base_quat
        # quats = quats.detach().cpu().numpy()
        # quats = quats.reshape(1,4)

        # record_df = np.concatenate((lin_vel_base, ang_vel_base, dof_pos, commands,height, end_effector_pos, base_pos_x_y, quats, end_effector_pos_world), axis=1)
        record_df = np.concatenate((dof_pos, commands,height, end_effector_pos), axis=1)
        df = pd.DataFrame(record_df)
        df.to_csv("imitation_data_cassie.csv",index=False, mode='a', header=False)
        # dof_pos = dof_pos.reshape(1,4*18)
        # print(dof_pos)

if __name__ == '__main__':
    EXPORT_POLICY = True
    # RECORD_FRAMES = False
    # MOVE_CAMERA = False
    args = get_args()

    df = pd.DataFrame(index_csv_imit) #convert to a dataframe
    df.to_csv("imitation_data_cassie.csv",index=False, mode='w', header=False) #save to file

    play(args)
