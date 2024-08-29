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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
import pandas as pd
df_imit = pd.read_csv('imitation_data/imitation_data_wtw.csv', parse_dates=False)

class A1(LeggedRobot):
    def _reward_imitation_angles(self):
        index_array = self.imitation_index.detach().cpu().numpy().astype(int)

        # Retrieve the corresponding rows from df_imit using array indexing
        dof_imit_arr = df_imit.iloc[index_array,6:].to_numpy()
        # Reshape the array to the desired shape
        dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)

        # Convert the array to a PyTorch tensor
        dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)
        dof_imit_error = torch.sum(torch.square(self.dof_pos - dof_imit_arr), dim=1)    
        # print(dof_imit_arr)
        # return torch.sum(torch.abs(self.dof_pos - dof_imit_arr), dim=1)
        # return torch.sum(torch.square(self.dof_pos - dof_imit), dim=1)
        return torch.exp(-dof_imit_error/self.cfg.rewards.tracking_sigma)


    def _reward_imitation_lin_vel(self):
        index_array = self.imitation_index.detach().cpu().numpy().astype(int)
        # print(index_array)
        # Retrieve the corresponding rows from df_imit using array indexing
        lin_vel_imit_arr = df_imit.iloc[index_array,:3].to_numpy()

        lin_vel_imit_arr = lin_vel_imit_arr.reshape(self.num_envs, 3)
        lin_vel_imit_arr = torch.from_numpy(lin_vel_imit_arr).float().to(self.device)
        lin_imit_error = torch.sum(torch.abs(self.base_lin_vel - lin_vel_imit_arr), dim=1)    
        return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_imitation_ang_vel(self):
        index_array = self.imitation_index.detach().cpu().numpy().astype(int)
        # print(index_array)
        # Retrieve the corresponding rows from df_imit using array indexing
        ang_vel_imit_arr = df_imit.iloc[index_array,3:6].to_numpy()
        # print(ang_vel_imit_arr.shape)
        ang_vel_imit_arr = ang_vel_imit_arr.reshape(self.num_envs, 3)
        ang_vel_imit_arr = torch.from_numpy(ang_vel_imit_arr).float().to(self.device)
        lin_imit_error = torch.sum(torch.abs(self.base_ang_vel - ang_vel_imit_arr), dim=1)
        return torch.exp(-lin_imit_error/self.cfg.rewards.tracking_sigma)

    def _reward_torque_symmetry(self):
        torques_abd1 = torch.square(self.torques[:,0]-self.torques[:,3])
        torques_abd2 = torch.square(self.torques[:,3]-self.torques[:,6])
        torques_abd3 = torch.square(self.torques[:,6]-self.torques[:,9])

        torques_sh1 = torch.square(self.torques[:,1]-self.torques[:,4])
        torques_sh2 = torch.square(self.torques[:,4]-self.torques[:,7])
        torques_sh3 = torch.square(self.torques[:,7]-self.torques[:,10])

        torques_el1 = torch.square(self.torques[:,2]-self.torques[:,5])
        torques_el2 = torch.square(self.torques[:,5]-self.torques[:,8])
        torques_el3 = torch.square(self.torques[:,8]-self.torques[:,11])

        sum_sym_torq = torques_abd1 + torques_abd2 + torques_abd3 + torques_sh1 + torques_sh2 + torques_sh3 + torques_el1 + torques_el2 + torques_el3
        return sum_sym_torq
    
    def _reward_hip_nominal(self):
        hip1_nominal = torch.abs(self.dof_pos[:,0] - self.cfg.init_state.default_joint_angles['FL_hip_joint'])
        hip2_nominal = torch.abs(self.dof_pos[:,3] - self.cfg.init_state.default_joint_angles['RL_hip_joint'])
        hip3_nominal = torch.abs(self.dof_pos[:,6] - self.cfg.init_state.default_joint_angles['FR_hip_joint'])
        hip4_nominal = torch.abs(self.dof_pos[:,9] - self.cfg.init_state.default_joint_angles['RR_hip_joint'])

        return hip1_nominal + hip2_nominal + hip3_nominal + hip4_nominal 
    
    def _reward_thigh_nominal(self):
        thigh1_nominal = torch.abs(self.dof_pos[:,1] - self.cfg.init_state.default_joint_angles['FL_thigh_joint'])
        thigh2_nominal = torch.abs(self.dof_pos[:,4] - self.cfg.init_state.default_joint_angles['RL_thigh_joint'])
        thigh3_nominal = torch.abs(self.dof_pos[:,7] - self.cfg.init_state.default_joint_angles['FR_thigh_joint'])
        thigh4_nominal = torch.abs(self.dof_pos[:,10] - self.cfg.init_state.default_joint_angles['RR_thigh_joint'])

        return thigh1_nominal + thigh2_nominal + thigh3_nominal + thigh4_nominal