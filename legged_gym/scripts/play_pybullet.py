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

#Pybullet Imports
import pybullet as p
import pybullet_data
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
# import hebi
import pandas as pd

EXPORT_POLICY = True
args = get_args()

env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
# override some parameters for testing
env_cfg.env.num_envs = 1
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False
env_cfg.env.episode_length_s = 120

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

rest_position_gym = np.array([0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, -1.57,
                     0, 0, 1.57])
#TODO change pos to 1.57
# obs_pyb = env.get_observations()
obs_pyb = obs.clone()
# obs_pyb = np.zeros((1,66), dtype='f')
# obs_pyb = torch.from_numpy(obs_pyb)
# obs_pyb = obs_pyb.to(device='cuda')
# rest_position_gym = torch.from_numpy(rest_position_gym)
# rest_position_gym = rest_position_gym.to(device='cuda')
# obs_pyb[:, 12:30] = rest_position_gym[:]
# print(obs)

# load policy
train_cfg.runner.resume = True
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = ppo_runner.get_inference_policy(device=env.device)

# export policy as a jit module (used to run it from C++)
if EXPORT_POLICY:
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# set visualization
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load world

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.81)

#Changing the init z pos from 0.5 to 0.2 so that it doesn't come in flying
YunaStartPos = [0,0,0.2]
YunaStartOrientation = p.getQuaternionFromEuler([0,0,0])
Yuna = p.loadURDF("/home/marmot/Sood/yuna_legged_gym/resources/robots/yuna/urdf/yuna.urdf",YunaStartPos, YunaStartOrientation)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

joint_num = p.getNumJoints(Yuna)
print("JOINT NUM", joint_num )
actuators = [i for i in range(joint_num) if p.getJointInfo(Yuna,i)[2] != p.JOINT_FIXED]

# p.setRealTimeSimulation(1)
legs = [0.] * len(actuators)
rest_position = np.array([0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, 1.57,
                     0, 0, 1.57])



forces = [40.] * len(actuators)
t=0
p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=rest_position,
                                positionGains=[0.5]*len(actuators),velocityGains=[1]*len(actuators),forces=forces)

def gym_to_pyb_act(actions):
    actions_fin = np.zeros(18)
    actions_fin[0:3] = actions[0:3]  # 1 to 1
    actions_fin[9:12] = actions[3:6]  # 3 to 2
    actions_fin[3:6] = actions[6:9]  # 5 to 3
    actions_fin[12:15] = actions[9:12]  # 2 to 4
    actions_fin[6:9] = actions[12:15]  # 5 to 3
    actions_fin[15:18] = actions[15:18]  # 6 to 6
    return actions_fin

def reorder_joint_pos(JointInfo):
    JointVal = np.zeros(18)
    JointVal[0:3] = JointInfo[0:3]  # 1 to 1
    JointVal[3:6] = JointInfo[9:12]  # 3 to 2
    JointVal[6:9] = JointInfo[3:6]  # 5 to 3
    JointVal[9:12] = JointInfo[12:15]  # 2 to 4
    JointVal[12:15] = JointInfo[6:9]  # 5 to 3
    JointVal[15:18] = JointInfo[15:18]  # 6 to 6
    return JointVal
     

Commands = np.array([0.4, 0.0, 0.0])      #Vx, Vy, Yaw

# actions_pyb = policy(obs_pyb.detach())
# prev_actions = actions_pyb.detach().cpu().numpy()[0]
# print("PREV", prev_actions)
# p.stepSimulation()
# for i in range(10*int(env.max_episode_length)):
df = pd.read_csv('imitation_data/imitation_data_yuna_torques.csv', parse_dates=False)

while(1):
    p.stepSimulation()
    
    actions = policy(obs.detach())

    # actions_np[3:] = rest_position[3:]

    # obs, _, rews, dones, infos = env.step(actions.detach())
    # print("JOINT ZEROS", obs[:,12:])
    """
    Pybyllet Observations
    """
    # YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
    # YunaOrn = np.array(YunaOrn, dtype='f')
    # YunaBaseLinVel, YunaBaseAngVel = np.array(p.getBaseVelocity(Yuna)[0], dtype='f'), np.array(p.getBaseVelocity(Yuna)[1],dtype='f')

    # if not np.any(np.isnan(YunaOrn)):
    #     RotMatrix = Rotation.from_quat(YunaOrn)
    # GravVec = np.matmul(RotMatrix.as_matrix(),np.array([0.0,0.0,-1.0]).T)

    
    # JointPos = p.getJointStates(Yuna, actuators)
    # JointPos = np.array(JointPos, dtype=object)[:,0]
    # JointPos = reorder_joint_pos(JointPos)
    # JointPos = JointPos - rest_position_gym
    # # print(JointPos)
    # JointVel = p.getJointStates(Yuna, actuators)
    # JointVel = np.array(JointVel, dtype=object)[:,1]
    # JointVel = reorder_joint_pos(JointVel)

    # # print(YunaBaseVel)
    # # print(GravVec)

    # obs_joined = np.concatenate((YunaBaseLinVel, YunaBaseAngVel, GravVec, Commands, JointPos, JointVel, prev_actions))
    # # print(obs_joined.shape)
    # obs_joined = np.float32(obs_joined)
    # obs_joined = np.reshape(obs_joined,(1,66))
    # # print(obs_joined.shape)
    # obs_joined = torch.from_numpy(obs_joined)
    # obs_joined = obs_joined.to(device='cuda')
    # obs_pyb = obs_joined

    if t>200:
        obs, _, rews, dones, infos = env.step(actions.detach())
        actions_pyb = policy(obs_pyb.detach())

        # print("OG Prev Actions", torch.round(obs[:,48:66], decimals=2))
        # print("Prev Actions", torch.round(obs_pyb[:,48:66], decimals=2))

        if t==201:   #First iteration of pybullet
            # print("ENTEREDDDDDD")
            prev_actions = actions_pyb.detach().cpu().numpy()[0]
        
        """
        Pybyllet Observations
        """
        YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
        YunaOrn = np.array(YunaOrn, dtype='f')
        YunaBaseLinVel, YunaBaseAngVel = np.array(p.getBaseVelocity(Yuna)[0], dtype='f'), np.array(p.getBaseVelocity(Yuna)[1],dtype='f')

        if not np.any(np.isnan(YunaOrn)):
            RotMatrix = Rotation.from_quat(YunaOrn)
        GravVec = np.matmul(RotMatrix.as_matrix(),np.array([0.0,0.0,-1.0]).T)

        
        JointPos = p.getJointStates(Yuna, actuators)
        # print("JOINTPOS", JointPos.shape)
        JointPos = np.array(JointPos, dtype=object)[:,0]
        JointPos = JointPos - rest_position
        # print("BEFORE REORDER", JointPos)
        JointPos = reorder_joint_pos(JointPos).copy()
        # print("AFTER REORDER", JointPos)
        # JointPos = JointPos - rest_position_gym
        # print(JointPos)
        JointVel = p.getJointStates(Yuna, actuators)
        JointVel = np.array(JointVel, dtype=object)[:,1]
        JointVel = reorder_joint_pos(JointVel).copy()

        # print(YunaBaseVel)
        # print(GravVec)

        obs_joined = np.concatenate((YunaBaseLinVel, YunaBaseAngVel, GravVec, Commands, JointPos, JointVel, prev_actions))
        # print(obs_joined.shape)
        obs_joined = np.float32(obs_joined)
        obs_joined = np.reshape(obs_joined,(1,66))
        # print(obs_joined.shape)
        obs_joined = torch.from_numpy(obs_joined)
        obs_joined = obs_joined.to(device='cuda')
        obs_pyb = obs_joined.detach().clone()

        #ACTIONS to np array for pybullet
        # actions_np = actions_pyb.detach().cpu().numpy()

        actions_np = actions.detach().cpu().numpy()     #Gym obs actions to pyb
        # actions_np = actions_pyb.detach().cpu().numpy()     #Pyb obs actions to pyb
        actions_np = gym_to_pyb_act(actions_np[0])

        actions_np = actions_np*0.25
        #FOR PD tracking
        actions_np = actions_np + rest_position

        actions_imit = df.iloc[t-201,0:18].to_numpy()
        actions_imit = gym_to_pyb_act(actions_imit)
        #TODO for testing moving only one leg
        # actions_np[:6] = rest_position[:6]
        # actions_np[9:] = rest_position[9:]

        # for i in range(len(actuators)):
        #     # p.setJointMotorControl2(Yuna, actuators[i], controlMode=p.POSITION_CONTROL, targetPosition=actions_np[i], 
        #     #                         targetVelocity = 0.0 ,positionGain=100,velocityGain=30,force=40, maxVelocity= 1.5)
        #     p.setJointMotorControl2(Yuna, actuators[i], controlMode=p.POSITION_CONTROL, targetPosition=actions_imit[i], 
        #                             targetVelocity = 0.0 ,positionGain=100,velocityGain=30,force=40, maxVelocity= 1.5)
        p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=actions_imit)
        # p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=actions_np,
        #                             positionGains=[0.005]*len(actuators),velocityGains=[0.00]*len(actuators),forces=forces)
        prev_actions = actions_pyb.detach().cpu().numpy()[0]


        # print("OG", torch.round(obs[:,48:], decimals=1))
        # print(torch.round(obs_pyb[:,48:], decimals=1))
        # print("OG OBS_lin_vel", torch.round(obs[:,:3], decimals=2))
        # print("OBS_lin_vel", torch.round(obs_pyb[:,:3], decimals=2))


        # print("OG OBS_ang_vel", torch.round(obs[:,3:6], decimals=2))
        # print("OBS_ang_vel", torch.round(obs_pyb[:,3:6], decimals=2))

        # print("OG grav_vec", torch.round(obs[:,6:9], decimals=2))
        # print("grav_vec", torch.round(obs_pyb[:,6:9], decimals=2))

        # # print("comm", torch.round(obs_pyb[:,9:12], decimals=2))

        # print("OG joint_pos", torch.round(obs[:,12:30], decimals=2))
        # print("joint_pos", torch.round(obs_pyb[:,12:30], decimals=2))

        # print("OG joint_vel", torch.round(obs[:,30:48], decimals=2))
        # print("joint_vel", torch.round(obs_pyb[:,30:48], decimals=2))

        # print("OG Actions by policy", np.round(actions.detach().cpu().numpy()[0], decimals=2))
        # print("Actions by policy", np.round(actions_pyb.detach().cpu().numpy()[0], decimals=2))
        # print("Actions to pyb", np.round(actions_np, decimals=2))
        print("*****************************************************")
        # plt.plot(actions_np)
        # plt.show()
        # print("JO")
        


    obs[:,9] = 0.4   #vel_x
    obs[:,10] = 0.0   #vel_y
    obs[:,11] = 0.0   #ang_vel
    t+=1
    time.sleep(0.04)


