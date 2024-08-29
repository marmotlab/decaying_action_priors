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
Go1StartPos = [0, 0, 0.4]
Go1StartOrientation = p.getQuaternionFromEuler([0,0,0])
Go1 = p.loadURDF("/home/marmot/Sood/yuna_legged_gym/resources/robots/go1/urdf/go1.urdf",Go1StartPos, Go1StartOrientation)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

joint_num = p.getNumJoints(Go1)
print("JOINT NUM", joint_num )
actuators = [i for i in range(joint_num) if p.getJointInfo(Go1,i)[2] != p.JOINT_FIXED]
print(actuators)

# p.setRealTimeSimulation(1)
legs = [0.] * len(actuators)
# rest_position = np.array([0.1, 0.8, -1.5,
#                      0.1, 1.0, -1.5,
#                      -0.1, 0.8, -1.5,
#                      -0.1, 1.0, -1.5])

# rest_position = np.array([-0.4321209192276001, 1.1246285438537598, -2.7154908180236816, 
#                           0.4876498878002167, 1.11493980884552, -2.7254621982574463, 
#                           -0.4588862359523773, 1.0815739631652832, -2.6966781616210938, 
#                           0.4934026300907135, 1.1094292402267456, -2.7037835121154785])

rest_position = np.array([0.15, 0.85, -1.55,
                     0.15, 1.05, -1.55,
                     -0.15, 0.85, -1.55,
                     -0.15, 1.05, -1.55])
# forces = [40.] * len(actuators)
t=0
p.setJointMotorControlArray(Go1, actuators, controlMode=p.POSITION_CONTROL, targetPositions=rest_position, forces=[20.]*len(actuators))
     
def reorder_gym_to_pyb_go1(actions):
    actions_fin = np.zeros(12)
    actions_fin[0:3] = actions[3:6] 
    actions_fin[3:6] = actions[0:3]
    actions_fin[6:9] = actions[9:12]
    actions_fin[9:12] = actions[6:9]
    return actions_fin

Commands = np.array([0.2, 0.0, 0.0])      #Vx, Vy, Yaw

# actions_pyb = policy(obs_pyb.detach())
# prev_actions = actions_pyb.detach().cpu().numpy()[0]
# print("PREV", prev_actions)
# p.stepSimulation()
# for i in range(10*int(env.max_episode_length)):
df = pd.read_csv('imitation_data/imitation_data_wtw.csv', parse_dates=False)

while(1):
    p.stepSimulation()
    
    actions = policy(obs.detach())

    # actions_np[3:] = rest_position[3:]

    # obs, _, rews, dones, infos = env.step(actions.detach())
    # print("JOINT ZEROS", obs[:,12:])
    """
    Pybyllet Observations
    """
    # Go1Pos, Go1Orn = p.getBasePositionAndOrientation(Go1)
    # Go1Orn = np.array(Go1Orn, dtype='f')
    # Go1BaseLinVel, Go1BaseAngVel = np.array(p.getBaseVelocity(Go1)[0], dtype='f'), np.array(p.getBaseVelocity(Go1)[1],dtype='f')

    # if not np.any(np.isnan(Go1Orn)):
    #     RotMatrix = Rotation.from_quat(Go1Orn)
    # GravVec = np.matmul(RotMatrix.as_matrix(),np.array([0.0,0.0,-1.0]).T)

    
    # JointPos = p.getJointStates(Go1, actuators)
    # JointPos = np.array(JointPos, dtype=object)[:,0]
    # # JointPos = reorder_joint_pos(JointPos)
    # JointPos = JointPos - rest_position
    # # print(JointPos)
    # JointVel = p.getJointStates(Go1, actuators)
    # JointVel = np.array(JointVel, dtype=object)[:,1]
    # # JointVel = reorder_joint_pos(JointVel)

    # # # print(Go1BaseVel)
    # # # print(GravVec)

    # obs_joined = np.concatenate((Go1BaseLinVel, Go1BaseAngVel, GravVec, Commands, JointPos, JointVel, prev_actions))
    # # print(obs_joined.shape)
    # obs_joined = np.float32(obs_joined)
    # obs_joined = np.reshape(obs_joined,(1,48))
    # # print(obs_joined.shape)
    # obs_joined = torch.from_numpy(obs_joined)
    # obs_joined = obs_joined.to(device='cuda')
    # obs_pyb = obs_joined

    if t>200:
        # print("OG", obs)
        obs, _, rews, dones, infos = env.step(actions.detach())
        actions_pyb = policy(obs_pyb.detach())

        # print("OG Prev Actions", torch.round(obs[:,48:66], decimals=2))
        # print("Prev Actions", torch.round(obs_pyb[:,48:66], decimals=2))

        if t==201:   #First iteration of pybullet
            # print("ENTEREDDDDDD")
            # prev_actions = actions_pyb.detach().cpu().numpy()[0]
            #Set previous actions to 0
            prev_actions = np.zeros(12)
        """
        Pybyllet Observations
        """
        Go1Pos, Go1Orn = p.getBasePositionAndOrientation(Go1)
        Go1Orn = np.array(Go1Orn, dtype='f')
        Go1BaseLinVel, Go1BaseAngVel = np.array(p.getBaseVelocity(Go1)[0], dtype='f'), np.array(p.getBaseVelocity(Go1)[1],dtype='f')

        if not np.any(np.isnan(Go1Orn)):
            RotMatrix = Rotation.from_quat(Go1Orn)
        GravVec = np.matmul(RotMatrix.as_matrix(),np.array([0.0,0.0,-1.0]).T)

        
        JointPos = p.getJointStates(Go1, actuators)
        # print("JOINTPOS", JointPos.shape)
        JointPos = np.array(JointPos, dtype=object)[:,0]
        JointPos = JointPos - rest_position
        JointPos = reorder_gym_to_pyb_go1(JointPos)
        # print("BEFORE REORDER", JointPos)
        # JointPos = reorder_joint_pos(JointPos).copy()
        # print("AFTER REORDER", JointPos)
        # JointPos = JointPos - rest_position_gym
        # print(JointPos)
        JointVel = p.getJointStates(Go1, actuators)
        JointVel = np.array(JointVel, dtype=object)[:,1]
        # JointVel = reorder_joint_pos(JointVel).copy()

        # # print(Go1BaseVel)
        # # print(GravVec)

        # obs_joined = np.concatenate((Go1BaseLinVel*2.0, Go1BaseAngVel*0.25, GravVec, Commands*2.0, JointPos, JointVel*0.05, prev_actions))
        obs_joined = np.concatenate((GravVec, Commands*2.0, JointPos, JointVel*0.05, prev_actions))
        # print("OBS", obs_joined)
        # print(obs_joined.shape)
        obs_joined = np.float32(obs_joined)
        obs_joined = np.reshape(obs_joined,(1,42))
        # print(obs_joined.shape)
        obs_joined = torch.from_numpy(obs_joined)
        obs_joined = obs_joined.to(device='cuda')
        obs_pyb = obs_joined.detach().clone()


        # ACTIONS to np array for pybullet
        actions_np = actions_pyb.detach().cpu().numpy()

        # actions_np = actions.detach().cpu().numpy()     #Gym obs actions to pyb

        actions_np = actions_np*7.0
        # actions_np[:,[0,3,6,9]] *= 0.5 
        # actions_np = actions_np*0.25
        actions_np = reorder_gym_to_pyb_go1(actions_np[0])

        # actions_np = actions_np + rest_position

        # actions_np = reorder_gym_to_pyb_go1(actions_np[0])

        # actions_np = env.torques.detach().cpu().numpy()
        # actions_np = reorder_gym_to_pyb_go1(actions_np[0])

        # print("ACTIONS", actions_np)
        #FOR PD tracking

        #TODO for testing moving only one leg
        # actions_np[:6] = rest_position[:6]
        # actions_np[9:] = rest_position[9:]

        # for i in range(len(actuators)):
        #     # p.setJointMotorControl2(Go1, actuators[i], controlMode=p.POSITION_CONTROL, targetPosition=actions_np[i], 
        #     #                         targetVelocity = 0.0 ,positionGain=100,velocityGain=30,force=40, maxVelocity= 1.5)
        #     p.setJointMotorControl2(Go1, actuators[i], controlMode=p.POSITION_CONTROL, targetPosition=actions_imit[i], 
        #                             targetVelocity = 0.0 ,positionGain=100,velocityGain=30,force=40, maxVelocity= 1.5)
        # for i in range(len(actuators)):
        for i in range(12):
            p.setJointMotorControl2(Go1, actuators[i], controlMode=p.VELOCITY_CONTROL, force=0.0)

        decimation = 1
        for j in range(decimation):
            p.setJointMotorControlArray(Go1, actuators, controlMode=p.TORQUE_CONTROL, forces=actions_np)
        # if t>200 and t<203:
        #     print("OBS", obs_joined)
        #     print("TORQUES", actions_np)
        # print("Gym torques", env.torques)
        actions_imit = df.iloc[t-201,6:18].to_numpy()
        actions_imit = reorder_gym_to_pyb_go1(actions_imit)
        # actions_imit[3:] = rest_position[3:]
        # p.setJointMotorControlArray(Go1, actuators, controlMode=p.POSITION_CONTROL, targetPositions=actions_np,forces=[30.]*len(actuators))
        # p.setJointMotorControlArray(Go1, actuators, controlMode=p.POSITION_CONTROL, targetPositions=actions_imit,forces=[20.]*len(actuators))
        # print("Actions", actions_np[0])
        # print("TORQUES", p.getJointStates(Go1, actuators)[2])
        # p.setJointMotorControlArray(Go1, actuators, controlMode=p.POSITION_CONTROL, targetPositions=actions_np,
        #                             positionGains=[0.005]*len(actuators),velocityGains=[0.00]*len(actuators),forces=forces)
        prev_actions = actions_pyb.detach().cpu().numpy()[0]

        
    t+=1
    time.sleep(0.002)