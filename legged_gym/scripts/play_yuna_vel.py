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
import robot_setup
from robot_setup.yunaKinematics import *
import hebi

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

xmk, imu, hexapod, fbk_imu, fbk_hp = robot_setup.setup_xmonster()
group_command = hebi.GroupCommand(hexapod.size)
hexapod.command_lifetime = 0
group_feedback = hebi.GroupFeedback(hexapod.size)

rest_vel_gym = np.array([0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0,
                     0, 0, 0])

rest_position_gym = np.array([0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, -1.57,
                     0, 0, 1.57])

# load policy
train_cfg.runner.resume = True
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = ppo_runner.get_inference_policy(device=env.device)

# export policy as a jit module (used to run it from C++)
if EXPORT_POLICY:
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)

group_command.position = rest_position_gym
hexapod.send_command(group_command)
time.sleep(1)

#For effort 5.0, for position 0.25
action_scale = 0.35

while(1):

    # time_old = time.time()
    actions = policy(obs.detach())

    
    obs, _, rews, dones, infos = env.step(actions.detach())
    obs[:,9] = 0.3   #vel_x
    obs[:,10] = 0.0   #vel_y
    obs[:,11] = 0.0   #ang_vel
    
    #ACTIONS to np array for pybullet

    actions_np = actions.detach().cpu().numpy()     #Gym obs actions to pyb
    # actions_np = actions_pyb.detach().cpu().numpy()     #Pyb obs actions to pyb

    actions_np = actions_np*action_scale
    actions_np - np.clip(actions_np, -2, 2)

    #TODO for testing moving only one leg
    # print("ACTIONS", actions_np.shape)
    # print(actions_np)
    # joint_pos = obs[:,12:30].detach().cpu().numpy()[0]
    
    actions_np = np.reshape(actions_np, (18,))
    # print(actions_np - joint_pos - rest_position_gym)
    # actions_np[3:] = rest_vel_gym[3:]
    # actions_np[9:] = rest_position[9:]

    print(actions_np)
    # group_feedback = hexapod.get_next_feedback(reuse_fbk=group_feedback)
    # print("EFFORT",group_feedback.effort)

    group_command.velocity = actions_np
    hexapod.send_command(group_command)

    # print("OG", torch.round(obs[:,48:], decimals=1))
    # print(torch.round(obs_pyb[:,48:], decimals=1))
    # print("OG OBS_lin_vel", torch.round(obs[:,:3], decimals=2))
    # print("OBS_lin_vel", torch.round(obs_pyb[:,:3], decimals=2))


    # print("OG OBS_ang_vel", torch.round(obs[:,3:6], decimals=2))
    # print("OBS_ang_vel", torch.round(obs_pyb[:,3:6], decimals=2))

    # print("OG grav_vec", torch.round(obs[:,6:9], decimals=2))
    # print("grav_vec", torch.round(obs_pyb[:,6:9], decimals=2))

    # print("comm", torch.round(obs_pyb[:,9:12], decimals=2))

    # print("OG joint_pos", torch.round(obs[:,12:30], decimals=2))
    # print("joint_pos", torch.round(obs_pyb[:,12:30], decimals=2))

    # print("OG joint_vel", torch.round(obs[:,30:48], decimals=2))
    # print("joint_vel", torch.round(obs_pyb[:,30:48], decimals=2))

    # print("OG Actions by policy", np.round(actions.detach().cpu().numpy()[0], decimals=2))
    # print("Actions by policy", np.round(actions_pyb.detach().cpu().numpy()[0], decimals=2))
    # print("Actions to pyb", np.round(actions_np, decimals=2))
    print("*****************************************************")   



    # t+=1
    # time.sleep(0.02)

