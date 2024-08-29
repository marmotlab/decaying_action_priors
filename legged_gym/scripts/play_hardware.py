
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
# import hebi

EXPORT_POLICY = True
args = get_args()

env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
# override some parameters for testing
env_cfg.env.num_envs = 1
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()
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


for i in range(10*int(env.max_episode_length)):
    actions = policy(obs.detach())
    # print(actions.shape)
    obs, _, rews, dones, infos = env.step(actions.detach())
    
    obs[:,11] = 0.0
    obs[:,9] = 0.4
    obs[:,10] = 0.0
    time.sleep(0.1)
    # obs[:,8] = -9.81
    print("OBS_grav",torch.round(obs[:,12:30], decimals=2))
    # print(obs[:,9:12])       
        # print("OBS",obs[:])
