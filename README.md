# DecAP
The repositiry contains the research code for the IROS paper : [DecAP] : Decaying Action Priors for Accelarated Learning of Torque-based Legged Locomotion Policies(https://arxiv.org/abs/2310.05714).


**Affiliation**: MARMoT Lab, NUS  
**Contact**: shivamsood2000@gmail.com

---

## Training in Simulation

### Installation
The following commands are to setup [leggged_gym](https://github.com/leggedrobotics/legged_gym) environment:

1. Create a python virtual environment with python=3.8
   
   - `conda create --name=decap python=3.8`
   - `conda activate decap`

2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 

5. Once isaac gym and rsl_rl is setup clone and install our repository:

   - `git clone https://github.com/marmotlab/decaying_action_priors.git`
   - `cd decaying_action_priors && pip install -e .`


### Running the trained examples
The trained policies are in the logs folder. Each folder has torque policies trained with DecAP + Imitation and Imitation alone. The logs are named decap_[reward_scale_value] and imi_[reward_scale_value].

To run the torque policy set `control_type = torques`, `action_scale = 8.0` in the config file. Then run the following command:<br/>
```console
python legged_gym/scripts/play.py --task=go1_flat --load_run=decap_0.75x
```
   - Different robots and polcies can be used for comparison

### Training your own policies

The parameters for DecAP can be changed inside 'legged_gym/envs/param_config.yaml':
   - `control_type` dictates whether the policy would be trained using torques, position or DecAP. By default it is set to `decap_torques` which trains torques using DecAP. For inference of the trained policy (playing the policy) change this to `torques`
   - `gamma` and `k` for DecAP (refer the paper) can also be varied here

To train the policies run the following command inside decaying_action_priors folder:

```
python legged_gym/scripts/train.py --task={task_name}
```
   - Tasks available with DecAP for now are `go1_flat`, `cassie` and `yuna`

Imitation Rewards used (inside corresponding {robot}_config files):
- Joint Angles
- End-effector position
- Foot height
- Base height


Imitation Reward scales tested: 0.75, 1.5, 7.5, 15.0. In order to train using just imitation without DecAP just set `control_type = torques`.




### Getting the imitation data:
You can train your own position policy/ use any position policy or Optimal Controller to get imitation data

The imitation data folder contains the data used. <br/>
The imitation rewards are defined in the respective robot files in the envs folder and the decaying action priors are set in `compute_torques` function in the same files.

## Upcoming developments:
- [x] Move DecAP params to yaml
- [ ] Go2 support
- [ ] Sim-to-Sim Mujoco support
- [ ] Code for hardware deployment
## Credit
If you find this work useful, please consider citing us and the following works:
```bibtex
@article{Sood2023DecAPD,
  title={DecAP : Decaying Action Priors for Accelerated Imitation Learning of Torque-Based Legged Locomotion Policies},
  author={Shivam Sood and Ge Sun and Peizhuo Li and Guillaume Sartoretti},
  journal={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023},
  pages={2809-2815},
  url={https://api.semanticscholar.org/CorpusID:263830010}
}
```

We used the codebase from [Legged Gym](https://github.com/leggedrobotics/legged_gym) and [RSL RL](https://github.com/leggedrobotics/rsl_rl):
  + Rudin, Nikita, et al. "Learning to walk in minutes using massively parallel deep reinforcement learning." CoRL 2022.



