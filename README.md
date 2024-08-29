# DecAP: Decaying Action Priors for Accelarated Learning of Torque-based Legged Locomotion Policies
The repositiry contains the research code for the IROS paper : [DecAP](https://arxiv.org/abs/2310.05714).

```
@article{Sood2023DecAPDA,
  title={DecAP: Decaying Action Priors for Accelerated Learning of Torque-Based Legged Locomotion Policies},
  author={Shivam Sood and Ge Sun and Peizhuo Li and Guillaume Sartoretti},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.05714},
  url={https://api.semanticscholar.org/CorpusID:263830010}
}
```
**Affiliation**: MARMoT Lab, NUS  
**Contact**: shivamsood2000@gmail.com

---
Follow [leggged_gym](https://github.com/leggedrobotics/legged_gym) repo for setting up the initial environment.
### Running the trained examples
The trained policies are in the logs folder. Each folder has torque policies trained with DecAP + Imitation and Imitation alone. The logs are named decap_[reward_scale_value] and imi_[reward_scale_value].

To run the torque policy set `control_type = T`, `action_scale = 8.0` in the robot config file. Then run the following command:<br/>
```console
cd decaying_action_priors
python legged_gym/scripts/play.py --task=go1_flat --load_run=decap_0.75x
```

### Training the torque policy 


Set the `action_scale = 8.0`

Imitation Rewards:
- Joint Angles
- End-effector position
- Foot height
- Base height

1. With imitation and DecAP:
    Set the `control_type`<br/>
    - For Go1 (Unitree): `T_ref_decay`<br/>
    - For Yuna (Hebi-Daisy)     : `T_vanish_yuna`<br/>
    - For Cassie (Agility)  : `T_vanish_humanoid`<br/>

    Uncomment the imitation rewards mentioned above.<br/>
    Imitation Reward scales tested: 0.75, 1.5, 7.5, 15.0

2. With just imitation data:<br/>
    Set the `control_type = T`

    Uncomment the imitation rewards mentioned

### Getting the imitation data:
Train a position policy/ use any position policy or Optimal Controller to get imitation data

Trained position policies are in the logs folder and can also be used to collect the imitation data. For running these position polcies set `action_scale = 0.25` and `control_type = P`. The imitation data folder contains the data used. <br/>
The imitation rewards are defined in the respective robot files in the envs folder. Decaying Action Priors are set in `compute_torques` function in legged_robot.py








