
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import yaml
import pandas as pd
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, exp_avg_filter

with open(f"legged_gym/envs/param_config.yaml", "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
	gamma_decap = config["gamma"]
	k_decap = config["k"]
	# visualize_imitation_data = config["visualize_imitation_data"]
	path_to_imitation_data = config["path_to_imitation_data"]

class H1Robot(LeggedRobot):
	
	def _get_noise_scale_vec(self, cfg):
		""" Sets a vector used to scale the noise added to the observations.
			[NOTE]: Must be adapted when changing the observations structure

		Args:
			cfg (Dict): Environment config file

		Returns:
			[torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
		"""
		noise_vec = torch.zeros_like(self.obs_buf[0])
		self.add_noise = self.cfg.noise.add_noise
		noise_scales = self.cfg.noise.noise_scales
		noise_level = self.cfg.noise.noise_level
		noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
		noise_vec[3:6] = noise_scales.gravity * noise_level
		noise_vec[6:9] = 0. # commands
		noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
		noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
		noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
		noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
		
		return noise_vec

	def _init_foot(self):
		self.feet_num = len(self.feet_indices)
		
		rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
		self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
		self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
		self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
		self.feet_pos = self.feet_state[:, :, :3]
		self.feet_vel = self.feet_state[:, :, 7:10]
		
	def _init_buffers(self):
		super()._init_buffers()
		self._init_foot()

	def update_feet_state(self):
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		
		self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
		self.feet_pos = self.feet_state[:, :, :3]
		self.feet_vel = self.feet_state[:, :, 7:10]
		
	def _post_physics_step_callback(self):
		self.update_feet_state()

		period = 0.8
		offset = 0.5
		self.phase = (self.episode_length_buf * self.dt) % period / period
		self.phase_left = self.phase
		self.phase_right = (self.phase + offset) % 1
		self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
		
		return super()._post_physics_step_callback()
	
	def _compute_torques(self, actions):
		actions_scaled = actions * self.cfg.control.action_scale
		actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
		
		if self.cfg.domain_rand.randomize_lag_timesteps:
			self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
			self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
		else:
			self.joint_pos_target = actions_scaled + self.default_dof_pos

		control_type = self.cfg.control.control_type

		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = self.df_imit.iloc[index_array,0:10].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)
		# Convert the array to a PyTorch tensor
		dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)

		if control_type == 'decap_torque':
			#Calculate the decay factor
			decap_factor = gamma_decap**(self.torque_ref_decay_factor/k_decap)
			#Add the imitation bias along with the decap factor
			torques = actions_scaled + decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
			#Test with one increasing and the other decreasing
			# torques = (1-decap_factor)*actions_scaled + decap_factor*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)

		elif control_type=="position":
			torques = self.p_gains* self.Kp_factors*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains*self.Kd_factors*self.dof_vel

		elif control_type == 'torque':
			torques = actions_scaled
		
		elif control_type == 'decap_position':
			torques = self.p_gains*(self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains*self.dof_vel + gamma_decap**(self.torque_ref_decay_factor/k_decap)*(self.p_gains*(dof_imit_arr - self.dof_pos)- self.d_gains*self.dof_vel)
		
		else:
			raise ValueError(f"Unknown control type: {control_type}")
		return torch.clip(torques, -self.torque_limits, self.torque_limits)
	
	def compute_observations(self):
		""" Computes observations
		"""
		sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
		cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
		self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
									self.projected_gravity,
									self.commands[:, :3] * self.commands_scale,
									(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
									self.dof_vel * self.obs_scales.dof_vel,
									self.actions,
									sin_phase,
									cos_phase
									),dim=-1)
		self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
									self.base_ang_vel  * self.obs_scales.ang_vel,
									self.projected_gravity,
									self.commands[:, :3] * self.commands_scale,
									(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
									self.dof_vel * self.obs_scales.dof_vel,
									self.actions,
									sin_phase,
									cos_phase
									),dim=-1)
		# add perceptive inputs if not blind
		# add noise if needed
		if self.add_noise:
			self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

		
	def _reward_contact(self):
		res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
		for i in range(self.feet_num):
			is_stance = self.leg_phase[:, i] < 0.55
			contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
			res += ~(contact ^ is_stance)
		return res
	
	def _reward_feet_swing_height(self):
		contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
		pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
		return torch.sum(pos_error, dim=(1))
	
	def _reward_alive(self):
		# Reward for staying alive
		return 1.0
	
	def _reward_contact_no_vel(self):
		# Penalize contact with no velocity
		contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
		contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
		penalize = torch.square(contact_feet_vel[:, :, :3])
		return torch.sum(penalize, dim=(1,2))
	
	def _reward_hip_pos(self):
		return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)
	
	def _reward_imitation_angles(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# Retrieve the corresponding rows from df_imit using array indexing
		dof_imit_arr = self.df_imit.iloc[index_array,0:10].to_numpy()
		# Reshape the array to the desired shape
		dof_imit_arr = dof_imit_arr.reshape(self.num_envs, self.num_actions)

		# Convert the array to a PyTorch tensor
		dof_imit_arr = torch.from_numpy(dof_imit_arr).float().to(self.device)
		dof_imit_error = torch.sum(torch.square(self.dof_pos - dof_imit_arr)*self.obs_scales.dof_imit, dim=1)  
		return torch.exp(-10*dof_imit_error)  
	
	def _reward_imitation_height_penalty(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		# print(index_array)
		# Retrieve the corresponding rows from df_imit using array indexing
		height = self.df_imit.iloc[index_array,10].to_numpy()
		# print("HEIGHT REFERENCE", height)
		base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
		height = height.reshape(self.num_envs, )
		height = torch.from_numpy(height).float().to(self.device)

		#Plot imitation height with draw_sphere:
		# if visualize_imitation_data:
		# self.clear_lines()
		sphere_radius = 0.07  # Adjust as needed
		sphere_color = (0.1, 0.8, 0.07)  # Green for target positions

		for env_id in range(1):
			# Extract robot's (x, y) position from root_states
			base_x = self.root_states[env_id, 0].item()
			base_y = self.root_states[env_id, 1].item()
			pos_world_frame = [base_x, base_y, height[env_id].item()]
			self.draw_sphere(pos_world_frame, sphere_radius, sphere_color, env_id)

		return torch.square(base_height - height)
	
	def _reward_imitate_end_effector_pos(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_ref = self.df_imit.iloc[index_array,11:17].to_numpy()
		end_effector_ref = end_effector_ref.reshape(self.num_envs, 6)
		end_effector_ref = torch.from_numpy(end_effector_ref).float().to(self.device)

		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		footsteps_in_body_frame = torch.zeros(self.num_envs, 2, 3, device=self.device)
		for i in range(2):
			footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
															  cur_footsteps_translated[:, i, :])
		footsteps_in_body_frame = footsteps_in_body_frame.reshape(self.num_envs, 6)

		end_effector_error = torch.sum(torch.square(end_effector_ref - footsteps_in_body_frame), dim=1)
		
		# if visualize_imitation_data:
		#Clear lines after 10 steps
		self.clear_lines()
		sphere_radius = 0.03  # Adjust as needed
		sphere_color = (0.1, 0.8, 0.07)  # Green for target positions

		for env_id in range(1):
			for i in range(2):
				# Extract reference end-effector position (body frame)
				pos_body_frame = end_effector_ref[env_id, i * 3: (i + 1) * 3]

				# Transform to world frame
				pos_world_frame = quat_apply_yaw(self.base_quat[env_id], pos_body_frame) + self.base_pos[env_id]

				# Convert to numpy and visualize
				self.draw_sphere(pos_world_frame.cpu().numpy(), sphere_radius, sphere_color, env_id)

		return torch.exp(-40*end_effector_error)

	def _reward_imitate_foot_height(self):
		index_array = self.imitation_index.detach().cpu().numpy().astype(int)
		end_effector_z1_ref = self.df_imit.iloc[index_array,11].to_numpy()
		end_effector_z2_ref = self.df_imit.iloc[index_array,16].to_numpy()

		end_effector_z1_ref = end_effector_z1_ref.reshape(self.num_envs, 1)
		end_effector_z2_ref = end_effector_z2_ref.reshape(self.num_envs, 1)

		end_effector_z1_ref = torch.from_numpy(end_effector_z1_ref).float().to(self.device)
		end_effector_z2_ref = torch.from_numpy(end_effector_z2_ref).float().to(self.device)
		# print("END_EFFECTOR_Z1_REF", end_effector_z1_ref[0])
		# print("END_EFFECTOR_Z2_REF", end_effector_z2_ref[0])
		cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
		cur_foot_pos = cur_footsteps_translated.reshape(self.num_envs, 6)
		# print("CUR_FOOT_POS", cur_foot_pos[0])

		foot_height_error = torch.sum(torch.square(end_effector_z1_ref - cur_foot_pos[:,2].unsqueeze(1)), dim=1) + torch.sum(torch.square(end_effector_z2_ref - cur_foot_pos[:,5].unsqueeze(1)), dim=1) 
		return torch.exp(-40*foot_height_error)
	
	def _reward_contact(self):
		res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
		for i in range(self.feet_num):
			is_stance = self.leg_phase[:, i] < 0.55
			contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
			res += ~(contact ^ is_stance)
		return res
	
	def _reward_feet_swing_height(self):
		contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
		pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
		return torch.sum(pos_error, dim=(1))
	
	def _reward_alive(self):
		# Reward for staying alive
		#Return 1.0 but as a tensor
		return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
	
	def _reward_contact_no_vel(self):
		# Penalize contact with no velocity
		contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
		contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
		penalize = torch.square(contact_feet_vel[:, :, :3])
		return torch.sum(penalize, dim=(1,2))
	
	def _reward_hip_pos(self):
		return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)
	
	def _reward_feet_slip(self):
		contact = self.contact_forces[:, self.feet_indices, 2] > 1.
		contact_filt = torch.logical_or(contact, self.last_contacts)
		self.last_contacts = contact
		foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
		rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
		return rew_slip
	'''
	For visualzing the imitation foot points
	'''
	# debug visualization
	def clear_lines(self):
		self.gym.clear_lines(self.viewer)

	def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
		sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
		sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
		gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)

	def draw_line(self, start_point, end_point, color, env_id):
		gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.envs[env_id])