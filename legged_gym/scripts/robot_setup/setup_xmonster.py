from robot_setup.setup_modules import *
from robot_setup.yunaKinematics import *
import numpy as np
import hebi



def setup_xmonster():

	print('Setting up Snake Monster...')
	# Establish connection with the modules and Kinematics Object
	imu, hexapod = setup_modules()
	xmk = HexapodKinematics()

	

	#Initialize Feedback Structs for both groups
	fbk_imu = hebi.GroupFeedback(imu.size)
	fbk_imu = imu.get_next_feedback(reuse_fbk=fbk_imu)
	while fbk_imu == None:
		fbk_imu = imu.get_next_feedback(reuse_fbk=fbk_imu)

	print('fbk_imu structure created')

	

	fbk_hp = hebi.GroupFeedback(hexapod.size)
	print('fbk_sm structure created')
	fbk_hp = hexapod.get_next_feedback(reuse_fbk=fbk_hp)

	#CF.update(fbk_imu)
	#CF.updateFilter(fbk_imu)


	print('Setup complete!')

	#print('Initial pose:', CF.R)



	return xmk, imu , hexapod, fbk_imu, fbk_hp


if __name__ == '__main__':
	setup_xmonster()