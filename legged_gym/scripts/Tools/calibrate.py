# bebin size e offseta chie va inke chetori save mishe mat file ha tahe file
# emkan ddare bata look up tools.look_up bashe

# X Monster Torque Modules Definition (intermediate modules of each leg)
import numpy as np
import hebi
from time import sleep, time

# import tools

pi = np.pi


names=[]
for i in range(0,6):
    names.append('base'+ str(i+1))
    names.append('shoulder'+ str(i+1))
    names.append('elbow'+ str(i+1))

names = np.array(names)

np.save("names",names)

base = names[range(0,len(names),3)]

Hebi = hebi.Lookup()

imu = Hebi.get_group_from_names("*", names)

if imu is None:
	print("Group 'imu' not found! Check that the family and name of a module on the network")
	print('matches what is given in the source file.')
	exit(1)

imu.feedback_frequency = 5.0

group_feedback = hebi.GroupFeedback(imu.size)

while True:
	feedback = imu.get_next_feedback(reuse_fbk=group_feedback)
	if feedback is not None:
		gyros = group_feedback.gyro
		accels = group_feedback.accelerometer
		efforts = group_feedback.effort
		break


gyros = gyros[range(0,len(names),3)]
accels = accels[range(0,len(names),3)]


# gyroOffsets = [[np.mean(gyros[:,0])], [np.mean(gyros[:,1])], [np.mean(gyros[:,2])]]
# accelOffsets =[[np.mean(accels[:,0])], [np.mean(accels[:,1])], [np.mean(accels[:,2])]]
# efforts = np.reshape(efforts,[6,3])


np.savetxt("gyroOffset.txt",gyros.T)
np.savetxt("accelOffset.txt",accels.T)
np.savetxt("torqueOffsets.txt",efforts)