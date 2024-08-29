import hebi
import numpy as np
from math import pi, cos, sin


def trans(xyzrpy):
    m = np.identity(4)
    m[0:3, 3] = xyzrpy[0:3]
    m = np.matmul(m, rotz(xyzrpy[5]))
    m = np.matmul(m, roty(xyzrpy[4]))
    m = np.matmul(m, rotx(xyzrpy[3]))
    return m


def rotx(theta):
    return np.array([[1, 0, 0, 0],
                     [0, cos(theta), -sin(theta), 0],
                     [0, sin(theta), cos(theta), 0],
                     [0, 0, 0, 1]])


def roty(theta):
    # Homogeneous transform matrix for a rotation about y
    return np.array([[cos(theta), 0, sin(theta), 0],
                     [0, 1, 0, 0],
                     [-sin(theta), 0, cos(theta), 0],
                     [0, 0, 0, 1]])


def rotz(theta):
    return np.array([[cos(theta), -sin(theta), 0, 0],
                     [sin(theta), cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


from numpy import *


def svd_transform(A, B):
    assert len(A) == len(B)
    # Inumpyut: expects Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    N = A.shape[0];  # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(transpose(AA), BB)

    U, S, Vt = linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A.T) + centroid_B.T

    H1 = np.zeros((4, 4))
    H1[0:3, 0:3] = R
    H1[0:3, 3] = t
    H1[3, 3] = 1

    return H1


class HexapodKinematics(object):
    def __init__(self):

        short_legs = 0.1875
        long_legs = 0.2375

        self.lfLeg = self.getLeg('left')
        self.rfLeg = self.getLeg('right')
        self.lmLeg = self.getLeg('left')
        self.rmLeg = self.getLeg('right')
        self.lbLeg = self.getLeg('left')
        self.rbLeg = self.getLeg('right')

        self.rfLeg.base_frame = np.dot(rotz(-pi / 6), trans([long_legs, 0, 0, 0, 0, 0]))
        self.rmLeg.base_frame = np.dot(rotz(-pi / 2), trans([short_legs, 0, 0, 0, 0, 0]))
        self.rbLeg.base_frame = np.dot(rotz(-5 * pi / 6), trans([long_legs, 0, 0, 0, 0, 0]))
        self.lfLeg.base_frame = np.dot(rotz(pi / 6), trans([long_legs, 0, 0, 0, 0, 0]))
        self.lmLeg.base_frame = np.dot(rotz(pi / 2), trans([short_legs, 0, 0, 0, 0, 0]))
        self.lbLeg.base_frame = np.dot(rotz(5 * pi / 6), trans([long_legs, 0, 0, 0, 0, 0]))

        self.group_fb1 = hebi.GroupFeedback(self.lfLeg.dof_count)
        self.group_fb2 = hebi.GroupFeedback(self.rfLeg.dof_count)
        self.group_fb3 = hebi.GroupFeedback(self.lmLeg.dof_count)
        self.group_fb4 = hebi.GroupFeedback(self.rmLeg.dof_count)
        self.group_fb5 = hebi.GroupFeedback(self.lbLeg.dof_count)
        self.group_fb6 = hebi.GroupFeedback(self.rbLeg.dof_count)

        # need to set world base frame
        self.robot_base = np.identity(4)

    def getLeg(self, side):

        if side == 'right':
            mount_side = 'right-inside'
        else:
            mount_side = 'left-inside'

        kin = hebi.robot_model.RobotModel()

        kin.add_actuator('X8-9')
        kin.add_bracket('X5-HeavyBracket', mount_side)
        kin.add_actuator('X8-16')
        kin.add_link('X5', 0.325, pi)
        kin.add_actuator('X8-9')
        kin.add_link('X5', 0.325, 0)
        kin.add_end_effector('custom')

        return kin

    def getFramesNum(self):
        # Return the num of the frame of each leg
        # for the frame type 'output' and 'com' is same for the end-effector

        Frames_num = []

        Frames_num.append(self.lfLeg.get_frame_count('output'))
        Frames_num.append(self.rfLeg.get_frame_count('output'))
        Frames_num.append(self.lmLeg.get_frame_count('output'))
        Frames_num.append(self.rmLeg.get_frame_count('output'))
        Frames_num.append(self.lbLeg.get_frame_count('output'))
        Frames_num.append(self.rbLeg.get_frame_count('output'))

        Frames = np.array(Frames_num)

        return Frames

    def getHexapodFrames(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the completes set of hexapod frames in World Coordinates
        # Shape of Output is [4,4,6]

        base = []
        shoulder = []
        elbow = []
        feet = []

        base.append(self.lfLeg.get_forward_kinematics('CoM', angles[0, 0:3])[0])
        base.append(self.rfLeg.get_forward_kinematics('CoM', angles[0, 3:6])[0])
        base.append(self.lmLeg.get_forward_kinematics('CoM', angles[0, 6:9])[0])
        base.append(self.rmLeg.get_forward_kinematics('CoM', angles[0, 9:12])[0])
        base.append(self.lbLeg.get_forward_kinematics('CoM', angles[0, 12:15])[0])
        base.append(self.rbLeg.get_forward_kinematics('CoM', angles[0, 15:18])[0])

        shoulder.append(self.lfLeg.get_forward_kinematics('CoM', angles[0, 0:3])[2])
        shoulder.append(self.rfLeg.get_forward_kinematics('CoM', angles[0, 3:6])[2])
        shoulder.append(self.lmLeg.get_forward_kinematics('CoM', angles[0, 6:9])[2])
        shoulder.append(self.rmLeg.get_forward_kinematics('CoM', angles[0, 9:12])[2])
        shoulder.append(self.lbLeg.get_forward_kinematics('CoM', angles[0, 12:15])[2])
        shoulder.append(self.rbLeg.get_forward_kinematics('CoM', angles[0, 15:18])[2])

        elbow.append(self.lfLeg.get_forward_kinematics('CoM', angles[0, 0:3])[4])
        elbow.append(self.rfLeg.get_forward_kinematics('CoM', angles[0, 3:6])[4])
        elbow.append(self.lmLeg.get_forward_kinematics('CoM', angles[0, 6:9])[4])
        elbow.append(self.rmLeg.get_forward_kinematics('CoM', angles[0, 9:12])[4])
        elbow.append(self.lbLeg.get_forward_kinematics('CoM', angles[0, 12:15])[4])
        elbow.append(self.rbLeg.get_forward_kinematics('CoM', angles[0, 15:18])[4])

        feet.append(self.lfLeg.get_forward_kinematics('EndEffector', angles[0, 0:3])[0])
        feet.append(self.rfLeg.get_forward_kinematics('EndEffector', angles[0, 3:6])[0])
        feet.append(self.lmLeg.get_forward_kinematics('EndEffector', angles[0, 6:9])[0])
        feet.append(self.rmLeg.get_forward_kinematics('EndEffector', angles[0, 9:12])[0])
        feet.append(self.lbLeg.get_forward_kinematics('EndEffector', angles[0, 12:15])[0])
        feet.append(self.rbLeg.get_forward_kinematics('EndEffector', angles[0, 15:18])[0])

        feet = np.array(feet)
        base = np.array(base)
        shoulder = np.array(shoulder)
        elbow = np.array(elbow)

        return [base, shoulder, elbow, feet]

    def getLegPositions(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of the end effectors in world coordinate
        # Shape of Output is [3,6]

        positions = []

        positions.append(self.lfLeg.get_forward_kinematics('EndEffector', angles[0, 0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('EndEffector', angles[0, 3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('EndEffector', angles[0, 6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('EndEffector', angles[0, 9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('EndEffector', angles[0, 12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('EndEffector', angles[0, 15:18]))

        positions = np.array(positions)
        positions = np.squeeze(positions, axis=1)
        positions = positions[:, 0:3, 3]

        positions = positions.T

        return positions

    def getElbowPositions(self, angles):
        positions = [[self.lfLeg.get_forward_kinematics('CoM', angles[0, 0:3])[4]],
                     [self.rfLeg.get_forward_kinematics('CoM', angles[0, 3:6])[4]],
                     [self.lmLeg.get_forward_kinematics('CoM', angles[0, 6:9])[4]],
                     [self.rmLeg.get_forward_kinematics('CoM', angles[0, 9:12])[4]],
                     [self.lbLeg.get_forward_kinematics('CoM', angles[0, 12:15])[4]],
                     [self.rbLeg.get_forward_kinematics('CoM', angles[0, 15:18])[4]]]

        positions = np.array(positions)
        positions = np.squeeze(positions, axis=1)
        positions = positions[:, 0:3, 3]

        positions = positions.T

        return positions

    def getFrames_com(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of the 'com' of each frame in world coordinate

        positions = []

        positions.append(self.lfLeg.get_forward_kinematics('com', angles[0, 0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('com', angles[0, 3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('com', angles[0, 6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('com', angles[0, 9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('com', angles[0, 12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('com', angles[0, 15:18]))

        positions = np.array(positions)

        return positions

    def getFrames_output(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of 'output' of each frame in world coordinate
        # Shape of Output is [3,6]

        positions = []

        positions.append(self.lfLeg.get_forward_kinematics('output', angles[0, 0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('output', angles[0, 3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('output', angles[0, 6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('output', angles[0, 9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('output', angles[0, 12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('output', angles[0, 15:18]))

        positions = np.array(positions)

        return positions

    def getLegJacobians(self, angles):
        J = np.zeros([6, 3, 6])

        J[:, :, 0] = (self.lfLeg.get_jacobian_end_effector(angles[0, 0:3]))
        J[:, :, 1] = (self.rfLeg.get_jacobian_end_effector(angles[0, 3:6]))
        J[:, :, 2] = (self.lmLeg.get_jacobian_end_effector(angles[0, 6:9]))
        J[:, :, 3] = (self.rmLeg.get_jacobian_end_effector(angles[0, 9:12]))
        J[:, :, 4] = (self.lbLeg.get_jacobian_end_effector(angles[0, 12:15]))
        J[:, :, 5] = (self.rbLeg.get_jacobian_end_effector(angles[0, 15:18]))
        return J

    def getJacobians_frames(self, angles):
        J = np.zeros([7, 6, 3, 6])
        # print(self.lfLeg.get_frame_count('output'))   # 7
        # print(self.lfLeg.get_jacobians('output',angles[0, 0:3]))
        J[:, :, :, 0] = (self.lfLeg.get_jacobians('output',angles[0, 0:3]))
        J[:, :, :, 1] = (self.rfLeg.get_jacobians('output',angles[0, 3:6]))
        J[:, :, :, 2] = (self.lmLeg.get_jacobians('output',angles[0, 6:9]))
        J[:, :, :, 3] = (self.rmLeg.get_jacobians('output',angles[0, 9:12]))
        J[:, :, :, 4] = (self.lbLeg.get_jacobians('output',angles[0, 12:15]))
        J[:, :,  :, 5] = (self.rbLeg.get_jacobians('output',angles[0, 15:18]))
        return J


    def getLegVelocites(self, angles, angleVels):

        J = self.getLegJacobians(angles)

        legVelocities = np.zeros([6]);

        for i in range(6):
            legVel = np.dot(J[:, :, i], (angleVels[0, 3 * i:3 * (i + 1)]).T)
            legVelocities[i] = np.mean(abs(legVel)[0:3])

        return legVelocities

    def getLegTorques(self, angles, angleTorques):

        J = self.getLegJacobians(angles)

        legTorques = np.zeros([6])

        for i in range(6):
            legtor = np.dot(np.linalg.pinv((J[:, :, i]).T), (angleTorques[0, 3 * i:3 * (i + 1)]).T)
            legTorques[i] = legtor[2]

        return legTorques

    def getGRFs(self, angles, angleTorques):

        J = self.getJacobians_frames(angles)
        # print(J)

        legTorques = np.zeros([6,3])

        for i in range(6):
            legtor = np.dot(np.linalg.pinv((J[6, :, :, i]).T), (angleTorques[0, 3 * i:3 * (i + 1)]).T)
            # print('legtor :', legtor)
            legTorques[i,:] = legtor[0:3]

        return legTorques

    def getContactLegs(self, legtorques):

        index_min = np.argmin(legtorques)

        if (legtorques[0] + legtorques[3] + legtorques[4]) < (legtorques[1] + legtorques[2] + legtorques[5]):
            return [0, 3, 4]
        else:
            return [1, 2, 5]

    def getLegIK(self, xd):

        # gets the xyz coordinates in world coordinate with size of [3,6]
        # return the joint angles with size of [18]
        angles = np.zeros([18])
        guess = [0.5852, -0.0766, -1.6584, -0.5852, 0.0766, 1.6584, 0.1851, -0.0753, -1.6982,
                 0.1335, 0.0767, 1.6506, -0.1568, -0.0731, -1.7289, 0.1568, 0.0731, 1.7289]
        # guess = [0.45, 0.32, -0.23, 0.45, -0.38, -0.32, 0, 0.5, -0.23,
        #          0, -0.5, -0.23, -0.4, 0.35, -0.23, -0.4, -0.35, -0.23]
        """
        [0.4     0.4     0.       0.     -0.4   -0.4]
        [0.38  - 0.38   0.5      -0.5    0.35   -0.35]
        [-0.23 - 0.23   -0.23   -0.23   -0.23   -0.23]]
        """

        ee_pos_objective1 = hebi.robot_model.endeffector_position_objective(xd[:, 0])
        angles[0:3] = self.lfLeg.solve_inverse_kinematics(guess[0:3], ee_pos_objective1)

        ee_pos_objective2 = hebi.robot_model.endeffector_position_objective(xd[:, 1])
        angles[3:6] = self.rfLeg.solve_inverse_kinematics(guess[3:6], ee_pos_objective2)

        ee_pos_objective3 = hebi.robot_model.endeffector_position_objective(xd[:, 2])
        angles[6:9] = self.lmLeg.solve_inverse_kinematics(guess[6:9], ee_pos_objective3)

        ee_pos_objective4 = hebi.robot_model.endeffector_position_objective(xd[:, 3])
        angles[9:12] = self.rmLeg.solve_inverse_kinematics(guess[9:12], ee_pos_objective4)

        ee_pos_objective5 = hebi.robot_model.endeffector_position_objective(xd[:, 4])
        angles[12:15] = self.lbLeg.solve_inverse_kinematics(guess[12:15], ee_pos_objective5)

        ee_pos_objective6 = hebi.robot_model.endeffector_position_objective(xd[:, 5])
        angles[15:18] = self.rbLeg.solve_inverse_kinematics(guess[15:18], ee_pos_objective6)

        return angles

    def updateBaseFrame(self, contactLegs, feet):

        xyzStance = np.zeros([3, 3])
        xyzCurr = np.zeros([3, 3])

        for i in range(len(contactLegs)):
            foot = contactLegs[i]
            xyzStance[i, :] = ((self.stanceFeet[foot])[0:3, 3]).T
            xyzCurr[i, :] = ((feet[foot])[0:3, 3]).T

        transformBase = svd_transform(xyzStance, xyzCurr)
        self.robot_base = np.dot(self.robot_base, np.linalg.pinv(transformBase))
        self.stanceFeet = feet

    def getCenterOfMasses(self, angles):
        # Gets the xyz positions of the COM of each joint in each leg in the body frame
        # angles is a 18 element vector of joint angles
        # positions is a 3x6 matrix

        CoMs = np.zeros([3, 9, 6]);

        CoMs[:, 1:5, 0] = self.getXYZ(self.lfLeg.get_forward_kinematics('CoM', angles[0, 0:3])[0])
        CoMs[:, 1:5, 1] = self.getXYZ(self.rfLeg.get_forward_kinematics('CoM', angles[0, 3:6])[0])
        CoMs[:, 1:5, 2] = self.getXYZ(self.lmLeg.get_forward_kinematics('CoM', angles[0, 6:9])[0])
        CoMs[:, 1:5, 3] = self.getXYZ(self.rmLeg.get_forward_kinematics('CoM', angles[0, 9:12])[0])
        CoMs[:, 1:5, 4] = self.getXYZ(self.lbLeg.get_forward_kinematics('CoM', angles[0, 12:15])[0])
        CoMs[:, 1:5, 5] = self.getXYZ(self.rbLeg.get_forward_kinematics('CoM', angles[0, 15:18])[0])

        return CoMs

    def getLegMasses(self):

        masses = np.zeros([7, 6]);
        masses[:, 0] = self.lfLeg.masses
        masses[:, 1] = self.rfLeg.masses
        masses[:, 2] = self.lmLeg.masses
        masses[:, 3] = self.rmLeg.masses
        masses[:, 4] = self.lbLeg.masses
        masses[:, 5] = self.rbLeg.masses

        return masses


if __name__ == "__main__":
    xmk = HexapodKinematics()
    angles = np.array([0, 0, -np.pi / 2, 0, 0, np.pi / 2, 0, 0, -np.pi / 2,
                       0, 0, np.pi / 2, 0, 0, -np.pi / 2, 0, 0, np.pi / 2]).reshape((1, 18))

    position = xmk.getLegPositions(angles)
    Frame_num = xmk.getFramesNum()
    Frames = xmk.getHexapodFrames(angles)
    Jacobians = xmk.getLegJacobians(angles)
    angles = xmk.getLegIK(position)
    mass = xmk.getLegMasses()
    # print(position)
    # print(angles)
    # print(mass)

"""
    left_leg = hebi.robot_model.import_from_hrdf("hrdf/daisyLeg-Left.hrdf")

    angles = np.array([0,0,np.pi/2])
    frames = left_leg.get_frame_count('output')
    positions_com = left_leg.get_forward_kinematics('com', angles)
    positions_output = left_leg.get_forward_kinematics('output', angles)
    ee_pos = left_leg.get_end_effector(angles)

    # print('frame num : \n', frames)
    # print('com : \n', positions_com[6])
    # print('output : \n',positions_output[6])
    # print('end-effector position : \n', ee_pos)
"""
