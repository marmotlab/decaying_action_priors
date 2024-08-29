
import hebi
import math
from math import pi,cos,sin
import numpy as np
#import quaternion
from Tools.transforms import *
from Tools.rigid_transform_3D import *




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

        self.rfLeg.base_frame=np.dot(rotz(-pi/6),trans([long_legs,0,0,0,0,0]))
        self.rmLeg.base_frame=np.dot(rotz(-pi/2),trans([short_legs,0,0,0,0,0])) 
        self.rbLeg.base_frame=np.dot(rotz(-5*pi/6),trans([long_legs,0,0,0,0,0]))
        self.lfLeg.base_frame=np.dot(rotz(pi/6),trans([long_legs,0,0,0,0,0]))
        self.lmLeg.base_frame=np.dot(rotz(pi/2),trans([short_legs,0,0,0,0,0]))
        self.lbLeg.base_frame=np.dot(rotz(5*pi/6),trans([long_legs,0,0,0,0,0]))


        self.group_fb1 = hebi.GroupFeedback(self.lfLeg.dof_count)
        self.group_fb2 = hebi.GroupFeedback(self.rfLeg.dof_count)
        self.group_fb3 = hebi.GroupFeedback(self.lmLeg.dof_count)
        self.group_fb4 = hebi.GroupFeedback(self.rmLeg.dof_count)
        self.group_fb5 = hebi.GroupFeedback(self.lbLeg.dof_count)
        self.group_fb6 = hebi.GroupFeedback(self.rbLeg.dof_count)


        #need to set world base frame
        self.robot_base = np.identity(4)




    def getLeg(self,side):
    
        if side == 'right':
            mount_side = 'right-inside'
        else:
            mount_side = 'left-inside'

        kin = hebi.robot_model.RobotModel()
            
        kin.add_actuator('X8-9')
        kin.add_bracket('X5-HeavyBracket', mount_side)
        kin.add_actuator('X8-16');
        kin.add_link('X5',0.325,pi)  
        kin.add_actuator('X8-9');
        kin.add_link('X5',0.325,0)  

        return kin




    def getLegPositions(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of the end effectors in world coordinate 
        # Shape of Output is [3,6]

        positions  = []

        positions.append(self.lfLeg.get_forward_kinematics('EndEffector',angles[0,0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('EndEffector',angles[0,3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('EndEffector',angles[0,6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('EndEffector',angles[0,9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('EndEffector',angles[0,12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('EndEffector',angles[0,15:18]))



        positions = np.array(positions)
        positions = np.squeeze(positions,axis=1)
        positions = positions[:,0:3,3]

        positions = positions.T

        return positions




    def getLegFrames(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of the end effectors in world coordinate 
        # Shape of Output is [3,6]

        positions  = []

        positions.append(self.lfLeg.get_forward_kinematics('output',angles[0,0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('output',angles[0,3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('output',angles[0,6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('output',angles[0,9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('output',angles[0,12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('output',angles[0,15:18]))

        positions = np.array(positions)

        return positions




    def getHexapodFrames(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the completes set of hexapod frames in World Coordinates 
        # Shape of Output is [4,4,6]

        base = []
        shoulder = []
        elbow = []
        feet  = []
        
        base.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[0])
        base.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[0])
        base.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[0])
        base.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[0])
        base.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[0])
        base.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[0])

        shoulder.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[2])
        shoulder.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[2])
        shoulder.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[2])
        shoulder.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[2])
        shoulder.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[2])
        shoulder.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[2])

        elbow.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[4])
        elbow.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[4])
        elbow.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[4])
        elbow.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[4])
        elbow.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[4])
        elbow.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[4])

        feet.append(self.lfLeg.get_forward_kinematics('EndEffector',angles[0,0:3])[0])
        feet.append(self.rfLeg.get_forward_kinematics('EndEffector',angles[0,3:6])[0])
        feet.append(self.lmLeg.get_forward_kinematics('EndEffector',angles[0,6:9])[0])
        feet.append(self.rmLeg.get_forward_kinematics('EndEffector',angles[0,9:12])[0])
        feet.append(self.lbLeg.get_forward_kinematics('EndEffector',angles[0,12:15])[0])
        feet.append(self.rbLeg.get_forward_kinematics('EndEffector',angles[0,15:18])[0])


        feet = np.array(feet)
        base = np.array(base)
        shoulder = np.array(shoulder)
        elbow = np.array(elbow)

        return [base,shoulder,elbow,feet]




    def getLegJacobians(self, angles):
        J = np.zeros([6,3,6]);

        J[:,:,0] = (self.lfLeg.get_jacobian_end_effector(angles[0,0:3]))
        J[:,:,1] = (self.rfLeg.get_jacobian_end_effector(angles[0,3:6]));
        J[:,:,2] = (self.lmLeg.get_jacobian_end_effector(angles[0,6:9]));
        J[:,:,3] = (self.rmLeg.get_jacobian_end_effector(angles[0,9:12]));
        J[:,:,4] = (self.lbLeg.get_jacobian_end_effector(angles[0,12:15]));           
        J[:,:,5] = (self.rbLeg.get_jacobian_end_effector(angles[0,15:18]));
        return J



    def getLegVelocites(self, angles, angleVels):

        J = self.getLegJacobians(angles)

        legVelocities = np.zeros([6]);

        for i in range(6):
            legVel = np.dot(J[:,:,i] ,(angleVels[0,3*i:3*(i+1)]).T)
            legVelocities[i] = np.mean(abs(legVel)[0:3])


        return legVelocities


    def getLegTorques(self, angles, angleTorques):

        J = self.getLegJacobians(angles)

        legTorques = np.zeros([6]);

        for i in range(6):
            legtor = np.dot(np.linalg.pinv((J[:,:,i]).T) ,(angleTorques[0,3*i:3*(i+1)]).T)
            legTorques[i] = legtor[2]
            
        return legTorques



    def getContactLegs(self,legtorques):

        index_min = np.argmin(legtorques)

        if (legtorques[0] + legtorques[3] + legtorques [4]) < (legtorques[1] + legtorques[2] + legtorques [5]):
            return [0,3,4]
        else:
            return [1,2,5]



    def getLegIK(self,xd,guess = [0.5852, -0.0766, -1.6584, -0.5852, 0.0766, 1.6584, 0.1851, -0.0753, -1.6982,
                 0.1335, 0.0767, 1.6506, -0.1568, -0.0731, -1.7289, 0.1568, 0.0731, 1.7289]):

        # gets the xyz coordinates in world coordinate with size of [3,6]
        # return the joint angles with size of [18]
        angles = np.zeros([18])
        #guess = [0.5852, -0.0766, -1.6584, -0.5852, 0.0766, 1.6584, 0.1851, -0.0753, -1.6982,
        #         0.1335, 0.0767, 1.6506, -0.1568, -0.0731, -1.7289, 0.1568, 0.0731, 1.7289]

        ee_pos_objective1 = hebi.robot_model.endeffector_position_objective(xd[:,0])
        angles[0:3] = self.lfLeg.solve_inverse_kinematics(guess[0:3],ee_pos_objective1)

        ee_pos_objective2 = hebi.robot_model.endeffector_position_objective(xd[:,1])
        angles[3:6] = self.rfLeg.solve_inverse_kinematics(guess[3:6], ee_pos_objective2)

        ee_pos_objective3 = hebi.robot_model.endeffector_position_objective(xd[:,2])
        angles[6:9] = self.lmLeg.solve_inverse_kinematics(guess[6:9], ee_pos_objective3)

        ee_pos_objective4 = hebi.robot_model.endeffector_position_objective(xd[:,3])
        angles[9:12] =self.rmLeg.solve_inverse_kinematics(guess[9:12], ee_pos_objective4)

        ee_pos_objective5 = hebi.robot_model.endeffector_position_objective(xd[:,4])
        angles[12:15] = self.lbLeg.solve_inverse_kinematics(guess[12:15], ee_pos_objective5)

        ee_pos_objective6 = hebi.robot_model.endeffector_position_objective(xd[:,5])
        angles[15:18] = self.rbLeg.solve_inverse_kinematics(guess[15:18], ee_pos_objective6)

        return angles

    
    def updateBaseFrame(self,contactLegs,feet):

        xyzStance = np.zeros([3,3])
        xyzCurr= np.zeros([3,3])

        for i in range(len(contactLegs)):
            foot = contactLegs[i]
            xyzStance[i,:] = ((self.stanceFeet[foot])[0:3,3]).T
            xyzCurr[i,:] = ((feet[foot])[0:3,3]).T
 
        transformBase = svd_transform(xyzStance, xyzCurr)
        self.robot_base = np.dot(self.robot_base, np.linalg.pinv(transformBase))
        self.stanceFeet = feet

    def getCenterOfMasses(self, angles):
        #Gets the xyz positions of the COM of each joint in each leg in the body frame
        #angles is a 18 element vector of joint angles
        #positions is a 3x6 matrix

        CoMs = np.zeros([3,9,6]);

        CoMs[:,1:5,0] = self.getXYZ(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[0]);
        CoMs[:,1:5,1] = self.getXYZ(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[0]);
        CoMs[:,1:5,2] = self.getXYZ(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[0]);
        CoMs[:,1:5,3] = self.getXYZ(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[0]);
        CoMs[:,1:5,4] = self.getXYZ(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[0]);           
        CoMs[:,1:5,5] = self.getXYZ(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[0])
        
        return CoMs
      

    def getLegMasses(self):

        masses = np.zeros([6,6]);
        masses[:,0] = self.lfLeg.masses;
        masses[:,1] = self.rfLeg.masses;
        masses[:,2] = self.lmLeg.masses;
        masses[:,3] = self.rmLeg.masses;
        masses[:,4] = self.lbLeg.masses;
        masses[:,5] = self.rbLeg.masses;

        return masses



'''

    def getRobotRPY(self, orientation):

        imuModules = range(0,18,3)

        backRots = [(self.lfLeg.base_frame[0:3,0:3]).T,
                    (self.rfLeg.base_frame[0:3,0:3]).T,
                    (self.lmLeg.base_frame[0:3,0:3]).T,
                    (self.rmLeg.base_frame[0:3,0:3]).T,
                    (self.lbLeg.base_frame[0:3,0:3]).T,
                    (self.rbLeg.base_frame[0:3,0:3]).T]

        Roll = 0.0
        Pitch = 0.0
        Yaw = 0.0
        counter = 0

        for i in range(6):

            Q = orientation[imuModules[i]]

            quat = np.quaternion(Q[0],Q[1],Q[2],Q[3])
            rot = quaternion.as_rotation_matrix(quat)

            if not np.isnan(np.min(rot)):
                trans = np.dot(rot , backRots[i])
                RPY = rotationMatrixToEulerAngles(trans)
    
                for j in range(3):
                    if RPY[j] > pi:
                        RPY[j] = RPY[i] - pi
                    RPY[j] = -RPY[j]
                Roll += RPY[0] 
                Pitch += RPY[1] 
                Yaw += RPY[2]
                counter += 1
    
        rollR = rotx(Roll/counter)
        pitchR =  roty(Pitch/counter)
        yawR =  rotz(Yaw/counter)
        FinalRot = np.dot(pitchR,rollR)
        
        return FinalRot 
'''



'''
            
import hebi
import math
from math import pi,cos,sin
import numpy as np
import quaternion
#from rigid_transform_3D import svd_transform
from transforms import *




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

        self.rfLeg.base_frame=np.dot(rotz(-pi/6),trans([long_legs,0,0,0,0,0]))
        self.rmLeg.base_frame=np.dot(rotz(-pi/2),trans([short_legs,0,0,0,0,0])) 
        self.rbLeg.base_frame=np.dot(rotz(-5*pi/6),trans([long_legs,0,0,0,0,0]))
        self.lfLeg.base_frame=np.dot(rotz(pi/6),trans([long_legs,0,0,0,0,0]))
        self.lmLeg.base_frame=np.dot(rotz(pi/2),trans([short_legs,0,0,0,0,0]))
        self.lbLeg.base_frame=np.dot(rotz(5*pi/6),trans([long_legs,0,0,0,0,0]))


        self.group_fb1 = hebi.GroupFeedback(self.lfLeg.dof_count)
        self.group_fb2 = hebi.GroupFeedback(self.rfLeg.dof_count)
        self.group_fb3 = hebi.GroupFeedback(self.lmLeg.dof_count)
        self.group_fb4 = hebi.GroupFeedback(self.rmLeg.dof_count)
        self.group_fb5 = hebi.GroupFeedback(self.lbLeg.dof_count)
        self.group_fb6 = hebi.GroupFeedback(self.rbLeg.dof_count)


        #need to set world base frame
        self.robot_base = np.identity(4)




    def getLeg(self,side):
    
        if side == 'right':
            mount_side = 'right-inside'
        else:
            mount_side = 'left-inside'

        kin = hebi.robot_model.RobotModel()
            
        kin.add_actuator('X8-9')
        kin.add_bracket('X5-HeavyBracket', mount_side)
        kin.add_actuator('X8-16');
        kin.add_link('X5',0.325,pi)  
        kin.add_actuator('X8-9');
        kin.add_link('X5',0.325,0)  

        return kin




    def getLegPositions(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the xyz coordinates of the end effectors in world coordinate 
        # Shape of Output is [3,6]

        positions  = []

        positions.append(self.lfLeg.get_forward_kinematics('EndEffector',angles[0,0:3]))
        positions.append(self.rfLeg.get_forward_kinematics('EndEffector',angles[0,3:6]))
        positions.append(self.lmLeg.get_forward_kinematics('EndEffector',angles[0,6:9]))
        positions.append(self.rmLeg.get_forward_kinematics('EndEffector',angles[0,9:12]))
        positions.append(self.lbLeg.get_forward_kinematics('EndEffector',angles[0,12:15]))
        positions.append(self.rbLeg.get_forward_kinematics('EndEffector',angles[0,15:18]))



        positions = np.array(positions)
        positions = np.squeeze(positions,axis=1)
        positions = positions[:,0:3,3]

        positions = positions.T


        return positions




    def getHexapodFrames(self, angles):
        # The input is the angles with the shape of [1.18]
        # The output is the completes set of hexapod frames in World Coordinates 
        # Shape of Output is [4,4,6]

        base = []
        shoulder = []
        elbow = []
        feet  = []

        base.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[0])
        base.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[0])
        base.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[0])
        base.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[0])
        base.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[0])
        base.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[0])

        shoulder.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[2])
        shoulder.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[2])
        shoulder.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[2])
        shoulder.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[2])
        shoulder.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[2])
        shoulder.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[2])

        elbow.append(self.lfLeg.get_forward_kinematics('CoM',angles[0,0:3])[4])
        elbow.append(self.rfLeg.get_forward_kinematics('CoM',angles[0,3:6])[4])
        elbow.append(self.lmLeg.get_forward_kinematics('CoM',angles[0,6:9])[4])
        elbow.append(self.rmLeg.get_forward_kinematics('CoM',angles[0,9:12])[4])
        elbow.append(self.lbLeg.get_forward_kinematics('CoM',angles[0,12:15])[4])
        elbow.append(self.rbLeg.get_forward_kinematics('CoM',angles[0,15:18])[4])

        feet.append(self.lfLeg.get_forward_kinematics('EndEffector',angles[0,0:3])[0])
        feet.append(self.rfLeg.get_forward_kinematics('EndEffector',angles[0,3:6])[0])
        feet.append(self.lmLeg.get_forward_kinematics('EndEffector',angles[0,6:9])[0])
        feet.append(self.rmLeg.get_forward_kinematics('EndEffector',angles[0,9:12])[0])
        feet.append(self.lbLeg.get_forward_kinematics('EndEffector',angles[0,12:15])[0])
        feet.append(self.rbLeg.get_forward_kinematics('EndEffector',angles[0,15:18])[0])


        feet = np.array(feet)
        base = np.array(base)
        shoulder = np.array(shoulder)
        elbow = np.array(elbow)

        return [base,shoulder,elbow,feet]




    def getLegJacobians(self, angles):
        #print(np.shape(angles))
        J = np.zeros([6,3,6]);

        J[:,:,0] = (self.lfLeg.get_jacobian_end_effector(angles[0,0:3]))
        J[:,:,1] = (self.rfLeg.get_jacobian_end_effector(angles[0,3:6]));
        J[:,:,2] = (self.lmLeg.get_jacobian_end_effector( angles[0,6:9]));
        J[:,:,3] = (self.rmLeg.get_jacobian_end_effector( angles[0,9:12]));
        J[:,:,4] = (self.lbLeg.get_jacobian_end_effector( angles[0,12:15]));           
        J[:,:,5] = (self.rbLeg.get_jacobian_end_effector( angles[0,15:18]));
        return J



    def getLegVelocites(self, angles, angleVels):

        J = self.getLegJacobians(angles)
        #print(angleVels[0,0:3])

        legVelocities = np.zeros([6]);

        for i in range(6):
            legVel = np.dot(J[:,:,i] ,(angleVels[0,3*i:3*(i+1)]).T)
            legVelocities[i] = np.mean(abs(legVel)[0:3])


        return legVelocities


    def getLegTorques(self, angles, angleTorques):

        J = self.getLegJacobians(angles)
        #print(angleTorques)
        #print(J)
        #0/0
        #print(angleVels[0,0:3])

        legTorques = np.zeros([6]);

        for i in range(6):
            legtor = np.dot(np.linalg.pinv((J[:,:,i]).T) ,(angleTorques[0,3*i:3*(i+1)]).T)
            #print(legtor)
            legTorques[i] = legtor[2]
            
        #print(legTorques)
        #0/0
        return legTorques



    def getContactLegs(self,legtorques):

        index_min = np.argmin(legtorques)

        if (legtorques[0] + legtorques[3] + legtorques [4]) < (legtorques[1] + legtorques[2] + legtorques [5]):
            return [0,3,4]
        else:
            return [1,2,5]



    def inverseKinematics(self,xd):

        # gets the xyz coordinates in world coordinate with size of [3,6]
        # return the joint angles with size of [18]

        angles = np.zeros([18])

        initial_joint_angles1 = np.zeros([3,1])#self.group_fb1.position
        initial_joint_angles2 = np.zeros([3,1])
        initial_joint_angles3 = np.zeros([3,1])
        initial_joint_angles4 = np.zeros([3,1])
        initial_joint_angles5 = np.zeros([3,1])
        initial_joint_angles6 = np.zeros([3,1])



        ee_pos_objective1 = hebi.robot_model.endeffector_position_objective(xd[:,0])
        angles[0:3] = self.lfLeg.solve_inverse_kinematics(initial_joint_angles1,ee_pos_objective1)


        ee_pos_objective2 = hebi.robot_model.endeffector_position_objective(xd[:,1])
        angles[3:6] = self.rfLeg.solve_inverse_kinematics(initial_joint_angles2, ee_pos_objective2)

        ee_pos_objective3 = hebi.robot_model.endeffector_position_objective(xd[:,2])
        angles[6:9] = self.lmLeg.solve_inverse_kinematics(initial_joint_angles3, ee_pos_objective3)

        ee_pos_objective4 = hebi.robot_model.endeffector_position_objective(xd[:,3])
        angles[9:12] =self.rmLeg.solve_inverse_kinematics(initial_joint_angles4, ee_pos_objective4)

        ee_pos_objective5 = hebi.robot_model.endeffector_position_objective(xd[:,4])
        angles[12:15] = self.lbLeg.solve_inverse_kinematics(initial_joint_angles5, ee_pos_objective5)

        ee_pos_objective6 = hebi.robot_model.endeffector_position_objective(xd[:,5])
        angles[15:18] = self.rbLeg.solve_inverse_kinematics(initial_joint_angles6, ee_pos_objective6)

        return angles




    def getRobotRPY(self, orientation):

        imuModules = range(0,18,3)

        backRots = [(self.lfLeg.base_frame[0:3,0:3]).T,
                    (self.rfLeg.base_frame[0:3,0:3]).T,
                    (self.lmLeg.base_frame[0:3,0:3]).T,
                    (self.rmLeg.base_frame[0:3,0:3]).T,
                    (self.lbLeg.base_frame[0:3,0:3]).T,
                    (self.rbLeg.base_frame[0:3,0:3]).T]


        Roll = 0.0
        Pitch = 0.0
        Yaw = 0.0
        counter = 0




        for i in range(6):


            Q = orientation[imuModules[i]]
            



            quat = np.quaternion(Q[0],Q[1],Q[2],Q[3])
            rot = quaternion.as_rotation_matrix(quat)



            if not np.isnan(np.min(rot)):
                trans = np.dot(rot , backRots[i])
                RPY = rotationMatrixToEulerAngles(trans)
    

                for j in range(3):
                    if RPY[j] > pi:
                        RPY[j] = RPY[i] - pi
                    RPY[j] = -RPY[j]
                Roll += RPY[0] 
                Pitch += RPY[1] 
                Yaw += RPY[2]
                counter += 1
    

        rollR = rotx(Roll/counter)
        pitchR =  roty(Pitch/counter)
        yawR =  rotz(Yaw/counter)
        FinalRot = np.dot(pitchR,rollR)
        

        return FinalRot 

    def updateBaseFrame(self,contactLegs,feet):

        xyzStance = np.zeros([3,3])
        xyzCurr= np.zeros([3,3])

        for i in range(len(contactLegs)):
            foot = contactLegs[i]
            xyzStance[i,:] = ((self.stanceFeet[foot])[0:3,3]).T
            xyzCurr[i,:] = ((feet[foot])[0:3,3]).T
 

        transformBase = svd_transform(xyzStance, xyzCurr)
        self.robot_base = np.dot(self.robot_base, np.linalg.pinv(transformBase))
        self.stanceFeet = feet
    
'''
         
