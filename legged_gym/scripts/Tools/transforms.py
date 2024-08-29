## Some common functions for manipulating homegeous matricies and rotations

import numpy as np
import math
from math import pi,cos,sin





# Checks if R is a valid Rotation Matrix
def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
     
     
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])




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
    #Homogeneous transform matrix for a rotation about y
    return np.array([[cos(theta), 0, sin(theta), 0],
                     [0, 1, 0, 0],
                     [-sin(theta), 0, cos(theta), 0],
                     [0, 0, 0, 1]])

def rotz(theta):

    return np.array([[cos(theta), -sin(theta), 0, 0],
                     [sin(theta), cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def eulerSO3(rotationMatrix):
    thetaX = np.arctan2(rotationMatrix[2,1],rotationMatrix[2,2]);
    thetaY = np.arctan2(-rotationMatrix[2,0],np.linalg.norm(rotationMatrix[2,1:3]));
    thetaZ = np.arctan2(rotationMatrix[1,0],rotationMatrix[0,0]);
    return [thetaX, thetaY, thetaZ] 

def XYrot(pose,Tg=np.eye(3)):

    x1 = np.dot(pose[0:3,0:3] , np.array([[1],[0],[0]]))
    x2 = np.array([[1],[0],[0]])
    rotM = rotz( np.arctan2(x1[1], x1[0]) - np.arctan2(x2[1], x2[0]) )

    newPose = rotM[0:3,0:3]
    newPose = newPose[0:3,0:3]
    T = np.dot(np.dot(newPose[0:3,0:3] , Tg[0:3,0:3]), pose[0:3,0:3].T)

    return [T,newPose]


def SE3(SO3, r):
#Builds an SE(3) transform
  T = np.eye(4);

  T[0:3,0:3] = SO3
  r = np.array(r)
  T[0:3,3] = r[:]

  return T