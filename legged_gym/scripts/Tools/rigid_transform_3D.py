from numpy import *
import numpy
from math import sqrt

# Inumpyut: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def svd_transform(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = numpy.dot(transpose(AA) , BB)

    U, S, Vt = linalg.svd(H)

    R = numpy.dot(Vt.T , U.T)

    # special reflection case
    if linalg.det(R) < 0:
       print ("Reflection detected")
       Vt[2,:] *= -1
       R = numpy.dot(Vt.T , U.T)

    t = numpy.dot(-R,centroid_A.T) + centroid_B.T

    H1 = numpy.zeros((4,4))
    H1[0:3,0:3] = R
    H1[0:3,3] = t
    H1[3,3] = 1

    return H1

