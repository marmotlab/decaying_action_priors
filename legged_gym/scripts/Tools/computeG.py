import numpy as np
from math import sqrt

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def norm(x):
    return sqrt(dot_product(x, x))

def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def computeG(cpg,t,r):
    #Gives the SO3 Transform between the world frame and the ground frame

    P = cpg['pose']

    #print(P)
    #print(cpg['CPGStance'])
    # Use the three points to form a plane; get its normal
    gPos = r[:,0:3] #positin of ee that are on the ground

    gNorm = np.cross(gPos[:,0] - gPos[:,1],gPos[:,0] - gPos[:,2])

    gZ = np.dot(P , (gNorm/np.linalg.norm(gNorm)))

    #print(gZ)

    # Ensure normal points up in world frame
    if gZ[2] < 0:
        gZ = -gZ;

    # Ensure heading of robot in the ground frame and heading in the body frame
    # are coplanar (project bY onto the ground plane)
    bX = np.dot(P , np.array([[1],[0],[0]])) #[0; 1; 0] is in body frame

    gX = bX - project_onto_plane(bX, gZ)

    gX = gX/np.linalg.norm(gX)

    gX = np.squeeze(gX)

    gY = np.cross(gZ, gX)

    #Build the SO(3)
    G = np.array([[gX],[gY],[gZ]])

    G = np.squeeze(G,axis=1)

    #print("end of Compute G")

    return G


