import numpy as np
def computeLegCM(cpg,t,r):
#Computes the center of mass of the legs in the body frame
#Computes the height of three lowest legs in the body frame

    xc = np.mean(r[0,:]);
    yc = np.mean(r[1,:]);
    zc = np.mean(r[2,:]);

    z3l = np.mean(r[2,0:3])

    return [xc,yc,zc,z3l]