from Tools.computeG import computeG
import numpy as np
from Tools.transforms import SE3, XYrot
from Tools.computeLegCM import computeLegCM

def  eulerSO3(rotationMatrix):
    thetaX = np.arctan2(rotationMatrix[2,1],rotationMatrix[2,2]);
    thetaY = np.arctan2(-rotationMatrix[2,0],np.linalg.norm(rotationMatrix[2,1:2]));
    thetaZ = np.arctan2(rotationMatrix[1,0],rotationMatrix[0,0]);
    return [thetaX, thetaY, thetaZ] 

def constrainSE3(cpg, t, dt):
#constraints to ...
# AB: expresses frame A w.r.t frame B
    np.set_printoptions(precision=5)
    r = np.vstack((cpg['xmk'].getLegPositions(cpg['legs']), np.ones([1,6])))
    cpg['G'] = np.identity(3)

    #Set SE3 tranforms
    GW = SE3(cpg['G'], np.zeros([3]))
    WG = GW.T

    BW = SE3(cpg['pose'], np.zeros(3))
    WB = BW.T

    GB = np.dot(GW, BW.T)
    BG = GB.T

    # Present legs positions in the ground frame
    rG = np.dot(GB, r); # R: Takes leg positions from body frame to ground frame

    target = XYrot(GB)[1];
    rot_matrix = np.dot(target, BG[0:3,0:3])

    # Calculate R(T,G) in the Ground frame

    _,Tx,_= eulerSO3(cpg['pose'])
    actual_bh = 0.085
    Xoffset = 0.08  #based on pose
    goal_pos = np.array([[-.08],[0],[cpg['h']],[1]])

    xc,yc,_,_ = computeLegCM(cpg,t,rG[0:3,:]); # R: computes the center of mass of the legs in the ground frame

    curr_pos = -np.vstack((xc * np.ones([1,6]),yc * np.ones([1,6]), rG[2,:], np.zeros([1,6])))

    eTrans = curr_pos - goal_pos

    if cpg['move']:
       eTrans[3,~cpg['CPGStance']] = 0;


    # Package both into R \in  SE(3) & Perform correction
    cpg['R'] = np.zeros([4,4,6])
    er = np.zeros([4,6])

    for leg in range(6):
        erStart = np.dot(BG[0:3,0:3] , (np.dot(rot_matrix , rG[0:3,leg]) - rG[0:3,leg]))        

        er1 = np.array([0,0,erStart[2],1])
        
        cpg['R'][:,:,leg] = SE3( np.eye(3), eTrans[0:3,leg] )
        er[:,leg] = -er1
        #Set an error dead-zone
        if np.linalg.norm(er[0:3,leg]) < 0.01:
            er[0:3,leg] = 0

    er = er[0:3,:]
    return [-er, cpg]


