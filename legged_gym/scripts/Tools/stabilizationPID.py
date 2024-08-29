import numpy as np
from XYrot import XYrot

def stabilizationPID(cpg):
    #Gives a rotation matrix for correct orientation given the world frame
    R = XYrot(cpg['pose'])[0]

    FKbody = cpg['smk'].getLegPositions(cpg['legs'])

    FKworld = np.dot(cpg['pose'], FKbody.T)

    # a 3*6 matrix with each row as x , y, z coordinates of ee 
    FKworldCor = np.matmul(R[:3,:3] , FKworld)

    #print(FKworldCor)

    # Set the stance to the lowest tripod
    use145 = np.mean(FKworldCor[2,[0,3,4]]) < np.mean(FKworldCor[2,[1,2,5]])
    if use145:
        cpg['on_ground'] = np.asarray(FKworldCor[2,:] <= np.max(FKworldCor[2,[0,3,4]]), dtype=int)
    else:
        cpg['on_ground'] = np.asarray(FKworldCor[2,:] <= np.max(FKworldCor[2,[1,2,5]]), dtype=int)

    ## Compute Height from Ground Plane
    dirVec = [[0],[0],[1]]

    #Determine normal of plane formed by ground feet
    if use145:
        cpg['groundNorm'] = np.cross((FKworldCor[:,0] - FKworldCor[:,3]).T, (FKworldCor[:,0] - FKworldCor[:,4]).T)
        cpg['groundD'] = np.dot(cpg['groundNorm'], FKworldCor[:,0])
    else:
        cpg['groundNorm'] = np.cross((FKworldCor[:,1] - FKworldCor[:,2]).T, (FKworldCor[:,1] - FKworldCor[:,5]).T)
        cpg['groundD'] = np.dot(cpg['groundNorm'], FKworldCor[:,1])


    #find intersection
    t = cpg['groundD'] / np.dot(cpg['groundNorm'], dirVec)

    #find height
    zDist = np.linalg.norm(dirVec * t)

    ## Adjust legs for Z
    cpg['zHistory'][0,cpg['zHistoryCnt']] = zDist
    cpg['zHistoryCnt']= np.mod(cpg['zHistoryCnt']+1,10)

    cpg['zErr'] = cpg['bodyHeight'] - np.median(cpg['zHistory'])

    FKworldCor[2] = FKworldCor[2] + cpg['zErr']

    dxWorld = FKworldCor - FKworld

    cpg['dx'] = - np.linalg.lstsq(cpg['pose'], dxWorld)[0]

    return cpg