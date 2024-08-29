import numpy as np
import copy



class PIDController:
    #UNTITLED Summary of this class goes here
    #   Detailed explanation goes here
    

  def __init__(self,vec,kP,kI,alpha,kD,limP,limI,limD,limTot):
    self.rows = vec[0];
    self.cols = vec[1];

    self.kP = kP;
    self.kI = kI;
    self.kD = kD;
    self.limP = limP;
    self.limI = limI;
    self.limD = limD;
    self.limTot = limTot;

    zeroInit = np.zeros(vec);

    self.error = copy.deepcopy(zeroInit);
    self.integral = copy.deepcopy(zeroInit);
    self.prevError = copy.deepcopy(zeroInit);
    self.out = copy.deepcopy(zeroInit);
    self.prevOut = copy.deepcopy(zeroInit);
    self.DeltaCO = copy.deepcopy(zeroInit);

    self.compP = copy.deepcopy(zeroInit);
    self.compI = copy.deepcopy(zeroInit);
    self.compD = copy.deepcopy(zeroInit);

    self.alpha = alpha;

  def update(self, dt, error, upd):
    self.dt = dt;
    self.error[:,upd] = error[:,upd]
    self.compP[:,upd] = np.clip(self.kP * error[:,upd],a_min = 0,a_max=self.limP);

    if np.array_equal(np.sign(error[:,upd]), np.sign(self.prevError[:,upd])):
      self.compD[:,upd] = np.clip((self.error[:,upd] - self.prevError[:,upd])/self.dt * self.kD,a_min=0,a_max=self.limD);
    else:
      self.compD[:,upd] = 0;


    if self.kI == 0:
        iLimStatic = 0;
        iLimDynamic = 0;
    else:
        iLimStatic = self.limI/self.kI;
        iLimDynamic = (self.limTot - abs(self.compP + self.compD))/self.kI

    iLim = np.minimum(iLimStatic, np.maximum(iLimDynamic,0));
    self.integral = self.integral + self.error * self.dt;
    self.integral = self.integral - self.integral * self.alpha;
    self.compI = np.clip(self.integral * self.kI,a_min=0,a_max=self.limI);
    self.out = np.clip(self.compP + self.compI + self.compD,a_min=0,a_max=self.limTot);

    self.prevError[:,upd] = error[:,upd]
    self.DeltaCO = self.out - self.prevOut
    self.prevOut = self.out

  def getCO(self):
    return self.out;

  def getDeltaCO(self):
    return self.DeltaCO
  
  def getStatus(self):
    print("PID Status")
    print("E:{} P:{} I:{} D:{}").format(self.error, self.compP, self.compI, self.compD)
