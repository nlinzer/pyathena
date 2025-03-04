import numpy as np

class CoolRosen95(object):
    """Compute Lambda from Rosen & Bregman (1995)
    """
    
    def __init__(self):
        
        self.T = np.logspace(1,8,1000)
    
    @staticmethod
    def LambdaRosen95(T):
        Lambda = np.where(np.logical_and(T >= 300.0, T < 2e3),
                          2.2380e-32*T**2.0, 0.0) + \
                 np.where(np.logical_and(T >= 2e3, T < 8e3),
                         1.0012e-30*T**1.5, 0.0) + \
                 np.where(np.logical_and(T >= 8e3, T < 1e5),
                          4.6240e-36*T**2.867, 0.0) + \
                 np.where(np.logical_and(T >= 1e5, T < 4e7),
                          1.6700e-18*T**-0.65, 0.0) + \
                 np.where(T >= 4e7, 3.2217e-27*T**0.5, 0.0)
        
        return Lambda
