import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot

class Density_distribution():
    def __init__(self, metric_name:str="Kerr", g_max_frac:float=0.0, a:float=0.01, L_type:str = "const"):
        self.eps = EqPot(metric_name=metric_name, g_max_frac=g_max_frac, a=a, L=L_type)
    
    def zx_to_rth(self, z:np.ndarray,x:np.ndarray):
        r= x**2 + z**2
        th = np.tan(z/x)
        return r,th

    def create_point_grid(self,r_max,r_s, N_r:int=100,N_th:int=100):
        r_linspace = np.linspace(r_s,r_max,N_r)
        th_linspace = np.linspace(0,np.pi/2, N_th)

        r,th = np.meshgrid(r_linspace,th_linspace)
        return r,th
    
    def mask(self, z_max:float, x_max:float, N:int=100, r_s:float = 2):
        z_linspace = np.linspace(0,z_max,N)
        x_linspace = np.linspace(0,x_max,N)

        z,x = np.meshgrid(z_linspace,x_linspace)

        r2 = z**2 + x**2
        mask = np.where(r2<r_s**2, 0, 1)
        return mask
        
    def W(self,z_max:float,x_max:float, N:int=100, r_s:float=3):
        r,th = self.create_point_grid(r_max=30,r_s=r_s)

        W_rth = self.eps.W(r=r,theta=th)

        return W_rth
        pass
        
if __name__=="__main__":
    dd = Density_distribution(metric_name="Kerr", g_max_frac=0, a=0.5, L_type="const")
    plt.figure()
    W=dd.W(z_max=30,x_max=30,N=100,r_s=2)
    plt.imshow(W)
    plt.plot()