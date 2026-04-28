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
        th_linspace = np.linspace(np.pi/4,np.pi/2, N_th, endpoint=True)

        r,th = np.meshgrid(r_linspace,th_linspace)
        return r,th
    
    def mask(self, z_max:float, x_max:float, N:int=100, r_s:float = 2):
        z_linspace = np.linspace(0,z_max,N)
        x_linspace = np.linspace(0,x_max,N)

        z,x = np.meshgrid(z_linspace,x_linspace)

        r2 = z**2 + x**2
        mask = np.where(r2<r_s**2, 0, 1)
        return mask
        
    def W(self, N:int=100, r_s:float=3, r_max:float = 30):
        r,th = self.create_point_grid(r_max=r_max,r_s=r_s, N_r=N,N_th=N)

        W_rth = self.eps.W(r=r,theta=th)

        return W_rth, (r,th)
    
    def plot_in_polar(self, W, coords ):
        x_vals = coords[0]*np.sin(coords[1])
        y_vals = coords[0]*np.cos(coords[1])
        im = plt.pcolormesh(x_vals,y_vals, W, edgecolors="face")
        plt.colorbar(im)
        plt.ylabel("z")
        plt.xlabel(r"$\rho$")
        
        
        
        
if __name__=="__main__":
    dd = Density_distribution(metric_name="Kerr", g_max_frac=0, a=0.9, L_type="const")

    W, coords=dd.W(N=1000,r_s=2.5)

    dd.plot_in_polar(W,coords)

    plt.show()
    