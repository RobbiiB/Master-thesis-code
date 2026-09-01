import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot
import matplotlib.colors as colors

class Density_distribution():
    def __init__(self, metric_name:str="Kerr", g_max_frac:float=0.0, a:float=0.01, L_type:str = "const"):
        self.eps = EqPot(metric_name=metric_name, g_max_frac=g_max_frac, a=a, L=L_type)
    
    def update_params(self,**kwargs):
        self.eps.spacetime_config.update_params(kwargs=kwargs)
        

    def zx_to_rth(self, z:np.ndarray,x:np.ndarray):
        r= x**2 + z**2
        th = np.tan(z/x)
        return r,th

    def create_point_grid(self,r_max,r_min, N_r:int=100,N_th:int=100):
        r_linspace = np.linspace(r_min,r_max,N_r)
        th_linspace = np.linspace(np.pi/4,np.pi/2, N_th, endpoint=True)

        r,th = np.meshgrid(r_linspace,th_linspace)
        return r,th
    
    def mask(self, z_max:float, x_max:float, N:int=100, r_min:float = 2):
        z_linspace = np.linspace(0,z_max,N)
        x_linspace = np.linspace(0,x_max,N)

        z,x = np.meshgrid(z_linspace,x_linspace)

        r2 = z**2 + x**2
        mask = np.where(r2<r_min**2, 0, 1)
        return mask
        
    def W(self, N:int=100, r_min:float=3, r_max:float = 30):
        r,th = self.create_point_grid(r_max=r_max,r_min=r_min, N_r=N,N_th=N)

        W_rth = self.eps.W(r=r,theta=th)

        return W_rth, (r,th)
    
    def rho(self, N:int, gamma:float, r_min:float=3, r_max:float=30, max_fill_r:float=30, normalized:bool=True,K:float|None=None, K_const_bool:bool=True, **kwarg):
        W,coords = self.W(N=N, r_min=r_min, r_max=r_max)

        h=np.exp(-W)

        r_ind=np.argmax(coords[0][0]>=max_fill_r)
        th_ind=-1
        h_0 = h[th_ind,r_ind]
        h_bar=h-h_0
        h_bar = np.where(h_bar>0,h_bar, 0.00000001)

        if not K_const_bool:
            try:
                K_func:np.ndarray=kwarg["K_func"](coords[0])
                inv_K_deriv = kwarg["inv_K_deriv"]
                C = self.solve_C(inv_K_deriv,h,coords,N)
                rho = (gamma-1)/gamma *(h/K_func + C)
            except:
                print("something went wrong")
                rho=0*h_bar
            
        
            pass


        elif type(K)==float:
            rho = ((gamma-1)*(h_bar)/(K*gamma))**(1/(gamma-1))
        else:
            rho = h_bar**(1/(gamma-1))
    
        if normalized:
            rho = rho/np.max(rho)
        
        return rho, coords
    
    def solve_C(self,inv_K_deriv,h,coords,N):
        r = coords[0].T
        th = coords[1].T
        h_transpost = h.T
        dr = (np.max(r)-np.min(r))/N

        C=np.zeros(np.shape(r))
        for n in range(N):
            C[n] = C[n-1]+ h[n]*inv_K_deriv(r[n])*dr
        
        return C.T




    def rho2(self, N:int, r_min:float=3.5, r_max:float=30,max_fill_r:float=30,normalized:bool=True): ##density distribution for gamma=1 and K=G/r
        W,coords = self.W(N=N, r_min=r_min, r_max=r_max)
        
        # I:np.ndarray = self.solve_I(W=W,coords=coords,N=N)
        # # print(I)
        # Kr = 10*self.eps.spacetime_config.M
        # K=Kr/coords[0]
        h=np.exp(-W)

        r_ind=np.argmax(coords[0][0]>=max_fill_r)
        th_ind=-1
        h_0 = h[th_ind,r_ind]
        
        gamma=5/3
        K=100
        h_bar=h-h_0
        # rho=W
        # print(rho)

        h_bar = np.where(h_bar>0,h_bar, 0.00000001)
        rho = (h_bar)**(1/(gamma-1))#((gamma-1)/(2*np.sqrt(K*gamma)))**(2/(gamma-1)) *

        if normalized:
            rho = rho/np.max(rho)

        # rho=np.where(rho<=rho[0][0], rho, rho[0][0])


        return rho, coords#, W, K, I
        
        pass
    
    def solve_I(self,W,coords, N):
        r = coords[0].T
        th = coords[1].T
        h = np.exp(-W).T
        dr = (np.max(r)-np.min(r))/N

        I=np.zeros(np.shape(r))
        for n in range(N):
            I[n] = I[n-1]+ h[n]*dr

        return I.T
        pass


    def plot_in_polar(self, data, coords, cmap = "viridis_r", log_offset:float=0.0000001, color_norm:bool=True):
        x_vals = coords[0]*np.sin(coords[1])
        y_vals = coords[0]*np.cos(coords[1])
        if color_norm:
            im = plt.pcolormesh(x_vals,y_vals, data, edgecolors="face", cmap=cmap, norm=colors.LogNorm(vmin=np.min(data)+log_offset, vmax=np.max(data)+log_offset))
        else:
            im = plt.pcolormesh(x_vals,y_vals, data, edgecolors="face", cmap=cmap, vmin=np.min([np.min(data),-np.max(data)]), vmax=-np.min([np.min(data),-np.max(data)]))
        plt.colorbar(im)
        plt.ylabel("z")
        plt.xlabel(r"$\rho$")



        
        
        
        
if __name__=="__main__":
    dd = Density_distribution(metric_name="Kerr", g_max_frac=0, a=0.5, L_type="const")
    dd2 = Density_distribution(metric_name="Hay", g_max_frac=1, a=0.5, L_type="const")


    N = 2000000 ## maximum number of steps (this will probably not be reached)
    dr = -0.0001 ## the step siz in the r direction
    th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    ## inital values for different runs of r 
    r_0 = 20
    r_1 = 25 

    rho,coords = dd.rho2(N=1000)
    rho2,coords2 = dd2.rho2(N=1000)
    # r,th = dd.eps.solve_loop(N,r_0,dr,th_0)
    # r2,th2 = dd.eps.solve_loop(N,r_1,dr,th_0)
    delta_rho = rho-rho2
    delta_rho=delta_rho
    #dd.plot_in_polar(rho,coords=coords)
    dd.plot_in_polar(rho2,coords=coords2,cmap="magma")#, color_norm=False)
    def rth_to_xz(r,th)->tuple:
        x =r*np.sin(th)
        z=-r*np.cos(th)
        return (x,z)
    # x,z = rth_to_xz(r,th)
    # plt.plot(x,z, c = "r")
    # x2,z2 = rth_to_xz(r2,th2)
    # plt.plot(x2,z2, c = "r")
    plt.show()
    