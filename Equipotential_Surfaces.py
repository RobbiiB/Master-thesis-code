import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from space_time_config import Spacetime_config as stc

class Equipotential_surface():
    def __init__(self, metric_name:str, g_max_frac:float = 0, a:float=0.01, M:float = 1, L:str="const", g_max:float = 0):
        self.spacetime_config = stc(metric_name=metric_name, g_max_frac=g_max_frac,a = a, M = M, L=L,g_max = g_max)
        
    def update_params(self,**kwargs):
        self.spacetime_config.update_params(kwargs=kwargs)
        
    def W(self,r,theta):
        D = self.Delta(r)
        sin = np.sin(theta)
        psi = self.psi(r,theta)    
        W = 0.5*np.log(np.abs((D*sin**2 )/ psi))
        return W
    
    def dr_W(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        cos = np.cos(theta)
        psi = self.psi(r,theta)

        dr_W = ((r-M-M_*r)*sin*psi - D*sin**2*(l-a*sin)**2 * (M*a**2*cos**2 - M*r**2 +M_*r*S)/S**2 - r*sin**2)
        return dr_W
        
    def dth_W(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        cos = np.cos(theta)
        psi = self.psi(r,theta)

        dth_W = sin*cos*(D*psi-D*sin**2*2*a*M*r*(l**2*a +2*a*sin**2*(r**2+a**2) - 2*l*(r**2+a**2))/S**2-r**2-a**2)
        return dth_W

    def dr_th(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        cos = np.cos(theta)
        psi = self.psi(r,theta)

        num = sin*(2*r-2*M-2*M_*r-D*self.dr_ln_psi(r,theta))
        den = D*(2*cos-sin*self.dth_ln_psi(r,theta))

        return -num/den

    def psi(self,r,theta):
        M = self.mass_func(r)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        #return (2*M*r*(l-a*sin**2)**2 / S - l**2 + sin**2*(r**2+a**2))
        return l**2 * (2*M*r/S - 1) - l*4*r*a*sin**2/S + sin**2 * (r**2 + a**2 +2*M*r*a**2*sin**2/S)

    def dr_ln_psi(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        S = self.Sigma(r,theta)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        sin = np.sin(theta)
        psi = self.psi(r,theta)

        return (2*r*sin**2 +2*((M+M_*r)/S - 2*M*r**2/S**2)*(l-a*sin**2)**2)/psi
    
    def dth_ln_psi(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        S = self.Sigma(r,theta)
        a = self.spacetime_config.a
        l = self.L(r,theta)
        sin = np.sin(theta)
        cos = np.cos(theta)
        psi = self.psi(r,theta)

        return (8*M*r*(a**2*sin**3*cos - a*sin*cos*l)/S + 4*M*r*sin*cos*a**2*(l-a*sin**2)**2/S**2 + 2*sin*cos*(r**2+a**2))/psi
        
    def L(self,r,theta)->float:
        return self.spacetime_config.L(r,theta)
    
    def L_kepler(self,r)->float:
        return self.spacetime_config.L_kepler(r)

    def L_rms(self)->float:
        return self.spacetime_config.L_rms()
        
    def mass_func(self, r:float )->float:
        return self.spacetime_config.mass_func(r)
        
    def drm_func(self,r)->float:
        return self.spacetime_config.drm_func(r)
    
    def Delta(self,r)->float:
        return self.spacetime_config.Delta(r)
    
    def Sigma(self,r,theta)->float:
        return self.spacetime_config.Sigma(r,theta)
    
    def f(self,r,theta)->float:
        return self.spacetime_config.f(r,theta)
    
    def solve_loop(self,N,r_0,dr, th_0 = np.pi/2):
        th = np.array([th_0])
        r = np.array([r_0])
        dth = 0

        for _ in range(N):
            r = np.append(r,r[-1]+dr)
            dth =  dr*self.dr_th(r[-1],th[-1])

            # print(dth)
            th = np.append(th,th[-1]+dth)
            
            # print(r)
            # print(r)
            # print(th)
            if r[-1]<2.5*self.spacetime_config.M:
                break
            if np.cos(th[-1])>0:
                r=r[:-1]
                th=th[:-1]
                break
            if np.sin(th[-1])<0:
                break


        return (r, th)

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)


if __name__=="__main__":
    eps1 = Equipotential_surface("Kerr",g_max_frac = 0.5, a=0.5, L="const") 
    eps2 = Equipotential_surface("Kaz",g_max_frac = 0.5, a=0.5, L="const") 
    eps3 = Equipotential_surface("Hay",g_max_frac = 0.5, a=0.5, L="const") 
    eps4 = Equipotential_surface("Zha",g_max_frac = 0.5, a=0.5, L="const") 


    r= np.linspace(2.5,40,1000)
    W1 = eps1.W(r,theta=np.pi/2)
    W2 = eps2.W(r,theta=np.pi/2)
    W3 = eps3.W(r,theta=np.pi/2)
    W4 = eps4.W(r,theta=np.pi/2)

    plt.figure()
    plt.plot(r, W1, c = "r", label=f"a = {eps1.spacetime_config.metric_name}")
    plt.plot(r, W2, c = "g", label=f"a = {eps2.spacetime_config.metric_name}")
    plt.plot(r, W3, c = "b", label=f"a = {eps3.spacetime_config.metric_name}")
    plt.plot(r, W4, c = "y", label=f"a = {eps4.spacetime_config.metric_name}")

    # N = 2000000 ## maximum number of steps (this will probably not be reached)
    # dr = -0.0001 ## the step siz in the r direction
    # th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    # ## inital values for different runs of r 
    # r_0 = 25 
    # r,th = eps1.solve_loop(N,r_0,dr,th_0)
    # x,z = rth_to_xz(r,th)
    # plt.plot(x,z, c= "r", label=f"{eps1.metric_name}")
    # r,th = eps2.solve_loop(N,r_0,dr,th_0)
    # x,z = rth_to_xz(r,th)
    # plt.plot(x,z, c= "b", label=f"{eps2.metric_name}")
    # r,th = eps3.solve_loop(N,r_0,dr,th_0)
    # x,z = rth_to_xz(r,th)
    # plt.plot(x,z, c= "g", label=f"{eps3.metric_name}")
    # r,th = eps4.solve_loop(N,r_0,dr,th_0)
    # x,z = rth_to_xz(r,th)
    # plt.plot(x,z, c= "y", label=f"{eps4.metric_name}")
    # plt.legend()
    # plt.ylim(-0.1,0.1)
    plt.show()
