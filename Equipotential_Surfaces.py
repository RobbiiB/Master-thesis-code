import numpy as np
from functools import partial
from matplotlib import pyplot as plt

class Equipotential_surface():
    def __init__(self, metric_name:str, g:float = 0, a:float=0.01, M:float = 1, L:str="const"):
        self.metric_name: str = metric_name 
        self.g:float = g #metripi length parameter probably going to be of the order of the planck length#
        self.a:float = a #rotation parameter#
        self.M:float = M #mass of the black hole
        self.L_type:str = L #type of angular momentum distribution#
        
    def W(self,r,theta):
        M = self.mass_func(r)
        a = self.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        psi = self.psi(r,theta)
        
        
        W = 0.5*np.log((D*sin**2 )/ psi)
        return W
    
    def dr_W(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        a = self.a
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
        a = self.a
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
        a = self.a
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
        a = self.a
        l = self.L(r,theta)
        S = self.Sigma(r,theta)
        D = self.Delta(r)
        sin = np.sin(theta)
        return (2*M*r*(l-a*sin**2)**2 / S - l**2 + sin**2*(r**2+a**2))
    
    def dr_ln_psi(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        S = self.Sigma(r,theta)
        a = self.a
        l = self.L(r,theta)
        sin = np.sin(theta)
        psi = self.psi(r,theta)

        return (2*r*sin**2 +2*((M+M_*r)/S - 2*M*r**2/S**2)*(l-a*sin**2)**2)/psi
    
    def dth_ln_psi(self,r,theta):
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        S = self.Sigma(r,theta)
        a = self.a
        l = self.L(r,theta)
        sin = np.sin(theta)
        cos = np.cos(theta)
        psi = self.psi(r,theta)

        return (8*M*r*(a**2*sin**3*cos - a*sin*cos*l)/S + 4*M*r*sin*cos*a**2*(l-a*sin**2)**2/S**2 + 2*sin*cos*(r**2+a**2))/psi
        

    def L(self,r,theta)->float:
        if self.L_type=="const":
            L = self.L_rms()
            return L
        elif self.L_type=="Kepler":
            return self.L_kepler(r)
        elif self.L_type=="Lei et all":
            if r>9*self.M:
                return self.L_kepler(r)
            else:
                return self.L_rms()
        else:
            print("unknown L type")
            return 0.0
    
    def L_kepler(self,r)->float:
        M = self.mass_func(r)
        M_ = self.drm_func(r)
        a = self.a
        L = ((r**2 + a**2)*np.sqrt(M-M_*r) - 2*a*M*r**0.5)/(r**1.5 - 2*M*r**0.5 + a * np.sqrt(M-M_*r))
        # print(L)
        return L

    def L_rms(self)->float:
        r_ms = 9*self.M
        return self.L_kepler(r_ms)
        

    def mass_func(self, r:float )->float:
        M = self.M
        g = self.g
        if self.metric_name == "Kaz":
            m = M + r/2 - 0.5*np.sqrt(r**2 - g**2)
            return m
        elif self.metric_name == "Hay":
            m = M * (r**3/(r**3 + g**3))
            return m
        elif self.metric_name == "Bar":
            m = M * (r**2/(r**2 + g**2))**(3/2)
            return m
        elif self.metric_name == "Zha":
            m=M + 2*M*g**2/r**2 - 2*M**2*g**2/r**3 - g**2/(2*r)
            return m 
        else: 
            self.__setattr__("metric_name", "Kerr")
            return M
        
    def drm_func(self,r)->float:
        M = self.M
        g = self.g
        if self.metric_name == "Kaz":
            drm = 0.5 - 0.5*r/np.sqrt(r**2 - g**2)
            return drm
        elif self.metric_name == "Hay":
            drm = M * (3*r**2*g**3/(r**3 + g**3)**2)
            return drm
        elif self.metric_name == "Bar":
            drm = M * (3*r**2*g**2/(r**2 + g**2))**(5/2)
            return drm
        elif self.metric_name == "Zha":
            drm = -4*M*g**2/r**3 + 6*M**2*g**2/r**4 + g**2/(2*r**2)
            return drm
        else: 
            return 0.0
    


    def Delta(self,r)->float:
        return r**2 + self.a**2 - 2*r*self.mass_func(r)
    def Sigma(self,r,theta)->float:
        return r**2 + self.a**2*np.cos(theta)**2
    def f(self,r,theta)->float:
        return 1-2*r*self.mass_func(r)/self.Sigma(r,theta)
    
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
            if r[-1]<2.5*self.M:
                break
            if np.cos(th[-1])>0:
                break
            if np.sin(th[-1])<0:
                break


        return (r, th)

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)
    
eps1 = Equipotential_surface("Kerr",g = 0.5, a=0.5, L="const") 
eps2 = Equipotential_surface("Kaz",g = 0.5, a=0.5, L="const") 
eps3 = Equipotential_surface("Hay",g = 0.5, a=0.5, L="const") 
eps4 = Equipotential_surface("Zha",g = 0.5, a=0.5, L="const") 


# r= np.linspace(2.5,40,1000)
# W1 = eps1.W(r,theta=np.pi/2)
# W2 = eps2.W(r,theta=np.pi/2)
# W3 = eps3.W(r,theta=np.pi/2)
# W4 = eps4.W(r,theta=np.pi/2)

plt.figure()
# plt.plot(r, W1, c = "r", label=f"a = {eps1.metric_name}")
# plt.plot(r, W2, c = "g", label=f"a = {eps2.metric_name}")
# plt.plot(r, W3, c = "b", label=f"a = {eps3.metric_name}")
# plt.plot(r, W4, c = "y", label=f"a = {eps4.metric_name}")

N = 2000000 ## maximum number of steps (this will probably not be reached)
dr = -0.0001 ## the step siz in the r direction
th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
## inital values for different runs of r 
r_0 = 25 
r,th = eps1.solve_loop(N,r_0,dr,th_0)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c= "r", label=f"{eps1.metric_name}")
r,th = eps2.solve_loop(N,r_0,dr,th_0)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c= "b", label=f"{eps2.metric_name}")
r,th = eps3.solve_loop(N,r_0,dr,th_0)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c= "g", label=f"{eps3.metric_name}")
r,th = eps4.solve_loop(N,r_0,dr,th_0)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c= "y", label=f"{eps4.metric_name}")
# plt.legend()
# plt.ylim(-0.1,0.1)
plt.show()
