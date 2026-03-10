import numpy as np
from functools import partial
from matplotlib import pyplot as plt
# from scipy import integrate as int

    
class Equipressure_surface():
    def __init__(self, metric_name:str, g:float = 0, a:float=0.01, M:float = 1, L:str="const", omg:str = "const"):
        self.metric_name: str = metric_name 
        self.g:float = g #metripi length parameter probably going to be of the order of the planck length#
        self.a:float = a #rotation parameter#
        self.M:float = M #mass of the black hole
        self.L_type:str = L #type of angular momentum distribution#
        self.Omg_type:str = omg #type of angular frequency distribution#
        
    
    def diff_func(self, r,theta)->float:
        ##The function of the differential equation using the angular momentum distribution and metrics with upper indices##

        L = self.L(r,theta)

        num = self.drg_utt(r,theta) - 2*L*self.drg_utp(r,theta) + L**2 *self.drg_upp(r,theta)
        den = self.dtg_utt(r,theta) - 2*L*self.dtg_utp(r,theta) + L**2 *self.dtg_upp(r,theta)

        return -num/den
    
    def diff_func2(self,r,theta)->float:
        ##The function of the differential equation using the angular frequency distribution and metrics with lower indices##
        Omg = self.Omega(r,theta)

        num = self.drg_tt(r,theta)  + 2*Omg*self.drg_tp(r,theta)  + Omg**2*self.drg_pp(r,theta)
        den = self.dthg_tt(r,theta) + 2*Omg*self.dthg_tp(r,theta) + Omg**2*self.dthg_pp(r,theta)

        return -num/den

    def drg_utt(self,r,theta)->float:
        ##the r derivative of the tt component of the metric with upper indices##
        a = self.a
        M=self.mass_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta) 

        drg_tt=self.drg_utp(r,theta)*(r**2+a**2)/a - (4*M*r**2)/(S*D)
        return drg_tt
    
    def drg_utp(self,r,theta)->float:
        ##the r derivative of the t phi component of the metric with upper indices##
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta) 
        drg_tp=-2*a* ((M + M_*r)/(S*D) - (2*M*r**2)/(D*S**2) - (2*M*r*(r - M - M_*r))/(S*D**2))
        return drg_tp
    
    def drg_upp(self,r,theta)->float:
        ##the r derivative of the phi phi component of the metric with upper indices##
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        sin = np.sin(theta)

        drg_pp = self.drg_utp(r,theta)/(a*sin**2) - (2*r - 2*M - 2*M_*r)/(D**2*sin**2)

        return drg_pp
    
    def dtg_utt(self,r,theta)->float:
        ##the theta derivative of the tt component of the metric with upper indices##
        a = self.a
        
        dtg_tt = self.dtg_utp(r,theta)*(r**2+a**2)/a
        
        return dtg_tt
    
    def dtg_utp(self,r,theta)->float:
        ##the theta derivative of the t phi component of the metric with upper indices##
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        dtg_tp = -(4*M*r*sin*cos*a**3)/(D*S**2)
        
        return dtg_tp
    
    def dtg_upp(self,r,theta)->float:
        ##the theta derivative of the phi phi component of the metric with upper indices##
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        dtg_pp = self.dtg_utp(r,theta)/(a*sin**2) - 2*cos*(1 - 2*M*r/S)/(D*sin**3)
        
        return dtg_pp
    
    def drg_tt(self,r,theta)->float:
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        drg_tt = 2*(-M*r**2 + M*a**2*cos**2 + M_*r*S)/S**2

        return drg_tt
    
    def drg_tp(self,r,theta)->float:
        a = self.a
        sin = np.sin(theta)

        drg_tp = -a*sin**2*self.drg_tt(r,theta)

        return drg_tp
    
    def drg_pp(self,r,theta)->float:
        a = self.a
        sin = np.sin(theta)

        drg_pp = 2*r*sin**2 + a**2*sin**4*self.drg_tt(r,theta)

        return drg_pp
    
    def dthg_tt(self,r,theta)->float:
        a = self.a
        M=self.mass_func(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        dthg_tt = 4*a**2*M*r*sin*cos/S**2

        return dthg_tt

    def dthg_tp(self,r,theta)->float:
        a = self.a
        M=self.mass_func(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        dthg_tp = -4*a*M*r*sin*cos*(r**2 + a**2)/S**2
        
        return dthg_tp
    
    def dthg_pp(self,r,theta)->float:
        a = self.a
        M=self.mass_func(r)
        M_=self.drm_func(r)
        D=self.Delta(r)
        S=self.Sigma(r,theta)
        sin = np.sin(theta)
        cos=np.cos(theta)

        dthg_pp = 2*sin*cos*(r**2 + a**2) + 2*M*r*a**2*sin**3*cos*(2*r**2 + (1 + cos**2)*a**2)/S**2

        return dthg_pp

    def Omega(self,r,theta)->float:
        if self.Omg_type =="const":
            return 10.0
        if self.Omg_type =="kepler":
            num = np.sqrt(self.mass_func(r)-r*self.drm_func(r))
            den = r**1.5 + self.a*np.sqrt(self.mass_func(r)-r*self.drm_func(r))
            print("Omega", num/den)
            return num/den
        else:
            print("unknown Omega type")
            return 0.0

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
    
    def solve_loop(self,N,r_0,dr, th_0 = np.pi/2)-> tuple:
        th = np.array([th_0])
        r = np.array([r_0])
        dth = 0

        for _ in range(N):
            r = np.append(r,r[-1]+dr)
            dth =  dr*self.diff_func(r[-1],th[-1])

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
    
    def solve_loop2(self,N,r_0,dth,th_0 = np.pi/2):
        th = np.array([th_0])
        r = np.array([r_0])
        dr = 0

        for _ in range(N):
            dr = dth/self.diff_func(r[-1],th[-1])
            th = np.append(th,th[-1]+dth)
            r = np.append(r,r[-1]+dr)
            if r[-1]<2.5*self.M:
                break
            if np.cos(th[-1])>0:
                break
        
        return r, th

    # def scipy_int(self, r_0, th_0):
    #     def zero_crossing_th(r,th):
    #         return np.cos(th[-1])
    #     zero_crossing_th.terminal = True # type: ignore

    #     sol = int.solve_ivp(self.diff_func, [r_0,2.5*self.M],[r_0,th_0], events=zero_crossing_th, method="BDF")

    #     r = sol.y[0]
    #     th = sol.y[1]
    #     print(th)
    #     return r,th

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)

##create the figure to plot##
plt.figure()

##initializing the equipressure surface calculators##
eps = Equipressure_surface("Kerr",g = 0.9, a=0.5, L="const") 
eps2 = Equipressure_surface("Kaz",g = 0.9, a=0.5, L="const")
eps3 = Equipressure_surface("Hay",g = 0.9, a=0.5, L="const")
eps4 = Equipressure_surface("Zha",g = 0.9, a=0.5, L="const")


N = 2000000 ## maximum number of steps (this will probably not be reached)
dr = -0.0001 ## the step siz in the r direction
th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
## inital values for different runs of r 
r_0 = 25 
r_01 = 50
r_02 = 12

### Solving the EPS for the different Black hole parameter/starting conditions ###


r,th = eps.solve_loop(N,r_0,dr,th_0)
# print(th)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c = "r", label=f"{eps.metric_name}")
r,th = eps.solve_loop(N,r_02,dr,th_0)
# print(th)
x,z = rth_to_xz(r,th)
plt.plot(x,z, c = "r")


r,th = eps2.solve_loop(N,r_0,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "b", label=f"{eps2.metric_name}")
r,th = eps2.solve_loop(N,r_02,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "b")

r,th = eps3.solve_loop(N,r_0,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "g", label=f"{eps3.metric_name}")
r,th = eps3.solve_loop(N,r_02,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "g")

r,th = eps4.solve_loop(N,r_0,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "y", label=f"{eps4.metric_name}")
r,th = eps4.solve_loop(N,r_02,dr,th_0)
# print(th)
x,z = rth_to_xz(r[:-1],th[:-1])
plt.plot(x,z, c = "y")

## plotting the plots
plt.legend()
plt.show()