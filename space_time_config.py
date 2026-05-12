import numpy as np
from functools import partial
from matplotlib import pyplot as plt

class Spacetime_config():
    def __init__(self, metric_name:str, g_max_frac:float = 1, a:float=0.01, M:float = 1, L:str="const", g_max:float = 1, omg:str = "const",):
        self.metric_name: str = metric_name 
        self.M:float = M #mass of the black hole
        self.a:float = a*self.M #rotation parameter#
        self.L_type:str = L #type of angular momentum distribution#
        self.g_max_frac = g_max_frac
        self.g_max = g_max #the maximum value of g
        self.g=0
        self.Omg_type:str = omg
        #metric length parameter probably going to be of the order of the planck length#
        if self.metric_name=="kerr":
            self.g:float = 0
        elif self.metric_name=="Hay":
            self.g:float = g_max_frac * self.g_max * self.M
        elif self.metric_name=="Bar":
            self.g:float = g_max_frac * self.g_max * self.M
        elif self.metric_name=="KS":
            self.g:float = g_max_frac * self.M
        elif self.metric_name=="RN":
            self.g:float = g_max_frac * np.sqrt(self.M**2-self.a**2)
        
    def update_params(self,kwargs):
        # print(kwargs)
        for kwarg in kwargs:
            if kwarg == "g_max":
                self.g_max = kwargs["g_max"]
            if kwarg=="g_max_frac":
                if self.metric_name=="kerr":
                    self.g:float = 0
                elif self.metric_name=="Hay":
                    self.g:float = kwargs[kwarg] * self.g_max * self.M
                elif self.metric_name=="Bar":
                    self.g:float = kwargs[kwarg] * self.g_max * self.M
                elif self.metric_name=="KS":
                    self.g:float = kwargs[kwarg] * self.M
                elif self.metric_name=="RN":
                    self.g:float = kwargs[kwarg] * np.sqrt(self.M**2-self.a**2)
            try:
                self.__setattr__(kwarg,kwargs[kwarg])
            except:
                print(f"{kwarg} is not a valid kwarg")
    
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
        if self.metric_name == "KS":
            m = M + r/2 - 0.5*np.sqrt(r**2 - g**2)
            return m
        elif self.metric_name == "Hay":
            m = M * (r**3/(r**3 + g**3))
            return m
        elif self.metric_name == "Bar":
            m = M * (r**2/(r**2 + g**2))**(3/2)
            return m
        elif self.metric_name == "Zha": 
            m=M - g**2/(2*r)*(1-4*M/r +4*M**2/r**2)
            return m 
        elif self.metric_name == "RN": ##Reissner Nordstrom
            m = M-g**2/(2*r)
            return m 
        elif self.metric_name == "EB":
            m=M - M*np.tanh(g**2/(2*M*r))
            return m
        elif self.metric_name == "GCSV":
            m = M*np.exp(-g**2/(2*M*r))
            return m
        else: 
            self.__setattr__("metric_name", "Kerr")
            return M
        
    def drm_func(self,r)->float:
        M = self.M
        g = self.g
        if self.metric_name == "KS":
            drm = 0.5 - 0.5*r/np.sqrt(r**2 - g**2)
            return drm
        elif self.metric_name == "Hay":
            drm = M * (3*r**2*g**3/(r**3 + g**3)**2)
            return drm
        elif self.metric_name == "Bar":
            drm = M * (3*r**2*g**2/(r**2 + g**2)**(5/2))
            return drm
        elif self.metric_name == "Zha":
            drm = -4*M*g**2/r**3 + 6*M**2*g**2/r**4 + g**2/(2*r**2)
            return drm
        elif self.metric_name == "RN":
            drm = g**2/(2*r**2)
            return drm
        elif self.metric_name == "EB":
            drm = g**2/(2*r**2)/(np.cosh(g**2/(2*M*r)))**2
            return drm
        elif self.metric_name == "GCSV":
            drm= g**2/(2*r**2)*np.exp(-g**2/(2*M*r))
            return drm
        else: 
            return 0.0
    


    def Delta(self,r)->float:
        return r**2 + self.a**2 - 2*r*self.mass_func(r)
    def Sigma(self,r,theta)->float:
        return r**2 + self.a**2 * np.cos(theta)**2
    def f(self,r,theta)->float:
        return 1-2*r*self.mass_func(r)/self.Sigma(r,theta)