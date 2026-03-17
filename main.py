import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot
from Equipressure_Surfaces import Equipressure_surface as EqPre

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)



if __name__=="__main__":
    eqpre = EqPre("Kaz",g_max_frac = 0.3, a=0.5, L="const") 
    eqpot = EqPot("Kerr",g_max_frac = 0.9, a=0.5, L="const") 


    N = 2000000 ## maximum number of steps (this will probably not be reached)
    dr = -0.0001 ## the step siz in the r direction
    th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    r_0 = 25 ## inital values for different runs of r 

    plt.figure()
    r,th = eqpre.solve_loop(N, r_0,dr,th_0)
    x,z=rth_to_xz(r,th)
    plt.plot(x,z, c="r", label=f"Equipressure surface")

    
    r,th = eqpot.solve_loop(N, r_0,dr,th_0)
    x,z=rth_to_xz(r,th)
    plt.plot(x,z, c="b", label=f"Equipotential surface")
    
    plt.legend()
    plt.show()