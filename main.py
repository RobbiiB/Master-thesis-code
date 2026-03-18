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
    eqpre = EqPre("Hay",g_max_frac = 0, a=0.5, L="const") 
    eqpre_kerr = EqPre("Kerr",g_max_frac = 0, a=0.5, L="const") 
    
    N = 2000000 ## maximum number of steps (this will probably not be reached)
    dr = -0.0001 ## the step siz in the r direction
    th_0 = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    r_0s = [10,13,16,19,21,25] ## inital values for different runs of r 
    g_span = [0,0.2,0.4,0.6,0.8,1]
    a_span = [0.5,0.6,0.7,0.8,0.9,1]

    for a in a_span:
        for g in g_span:
            eqpre.update_params(g_max_frac=g,a=a)

            plt.figure()
            for r_0 in r_0s:
                r_kerr,th_kerr = eqpre_kerr.solve_loop(N,r_0,dr,th_0)
                x_kerr,z_kerr=rth_to_xz(r_kerr,th_kerr)

                r,th = eqpre.solve_loop(N,r_0,dr,th_0)
                x,z = rth_to_xz(r,th)
                if r_0==r_0s[0]:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-", label=f"{eqpre_kerr.metric_name}, a={eqpre_kerr.a}")
                    plt.plot(x,z,color="#bc0031",linestyle="-.", label=f"{eqpre.metric_name}, g={eqpre.g}, a={eqpre.a}")
                else:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-")
                    plt.plot(x,z,color="#bc0031",linestyle=":")
            plt.ylabel("z")
            plt.xlabel(r"$\rho$")
            plt.legend()
            plt.savefig(fname=f"{eqpre.metric_name}_{eqpre.g}_{eqpre.a}_comp_kerr.pdf")
            plt.show()


