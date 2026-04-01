import numpy as np
import json
import bisect as bs
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot
from Equipressure_Surfaces import Equipressure_surface as EqPre

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)



if __name__=="__main__":
    with open("param_vals.txt", "r") as file:
        param_values = json.load(file)
        file.close
    a_vals = param_values["a_vals"]
    g_max_hay = param_values["g_hay"]
    g_max_bar = param_values["g_bar"]
    
    eqpot_hay = EqPot("Hay",g_max_frac = 0, a=0.5, L="const") 
    eqpot_bar = EqPot("Bar",g_max_frac = 0, a=0.5, L="const") 
    eqpot_kerr = EqPot("Kerr",g_max_frac = 0, a=0.5, L="const") 
    
    N: int = 2000000 ## maximum number of steps (this will probably not be reached)
    dr: float = -0.0001 ## the step siz in the r direction
    th_0: float = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    r_0s: list = [10,13,16,19,22,25] ## inital values for different runs of r 
    g_span = [0,0.2,0.4,0.6,0.8,1]
    a_span = [0.5,0.6,0.7,0.8,0.9,1] #

    


    for a in a_span:
        index_g_max_val = bs.bisect_left(a_vals,a)
        eqpot_hay.update_params(g_max = g_max_hay[index_g_max_val])
        eqpot_bar.update_params(g_max = g_max_bar[index_g_max_val])
        for g in g_span:
            eqpot_kerr.update_params(a=a)
            eqpot_hay.update_params(g_max_frac=g,a=a)
            eqpot_bar.update_params(g_max_frac=g,a=a)
            # print(eqpot_hay.g, eqpot_hay.g_max)

            plt.figure()
            for i,r_0 in enumerate(r_0s):
                
                r_kerr,th_kerr = eqpot_kerr.solve_loop(N,r_0,dr,th_0)
                x_kerr,z_kerr=rth_to_xz(r_kerr,th_kerr)

                r_hay,th_hay = eqpot_hay.solve_loop(N,r_0,dr,th_0)
                x_hay,z_hay = rth_to_xz(r_hay,th_hay)

                r_bar,th_bar = eqpot_bar.solve_loop(N,r_0,dr,th_0)
                x_bar,z_bar = rth_to_xz(r_bar,th_bar)

                print(f"a: {a}, g: {g}, percentage: {int(100*(i+1)/len(r_0s))}%")
                
                if r_0==r_0s[0]:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-", label=f"{eqpot_kerr.metric_name}, a={eqpot_kerr.a}")
                    plt.plot(x_hay,z_hay,color="#bc0031",linestyle="-.", label=f"{eqpot_hay.metric_name}, g={int(100*eqpot_hay.g)/100}, a={eqpot_hay.a}")
                    plt.plot(x_bar,z_bar,color="#1d7492",linestyle=":", label=f"{eqpot_bar.metric_name}, g={int(100*eqpot_bar.g)/100}, a={eqpot_bar.a}")
                else:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-")
                    plt.plot(x_hay,z_hay,color="#bc0031",linestyle="-.")
                    plt.plot(x_bar,z_bar,color="#1d7492",linestyle=":")
            plt.ylabel("z")
            plt.xlabel(r"$\rho$")
            plt.legend()
            plt.savefig(fname=f"{eqpot_hay.g_max_frac}_{eqpot_hay.a}_comp_kerr.pdf")
            plt.show()

