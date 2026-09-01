import numpy as np
import json
import bisect as bs
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot
from Equipressure_Surfaces import Equipressure_surface as EqPre
from Density_distribution import Density_distribution as DenDist

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)



if __name__=="__main__":
    
    #### Comparison kerr hayward bardeen ####
    """
    # with open("param_vals.txt", "r") as file:
    #     param_values = json.load(file)
    #     file.close
    # a_vals = param_values["a_vals"]
    # g_max_hay = param_values["g_hay"]
    # g_max_bar = param_values["g_bar"]
    
    eqpot = EqPot("Bar",g_max_frac = 1, a=0.7, L="const") 
    # eqpot2 = EqPot("Hay",g_max_frac = 1, a=0.7, L="const") 
    dd = DenDist(metric_name="Bar",g_max_frac = 1, a=0, L_type="const")
    # dd_kerr = DenDist(metric_name="Kerr", g_max=1 ,g_max_frac = 1, a=0, L_type="const")
    
    N: int = 2000000 ## maximum number of steps (this will probably not be reached)
    dr: float = -0.0001 ## the step siz in the r direction
    th_0: float = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    r_0s: list = [10,13,16,19,22,25] ## inital values for different runs of r 
    g_span = [0,0.2,0.4,0.6,0.8,1] #
    a_span = [0.5,0.6,0.7,0.8,0.9,1] #

    


    for a in a_span:
        # index_g_max_val = bs.bisect_left(a_vals,a)
        # eqpot.update_params(g_max = g_max_bar[index_g_max_val])
        # eqpot2.update_params(g_max = g_max_hay[index_g_max_val])
        # eqpot_kerr.update_params(g_max = g_max_bar[index_g_max_val])
        for g in g_span:
            # eqpot_kerr.update_params(a=a)
            eqpot.update_params(a=a,g_max_frac=g)
            # eqpot.update_params(g_max_frac=0,a=a)
            # eqpot_bar.update_params(g_max_frac=g,a=a)
            # print(eqpot_hay.g, eqpot_hay.g_max)
            dd.update_params(a=a,g_max_frac=g)
            # dd_kerr.update_params(a=a,g_max_frac=g)

            plt.figure()
            W, coords=dd.W(N=1000,r_min=4, r_max=25)
            # W_kerr, coords_kerr = dd_kerr.rho(N=1000,r_min=4, r_max=25,gamma=2,K=0.5)
            dd.plot_in_polar(data=W ,coords=coords, log_offset=0.0001)
            for i,r_0 in enumerate(r_0s):
                

                r,th = eqpot.solve_loop(N,r_0,dr,th_0)
                x,z = rth_to_xz(r,th)

                # r2,th2 = eqpot2.solve_loop(N,r_0,dr,th_0)
                # x2,z2 = rth_to_xz(r2,th2)

                print(f"a: {a}, g: {g}, percentage: {int(100*(i+1)/len(r_0s))}%")
                
                if r_0==r_0s[0]:
                    plt.plot(x,z,color="#1B1918",linestyle="-")#, label=f"{eqpot.spacetime_config.metric_name}, a={a}")
                    # plt.plot(x2,z2,color="#1B1918",linestyle="-")#, label=f"{eqpot.spacetime_config.metric_name}, a={a}")
                
                else:
                    plt.plot(x,z,color="#1B1918",linestyle="-")
                    # plt.plot(x2,z2,color="#1B1918",linestyle="-")
            
            
            plt.ylabel("z")
            plt.xlabel(r"$\rho$")
            # plt.legend()
            plt.savefig(fname=f"/Users/robin/Documents/Master thesis 1/figs/Potential_stuff/bar_{a}_g_{g}.pdf")
            # plt.show()
    #"""
    
    
    ##Kazakov-solodukhin
    """
    #### comparison kazakov-solodukhin and kerr ####

    eqpot_kerr = EqPot("Kerr",g_max_frac = 0, a=0.5, L="const") 
    eqpot_kaz = EqPot("Kaz",g_max_frac = 0, a=0.5, L="const") 


    N: int = 2000000 ## maximum number of steps (this will probably not be reached)
    dr: float = -0.0001 ## the step siz in the r direction
    th_0: float = np.pi/2+0.0001 ## the initial value of theta, not that it is not exactly 0.5*pi as that would be problematic 
    r_0s: list = [10,13,16,19,22,25] ## inital values for different runs of r 
    a_vals = [0.5,0.6,0.7,0.8,0.9,1]
    g_vals = [0.2,0.4,0.6,0.8,1]

    for a in a_vals:
        for g in g_vals:
            eqpot_kaz.update_params(g_max_frac=g,a=a)
            eqpot_kerr.update_params(g_max_frac=g,a=a)

            plt.figure()
            for i,r_0 in enumerate(r_0s):
                r_kerr,th_kerr = eqpot_kerr.solve_loop(N,r_0,dr,th_0)
                x_kerr,z_kerr=rth_to_xz(r_kerr,th_kerr)

                r_kaz,th_kaz = eqpot_kaz.solve_loop(N,r_0,dr,th_0)
                x_kaz, z_kaz = rth_to_xz(r_kaz,th_kaz)

                print(f"a: {a}, g: {g}, percentage: {int(100*(i+1)/len(r_0s))}%")
                if r_0==r_0s[0]:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-", label=f"{eqpot_kerr.metric_name}, a={eqpot_kerr.a}")
                    plt.plot(x_kaz,z_kaz,color="#bc0031",linestyle="-.", label=f"{eqpot_kaz.metric_name}, a={eqpot_kaz.a}, g={eqpot_kaz.g}")
                else:
                    plt.plot(x_kerr,z_kerr,color="#1B1918",linestyle="-")
                    plt.plot(x_kaz,z_kaz,color="#bc0031",linestyle="-.")
            plt.ylabel("z")
            plt.xlabel(r"$\rho$")
            plt.legend()
            plt.savefig(fname=f"/Users/robin/Documents/Master thesis 1/figs/{eqpot_kaz.g_max_frac}_{eqpot_kaz.a}_kazakov.pdf")
            plt.show()
    #"""
    
    
    
    ##calculating the density distribution
    """
    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]
    
    metric_name:str="GCSV"
    L_type:str = "const"



    plt.figure()
    for j,g_frac in enumerate(g_span):
        print(f"{j+1}/{len(g_span)}")
        for i,a in enumerate(a_span):
            g_max_frac:float=g_frac
            a:float=a
            

            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type)

            N_dd=1000
            gamma=5/3
            max_extension=25

            rho,coords=dd.rho(N=N_dd,gamma=gamma,max_fill_r=max_extension)

            
            dd.plot_in_polar(data=rho,coords=coords,cmap="magma")

            N_eps = 2000000 
            dr = -0.0001 
            th_0 = np.pi/2+0.0001 
            rs = [25,20,15]
            for r_0 in rs:
                r,th = dd.eps.solve_loop(N=N_eps,r_0=r_0,dr=dr,th_0=th_0)
                x,z = rth_to_xz(r,th)
                plt.plot(x,z, c = "g")
            
            print(f"\t{(i+1)*100//len(a_span)}%")
            plt.savefig(fname=f"/Users/robin/Documents/Master thesis 1/figs/densities/{metric_name}_{a}_g_{g_frac}.pdf")
            # plt.show()
            plt.clf()
    #"""        

    
    ##Comparing metrics with Kerr
    """
         
    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0,0.2,0.4,0.6,0.8,1]
    
    metric_name:str="GCSV"
    L_type:str = "const"
    N_dd=1000
    gamma=5/3
    max_extension=25
    

    plt.figure()
    for i,a in enumerate(a_span):
        print(f"{i+1}/{len(a_span)}")
        dd_kerr = DenDist(metric_name="Kerr", g_max_frac=0,a=a,L_type=L_type)
        rho_kerr,coords = dd_kerr.rho(N=N_dd, gamma=gamma, max_fill_r=max_extension)

        for j,g_frac in enumerate(g_span):
            g_max_frac:float=g_frac
            a:float=a

            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type)

            rho,coords=dd.rho(N=N_dd,gamma=gamma,max_fill_r=max_extension)

            delta_rho = rho-rho_kerr

            dd.plot_in_polar(data=delta_rho,coords=coords,cmap="berlin",color_norm=False)
            print(f"\t{(j+1)*100//len(g_span)}%")


            plt.savefig(fname=f"/Users/robin/Documents/Master thesis 1/figs/density_comp/{metric_name}_{a}_g_{g_frac}.pdf")
            # plt.show()
            plt.clf()


            #'/Users/robin/Documents/Master thesis 1/figs/density_comp'
    #"""

    ##plotting correction vals

    with open("param_vals.txt", "r") as file:
        param_values = json.load(file)
        file.close
    
    # print(param_values)


    metric_name="GCSV"
    a_vals = param_values["a_vals"]
    g_vals = param_values["g"][metric_name]

    plt.figure()
    plt.plot(a_vals,g_vals, 'k')
    plt.xlabel("|a|/m")
    plt.ylabel("|g|/m")

    plt.fill_between(a_vals,g_vals, alpha=0.4, color="k")
    plt.savefig(f"/Users/robin/Documents/Master thesis 1/figs/g_vals/{metric_name}.pdf")
    plt.show()