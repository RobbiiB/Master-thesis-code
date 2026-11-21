import numpy as np
import json
import bisect as bs
from matplotlib import pyplot as plt
from Equipotential_Surfaces import Equipotential_surface as EqPot
from Equipressure_Surfaces import Equipressure_surface as EqPre
from Density_distribution import Density_distribution as DenDist
from os import path

def rth_to_xz(r,th)->tuple:
    x =r*np.sin(th)
    z=-r*np.cos(th)
    return (x,z)



if __name__=="__main__":
    

    ## comment out the first set of """ to activate the corresponding block of code.   
    
    
    #### calculating the density distribution ####
    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"

    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]
    
    metric_name:str="GCSV" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
    L_type:str = "const" 



    plt.figure() 

    for j,g_frac in enumerate(g_span):
        print(f"{j+1}/{len(g_span)}")

        for i,a in enumerate(a_span):
            g_max_frac:float=g_frac
            a:float=a
            
            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type) #initializing the density distribution object 

            N_dd=1000
            gamma=5/3
            max_extension=25

            rho,coords=dd.rho(N=N_dd,gamma=gamma,max_fill_r=max_extension) #calculating the density distribution

            
            dd.plot_in_polar(data=rho,coords=coords,cmap="magma") #plotting the density in cylindrical coordinates

            
            N_eps = 2000000 
            dr = -0.0001 
            th_0 = np.pi/2+0.0001 
            rs = [25,20,15]
            for r_0 in rs:
                r,th = dd.eps.solve_loop(N=N_eps,r_0=r_0,dr=dr,th_0=th_0) #calculating the equipotential surfaces originating from r_0
                x,z = rth_to_xz(r,th)
                plt.plot(x,z, c = "g")
            
            print(f"\t{(i+1)*100//len(a_span)}%")

            png_file_path:str = path.join(folder_path,"figs","densities",f"{metric_name}_{a}_g_{g_frac}.png")
            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()
    #"""        

    
    #### Comparing metrics with Kerr ####
    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"

         
    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0,0.2,0.4,0.6,0.8,1]
    
    metric_name:str="GCSV" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
    L_type:str = "const"
    N_dd=1000
    gamma=5/3
    max_extension=25
    

    plt.figure()
    for i,a in enumerate(a_span):
        print(f"{i+1}/{len(a_span)}")
        dd_kerr = DenDist(metric_name="Kerr", g_max_frac=0,a=a,L_type=L_type) #initializing the density distribution object for the Kerr metric
        rho_kerr,coords = dd_kerr.rho(N=N_dd, gamma=gamma, max_fill_r=max_extension) #calculating the density distribution for the Kerr metric

        for j,g_frac in enumerate(g_span):
            g_max_frac:float=g_frac
            a:float=a

            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type) #initializing the density distribution object 

            rho,coords=dd.rho(N=N_dd,gamma=gamma,max_fill_r=max_extension) #calculating the density distribution 

            delta_rho = rho-rho_kerr

            dd.plot_in_polar(data=delta_rho,coords=coords,cmap="berlin",color_norm=False)

            print(f"\t{(j+1)*100//len(g_span)}%")

            png_file_path:str = path.join(folder_path,"figs","density_comp",f"{metric_name}_{a}_g_{g_frac}.png")
            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()
    #"""

    #### plotting correction vals ####
    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"


    with open("param_vals.txt", "r") as file:
        param_values = json.load(file)
        file.close
    
    # print(param_values)


    metric_name="GCSV"
    a_vals = param_values["a_vals"]
    g_vals = param_values["g"][metric_name]

    plt.figure()
    plt.plot(a_vals,g_vals, 'k')
    plt.xlabel(r"$\frac{|a|}{m}$")
    plt.ylabel(r"$\frac{|g|}{m}$")

    plt.fill_between(a_vals,g_vals, alpha=0.4, color="k")

    png_file_path:str = path.join(folder_path,"figs","g_vals",f"{metric_name}.png")

    plt.savefig(fname=png_file_path)
    plt.show()
    #"""



    #### Fixed mass accretion disk ####
    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"


    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0,0.2,0.4,0.6,0.8,1]

    metric_name:str="RN" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
    L_type:str = "const"
    N_dd=1000
    gamma=5/3
    max_extension=25 #initial maximum extension of the accretion disk 
    target_M=10 #Target accretion disk mass

    for i,a in enumerate(a_span):
        for j,g_frac in enumerate(g_span):
            g_max_frac:float=g_frac
            a:float=a

            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type) #initializing the density distribution object

            r_max_extension = max_extension

            rho,coords,r_max_extension = dd.rho_fixed_mass(target_M=target_M,N=N_dd,gamma=gamma) #disk mass matched exactly by root finding on the outer radius
            

            dd.plot_in_polar(data=rho,coords=coords,cmap="magma")
            print(f"\t{(i*len(g_span)+j+1)*100//(len(a_span)*len(g_span))}%")

            png_file_path:str = path.join(folder_path,"figs","densities_with_constant_mass",f"{metric_name}_{a}_g_{g_frac}.png")

            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()

    #"""

    #### Fixed mass accretion disk comparison with Kerr####

    # """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"
    diff_filename:str = "sum_diff.txt"
    diff2_filename:str = "sum_diff2.txt"

    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]

    metric_name:str="RN" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
    L_type:str = "const"
    N_dd:int=1000
    gamma:float=5/3
    max_extension:float=25 #initial maximum extension of the accretion disk
    target_M:float=10 #Target accretion disk mass
    measure:str = "rest"
    


    dd_kerr = DenDist(metric_name="Kerr", g_max_frac=0,a=1,L_type=L_type) #initializing the density distribution object for the kerr metric
    dd=DenDist(metric_name=metric_name,g_max_frac=0,a=1,L_type=L_type) #initializing the density distribution object


    print("Reading %s" % diff_filename)
    try:
        with open(diff_filename, "rt") as fp:
            diff_dict:dict = json.load(fp)
        print("Loaded data from %s" % diff_filename)
    except:
        print("Could not read file, starting from scratch")
        diff_dict:dict = {}

    print("Reading %s" % diff2_filename)
    try:
        with open(diff2_filename, "rt") as fp:
            diff2_dict:dict = json.load(fp)
        print("Loaded data from %s" % diff2_filename)
    except:
        print("Could not read file, starting from scratch")
        diff2_dict:dict = {}


    metric_diff_dict:dict = {}
    metric_diff2_dict:dict = {}
    for i,a in enumerate(a_span):
        dd_kerr.update_params(a=a) #updating the kerr metric spin 
        dd.update_params(a=a) #updating the metric spin

        rho_kerr,coords,r_max_kerr = dd_kerr.rho_fixed_mass(target_M=target_M, N=N_dd,gamma=gamma,measure=measure)
        print("########")
        print(dd_kerr.disk_mass(rho=rho_kerr, coords=coords, measure="rest")-target_M)

        metric_diff_dict[a] = []
        metric_diff2_dict[a] = []

        for j,g_frac in enumerate(g_span):
            dd.update_params(g_max_frac=g_frac) #updating the metric deformation parameter

            rho,coords,r_max = dd.rho_fixed_mass(target_M=target_M, N=N_dd,gamma=gamma,measure=measure)
            # print(dd.disk_mass(rho=rho, coords=coords, measure="rest")-target_M)
            

            delta_rho = rho-rho_kerr #calculating the difference with the kerr metric 

            metric_diff_dict[a].append(np.sum(delta_rho))
            metric_diff2_dict[a].append(dd.disk_mass(rho=rho,coords=coords,measure=measure)-dd_kerr.disk_mass(rho=rho_kerr,coords=coords,measure=measure))


            dd.plot_in_polar(data=delta_rho,coords=coords,cmap="seismic",color_norm=False)
            print(f"\t{(i*len(g_span)+j + 1)*100//(len(a_span)*len(g_span))}%")
                
            png_file_path:str = path.join(folder_path,"figs","densities_with_const_mass_comp",f"{metric_name}_{a}_g_{g_frac}.png")

            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()
    
    diff_dict[metric_name] = metric_diff_dict
    diff2_dict[metric_name] = metric_diff2_dict

    print("Overwriting %s" % diff_filename)
    with open(diff_filename, "wt") as fp:
        json.dump(diff_dict, fp)
    print("Overwriting %s" % diff2_filename)
    with open(diff2_filename, "wt") as fp:
        json.dump(diff2_dict, fp)
    
    
    #"""
            
    #### creating plots for diff per g ####

    # """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"
    diff_filename:str = "sum_diff.txt"
    diff2_filename:str = "sum_diff2.txt"
    dpi = 300

    with open(diff_filename, "rt") as fp:
        diff_dict = json.load(fp)
    with open(diff2_filename, "rt") as fp:
        diff2_dict = json.load(fp)
    
    metric_names:list = ["RN"] #["Hay","Bar", "EB", "GCSV", "KS", 

    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]
    
    def color_line(i, len_a_span:int = len(a_span)):
        r = 255//len_a_span * i
        g = 0
        b = 255//len_a_span * (len_a_span-i)

        return '#%02x%02x%02x' % (r, g, b)

    for metric_name in metric_names:
        plt.figure()
        for i,a in enumerate(a_span):
            diff = diff_dict[metric_name][str(a)]
            plt.plot(g_span, diff, label=f"a = {a}", color=color_line(i=i))
        
        png_file_path:str = path.join(folder_path,"figs","sum_diff",f"{metric_name}.png")

        plt.legend()
        plt.xlabel(r"$\frac{|g|}{m}$")
        plt.ylabel(r"$\sum (\rho-\rho_{kerr})$")
        # plt.show()
        plt.savefig(dpi = dpi,fname=png_file_path)

    
        plt.figure()
        for i,a in enumerate(a_span):
            diff2 = diff2_dict[metric_name][str(a)]
            print(g_span[np.argmax(diff2)])
            plt.plot(g_span, diff2, label=f"a = {a}", color=color_line(i=i))
        
        png_file_path:str = path.join(folder_path,"figs","sum_diff2",f"{metric_name}.png")

        plt.legend()
        plt.xlabel(r"$\frac{|g|}{m}$")
        plt.ylabel(r"$\Delta M$")
        # plt.show()
        plt.savefig(dpi = dpi, fname=png_file_path)


    



    # """