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
    measure:str = "rest"

    for i,a in enumerate(a_span):
        for j,g_frac in enumerate(g_span):
            g_max_frac:float=g_frac
            a:float=a

            dd=DenDist(metric_name=metric_name,g_max_frac=g_max_frac,a=a,L_type=L_type) #initializing the density distribution object

            r_max_extension = max_extension

            rho,coords,r_max_extension = dd.rho_fixed_mass(target_M=target_M,N=N_dd,gamma=gamma, measure=measure) #disk mass matched exactly by root finding on the outer radius
            

            dd.plot_in_polar(data=rho,coords=coords,cmap="magma")
            print(f"\t{(i*len(g_span)+j+1)*100//(len(a_span)*len(g_span))}%")

            png_file_path:str = path.join(folder_path,"figs","densities_with_constant_mass",f"{metric_name}_{a}_g_{g_frac}.png")

            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()

    #"""

    #### Fixed mass accretion disk comparison with Kerr####

    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"
    diff_filename:str = "sum_diff.txt"
    diff2_filename:str = "sum_diff2.txt"
    Lum_filename:str = "luminosity_diff.txt"

    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]

    metric_name:str="Hay" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
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

    print("Reading %s" % Lum_filename)
    try:
        with open(Lum_filename, "rt") as fp:
            Lum_dict:dict = json.load(fp)
        print("Loaded data from %s" % Lum_filename)
    except:
        print("Could not read file, starting from scratch")
        Lum_dict:dict = {}


    metric_diff_dict:dict = {}
    metric_diff2_dict:dict = {}
    metric_Lum_dict:dict = {}
    for i,a in enumerate(a_span):
        dd_kerr.update_params(a=a) #updating the kerr metric spin 
        dd.update_params(a=a) #updating the metric spin

        rho_kerr,coords,r_max_kerr = dd_kerr.rho_fixed_mass(target_M=target_M, N=N_dd,gamma=gamma,measure=measure)
        L_kerr = dd_kerr.Luminosity(rho_kerr,coords,gamma=gamma)
    

        metric_diff_dict[a] = []
        metric_diff2_dict[a] = []
        metric_Lum_dict[a] = []

        for j,g_frac in enumerate(g_span):
            dd.update_params(g_max_frac=g_frac) #updating the metric deformation parameter

            rho,coords,r_max = dd.rho_fixed_mass(target_M=target_M, N=N_dd,gamma=gamma,measure=measure)
            
            L = dd.Luminosity(rho,coords,gamma=gamma)



            delta_rho = rho-rho_kerr #calculating the difference with the kerr metric 

            metric_diff_dict[a].append(np.sum(delta_rho))
            metric_diff2_dict[a].append(dd.disk_mass(rho=rho,coords=coords,measure=measure)-dd_kerr.disk_mass(rho=rho_kerr,coords=coords,measure=measure))
            metric_Lum_dict[a].append(L/L_kerr)
            


            dd.plot_in_polar(data=delta_rho,coords=coords,cmap="seismic",color_norm=False)
            print(f"\t{(i*len(g_span)+j + 1)*100//(len(a_span)*len(g_span))}%")
                
            png_file_path:str = path.join(folder_path,"figs","densities_with_const_mass_comp",f"{metric_name}_{a}_g_{g_frac}.png")

            plt.savefig(fname=png_file_path)
            # plt.show()
            plt.clf()
    
    diff_dict[metric_name] = metric_diff_dict
    diff2_dict[metric_name] = metric_diff2_dict
    Lum_dict[metric_name] = metric_Lum_dict

    print("Overwriting %s" % diff_filename)
    with open(diff_filename, "wt") as fp:
        json.dump(diff_dict, fp)
    print("Overwriting %s" % diff2_filename)
    with open(diff2_filename, "wt") as fp:
        json.dump(diff2_dict, fp)
    print("Overwriting %s" % Lum_filename)
    with open(Lum_filename, "wt") as fp:
        json.dump(Lum_dict, fp)
    print("Done with %s" % metric_name)
    
    
    #"""
            
    # type: ignore #### creating plots for diff per g ####

    """
    folder_path:str= "/Users/robin/Documents/Master thesis 1"
    diff_filename:str = "sum_diff.txt"
    diff2_filename:str = "sum_diff2.txt"
    Lum_filename:str = "luminosity_diff.txt"

    dpi = 300

    with open(diff_filename, "rt") as fp:
        diff_dict = json.load(fp)
    with open(diff2_filename, "rt") as fp:
        diff2_dict = json.load(fp)
    with open(Lum_filename, "rt") as fp:
        Lum_dict = json.load(fp)
    
    metric_names:list = ["Hay", "Bar", "EB", "GCSV", "KS", "RN"]

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
            plt.plot(g_span, diff2, label=f"a = {a}", color=color_line(i=i))
        
        png_file_path:str = path.join(folder_path,"figs","sum_diff2",f"{metric_name}.png")

        plt.legend()
        plt.xlabel(r"$\frac{|g|}{m}$")
        plt.ylabel(r"$\Delta M$") 
        # plt.show()
        plt.savefig(dpi = dpi, fname=png_file_path)

        plt.figure()
        for i,a in enumerate(a_span):
            Lum_frac = Lum_dict[metric_name][str(a)]
            plt.plot(g_span, Lum_frac, label=f"a = {a}", color=color_line(i=i))
        
        png_file_path:str = path.join(folder_path,"figs","Lum_diff",f"{metric_name}.png")

        plt.legend()
        plt.xlabel(r"$\frac{|g|}{m}$")
        plt.ylabel(r"$\frac{L(a)}{L_{Kerr}(a)}$")
        # plt.show()
        plt.savefig(dpi = dpi, fname=png_file_path)

    # """

    #####
    # """
    a_span:list = [0.5,0.6,0.7,0.8,0.9,1]
    g_span:list = [0.2,0.4,0.6,0.8,1]

    metric_name:str="Hay" #metric name from ["Kerr", "Hay","Bar", "EB", "GCSV", "KS", "Zha", "RN"]
    L_type:str = "const"
    N:int=1000
    gamma:float=5/3
    max_extension:float=25 #initial maximum extension of the accretion disk
    target_M = [30,25,20,15,10,5] #Target accretion disk mass
    measure:str = "rest"

    # dd_kerr = DenDist(metric_name="Kerr", g_max_frac=0,a=1,L_type=L_type) #initializing the density distribution object for the kerr metric
    dd=DenDist(metric_name=metric_name,g_max_frac=0,a=1,L_type=L_type) #initializing the density distribution object

    metric_comp_filename:str = "metric_comp_data.txt" 

    print("Reading %s" % metric_comp_filename)
    try:
        with open(metric_comp_filename, "rt") as fp:
            comp_dict:dict = json.load(fp)
        print("Loaded data from %s" % metric_comp_filename)
    except:
        print("Could not read file, starting from scratch")
        comp_dict:dict = {}
        comp_dict["a"] = a_span
        comp_dict["g"] = g_span
        comp_dict["M"] = target_M
    


    comp_dict[metric_name]={
        "r_cusp":[],
        "pot_diff":[],
        "rho_max":[] 
        }

    for i,a in enumerate(a_span):
        # dd_kerr.update_params(a=a)
        dd.update_params(a=a)

    
        for j,g_frac in enumerate(g_span):
            dd.update_params(g_max_frac = g_frac)

            # W_kerr,coords = dd_kerr.W_raw(N=N)
            W,coords = dd.W_raw(N=N)


            # r_cusp_kerr = coords[-1][np.argmax(W_kerr[-1])]
            r_cusp = coords[-1][np.argmax(W[-1])]
            print(np.argmax(W[-1]))
            comp_dict[metric_name]["r_cusp"].append(r_cusp)
            # print(np.shape((comp_dict[metric_name]["r_cusp"])))


            # potential_range_kerr = np.max(W_kerr[-1]) - np.min(W_kerr[-1][np.argmax(W_kerr[-1]):])
            potential_range = np.max(W[-1]) - np.min(W[-1][np.argmax(W[-1]):])
            comp_dict[metric_name]["pot_diff"].append(potential_range)
            # print(len(comp_dict[metric_name]["pot_diff"]))
            
            for k,M in enumerate(target_M):
                # rho_kerr,coords,_ = dd_kerr.rho_fixed_mass(target_M=M,N=N,gamma=gamma, measure="rest",W_coords=(W_kerr,coords))
                rho,coords,_ = dd.rho_fixed_mass(target_M=M,N=N,gamma=gamma, measure="rest",W_coords=(W,coords))
                rho_max = np.max(rho)
                

                comp_dict[metric_name]["rho_max"].append(rho_max)
                # print(len(comp_dict[metric_name]["rho_max"]))
            
            print(f"\t{(i*len(g_span)+j + 1)*100//(len(a_span)*len(g_span))}%")
    
    r_cusps = np.array(comp_dict[metric_name]["r_cusp"])
    np.reshape(r_cusps,(len(a_span),len(g_span)))
    comp_dict[metric_name]["r_cusp"] = r_cusps.tolist()

    pot_diff = np.array(comp_dict[metric_name]["pot_diff"])
    np.reshape(pot_diff,(len(a_span),len(g_span)))
    comp_dict[metric_name]["pot_diff"] = pot_diff.tolist()

    rho_max = np.array(comp_dict[metric_name]["rho_max"])
    print(rho_max)
    np.reshape(rho_max,(len(a_span)*len(g_span),len(target_M)))
    rho_max = rho_max.T
    np.reshape(rho_max,(len(target_M),len(a_span),len(g_span)))
    comp_dict[metric_name]["rho_max"] = rho_max.tolist()
    print(rho_max)

    print("Overwriting %s" % metric_comp_filename)
    with open(metric_comp_filename, "wt") as fp:
        json.dump(comp_dict, fp)
    print("Done with %s" % metric_name)
            



                

                
    #"""