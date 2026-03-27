import numpy as np
from matplotlib import pyplot as plt


def g_rr_inv_hay(r,a,g)->float:
    g_rr = (r**2+a**2)*(r**3+g**3)-2*r**4
    return g_rr

def g_rr_inv_bar(r,a,g)->float:
    g_rr = (r**2+a**2)*(r**2+g**2)**(1.5)-2*r**4
    return g_rr

r=np.linspace(1.0,1.5,10000)

a=np.linspace(0,1,1000)
g_hay=np.linspace(0,1.1,1000)
g_bar=np.linspace(0,0.8,1000)

list_of_g_hay=[]
for i,aval in enumerate(a):
    if i%10==0:
        print(f"{i//10}%")
    for gval in g_hay[::-1]:
        if np.min(g_rr_inv_hay(r,aval,gval))<=0:
            list_of_g_hay.append(gval)
            # print(f"{len(list_of_g_hay)//10}%")
            break

list_of_g_bar=[] 
for i,aval in enumerate(a):
    if i%10==0:
        print(f"{i//10}%")
    for gval in g_bar[::-1]:
        if np.min(g_rr_inv_bar(r,aval,gval))<=0:
            list_of_g_bar.append(gval)
            # print(f"{len(list_of_g_bar)//10}%")
            break
        

plt.figure()
plt.plot(a,list_of_g_hay,c="r", label="Hayward BH")
plt.fill_between(a,list_of_g_hay,alpha=0.5,color="r")
plt.plot(a,list_of_g_bar,c="b", label="Bardeen BH")
plt.fill_between(a,list_of_g_bar, alpha=0.5,color="b")
plt.legend()
plt.ylabel(r"g")
plt.xlabel(r"a")
plt.savefig("allowed_g_vals.pdf")
plt.show() 



