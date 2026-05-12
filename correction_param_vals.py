import numpy as np
import json
from matplotlib import pyplot as plt


def g_rr_inv_hay(r,a,g)->float:
    g_rr = (r**2+a**2)*(r**3+g**3)-2*r**4
    return g_rr

def g_rr_inv_bar(r,a,g)->float:
    g_rr = (r**2+a**2)*(r**2+g**2)**(1.5)-2*r**4
    return g_rr

def g_rr_inv_gcsv(r,a,j)->float:
    g_rr=r**2 + a**2-2*r*np.exp(-j/r)
    return g_rr

def g_rr_inv_eb(r,a,j)->float:
    g_rr=r**2 + a**2-2*r*(1-np.tanh(j/r))
    return g_rr

def g_rr_inv_zha(r,a,g)->float:
    g_rr = r**2+a**2 - 2*r*(1-g**2/(2*r)*(1-4/r+4/r**2))
    return g_rr
    pass

# def g_hay(r_root,a):
#     print(4*r_root**2-5/2*r_root**3-3/2*a*r_root)
#     return np.power(4*r_root**2-5/2*r_root**3-3/2*a*r_root,1/3)

# def g_bar(r_root,a):
#     pass


r=np.linspace(1.0,1.5,10000)

a=np.linspace(0,1,1000)



g_hay=np.linspace(0,1.1,1000)
g_bar=np.linspace(0,0.8,1000)
j_gcsv=np.linspace(0,0.8,1000)
j_eb=np.linspace(0,0.6,1000)
g_bar=np.linspace(0,0.8,1000)

list_of_g_hay=[]
for i,aval in enumerate(a):
    if i%10==0:
        print(f"{i//10}%")
    # coeffs_hay = [-1.5,2,3*aval**2, 4*aval**2,-1.5*aval**4]
    # roots_hay = np.polynomial.polynomial.polyroots(coeffs_hay[::-1])
    # print(roots_hay)
    # r_root = np.real(roots_hay[0])
    # print(g_hay(r_root=r_root,a=aval))
    
    
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

list_of_g_gcsv=[] 
for i,aval in enumerate(a):
    if i%10==0:
        print(f"{i//10}%")
    for jval in j_gcsv[::-1]:
        if np.min(g_rr_inv_bar(r,aval,jval))<=0:
            list_of_g_gcsv.append(np.sqrt(2*jval))
            # print(f"{len(list_of_g_bar)//10}%")
            break

list_of_g_eb=[] 
for i,aval in enumerate(a):
    if i%10==0:
        print(f"{i//10}%")
    for jval in j_eb[::-1]:
        if np.min(g_rr_inv_eb(r,aval,jval))<=0:
            list_of_g_eb.append(np.sqrt(2*jval))
            # print(f"{len(list_of_g_bar)//10}%")
            break

# list_of_g_zha=[] 
# for i,aval in enumerate(a):
#     if i%10==0:
#         print(f"{i//10}%")
#     for gval in g_zha[::-1]:
#         if np.min(g_rr_inv_zha(r,aval,gval))<=0:
#             list_of_g_zha.append(np.sqrt(2*gval))
#             # print(f"{len(list_of_g_bar)//10}%")
#             break

list_of_a = a.tolist()
dictionary = {
    "a_vals": list_of_a,
    "g_hay": list_of_g_hay,
    "g_bar": list_of_g_bar,
    "g_gcsv": list_of_g_gcsv,
    "g_eb": list_of_g_eb
}

with open("param_vals.txt", "w") as file:
    json.dump(dictionary,file)
    file.close

plt.figure()
plt.plot(a,list_of_g_gcsv,c="b", label="GCSV BH")
# plt.fill_between(a,list_of_g_gcsv, alpha=1,color="b")
plt.plot(a,list_of_g_hay,c="r", label="Hayward BH")
# plt.fill_between(a,list_of_g_hay,alpha=1,color="r")
plt.plot(a,list_of_g_eb,c="y", label="EB BH")
# plt.fill_between(a,list_of_g_eb, alpha=1,color="y")
plt.plot(a,np.sqrt(1-a**2),c="pink", label="RN BH")
# plt.fill_between(a,list_of_g_bar, alpha=1,color="g")
plt.plot(a,list_of_g_bar,c="g", label="Bardeen BH")
# plt.fill_between(a,list_of_g_bar, alpha=1,color="g")
plt.legend()
plt.ylabel(r"g")
plt.xlabel(r"a")
plt.savefig("allowed_g_vals.pdf")
plt.show() 



