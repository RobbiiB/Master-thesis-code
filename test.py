import numpy as np
from matplotlib import pyplot as plt

a = np.array([1,
              0.99,
              0.975,
              0.95,
              0.9,
              0.85,
              0.8,
              0.75,
              0.7,
              0.65,
              0.6,
              0.55,
              0.5,
              0.45,
              0.4,
              0.35,
              0.3,
              0.25,
              0.2,
              0.15,
              0.1,
              0.05,
              0])
# print(a)

g = np.array([0.0,
            0.2171708,
            0.297963,
            0.38137,
            0.492832,
            0.5753213,
            0.64210189,
            0.701299544,
            0.7524116,
            0.797856584,
            0.83851723,
            0.87497525,
            0.90762789,
            0.936751486,
            0.9625397,
            0.985126962,
            1.004604736,
            1.02103292,
            1.0344484794,
            1.044872258,
            1.0523144891,
            1.0567791775,
            2/3*2**(2/3)])


def g_func(a, *args):
    g=0
    for i,coeff in enumerate(args):
        g+=coeff*(1-a**(2*(i+1)))**(1/(2*(i+1)))
    return 2/3*2**(2/3)*g

def dloss_dwj(a,j,*args):
    val= -2*np.sum(g-g_func(a,*args)*(1-a**(2*j))**(1/2*j))*2/3*2**(2/3)
    return val

def grad_descent(alpha,a,*coeffs):
    new_coeffs=[]
    for j in range(len(coeffs)):
        new_wj = alpha/dloss_dwj(a,j,*coeffs) + coeffs[j]
        new_coeffs.append(new_wj)
    return new_coeffs
        


coeffs=np.array([0.4,0.2,0.2,0.2])

itterations=100
alpha=0.001
for itteration in range(itterations):
    coeffs = grad_descent(alpha,a,*coeffs)
    coeffs=coeffs/np.sum(coeffs)
print(coeffs)
print(np.sum(coeffs))
plt.figure()
plt.plot(a,g)
plt.plot(a,g_func(a,*coeffs),c="k")
plt.show()