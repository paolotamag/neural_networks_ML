#Paolo Tamagnini - 1536242
#paolotamag@gmail.com

import random
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import time

def f(x):
    summa = 0
    summa += +0.75*np.exp(-(9*x[0]-2)**2/4-(9*x[1]-2)**2/4)
    summa += +0.75*np.exp(-(9*x[0]+1)**2/49-(9*x[1]+1)**1/10) 
    summa += +0.75*np.exp(-(9*x[0]-7)**2/4-(9*x[1]-3)**2/4) 
    summa += -0.2*np.exp(-(9*x[0]-4)**2/1-(9*x[1]-7)**2/1)
    return summa

def yi(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(0,n):
        y[i] = f(x[i]) + random.uniform(0,10**(-2))
    return y

def sample_x(n,seed):
    random.seed(seed)
    x = np.zeros((n,2))
    for i in range(0,n):
        x[i,:] = [random.random(),random.random()]
    return x



def phi(c,x,sigma):
    return np.exp(-(np.linalg.norm(x-c)/sigma)**2)

def y_pred(omega,c,x_val):
    som = 0
    for j in range(0,N):
        som += omega[j]*phi(c[j],x_val,sigma)

    return som

def error(var):
    omega = var[:N]
    c = np.zeros((N,2))
    for i in range(0,N):
        c[i,0] = var[i+N]
        c[i,1] = var[i+2*N]
    P = len(x_train)
    somu = 0
    for i in range(0,P):
        somu += (y_pred(omega,c,x_train[i])-y_train[i])**2
    somu += ro1*np.linalg.norm(omega)**2
    somu += ro2*np.linalg.norm(c)**2
    return somu/2
    
def error_test(var):
    omega = var[:N]
    c = np.zeros((N,2))
    for i in range(0,N):
        c[i,0] = var[i+N]
        c[i,1] = var[i+2*N]
    P = len(x_train)
    somu = 0
    for i in range(0,n-P):
        somu += (y_pred(omega,c,x_test[i])-y_test[i])**2
    somu += ro1*np.linalg.norm(omega)**2
    somu += ro2*np.linalg.norm(c)**2
    return somu/2

def yi_noNoise(x1,x2):
    n = len(x1)
    y = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            y[i,j] = f([x1[i,j],x2[i,j]]) 
    return y

def yi_pred_plot(x1,x2,var):
    omega = var[:N]
    c = np.zeros((N,2))
    for i in range(0,N):
        c[i,0] = var[i+N]
        c[i,1] = var[i+2*N]

    n = len(x1)
    y = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            x_val = [x1[i,j],x2[i,j]]
            y[i,j] = y_pred(omega,c,x_val)
    return y

def print_result(ress):
    print
    print "w"
    print ress[:N]
    print
    print "c"
    print ress[N:]
    print


def print_res_info(res):
    print
    print "result:"
    r = res.x
    print_result(r)
    print "final error over train:"
    print error(res.x)
    print "over test:"
    print error_test(res.x)
    print


N = 7
sigma = 0.5

n = 100
perc_train = 0.75
P = int(perc_train * n)
seed = 1536242
print
print "Generating data-set.."
print "Using seed: ",seed
x = sample_x(100,seed)
y = yi(x)

y_train = y[0:P]
x_train = x[0:P]
x_test = x[P:]
y_test = y[P:]


ro1 = 0.0001
ro2 = 0.0001


x1_scatter = []
x2_scatter = []
for a in x_train:
    x1_scatter.append(a[0])
    x2_scatter.append(a[1])
y_scatter = y_train

print "Scatter plots with f(x).."

fig = plt.figure()
ax = fig.gca(projection='3d')
x1_plot_old = np.arange(0, 1, 0.01)
x2_plot_old = np.arange(0, 1, 0.01)
x1_plot_old, x2_plot_old = np.meshgrid(x1_plot_old, x2_plot_old)
y_plot_old = yi_noNoise(x1_plot_old,x2_plot_old)
ax.plot_surface(x1_plot_old, x2_plot_old, y_plot_old, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.scatter(x1_scatter, x2_scatter, y_scatter, c='red')
plt.show()
print
print("Plotting approximation function found..")
fig = plt.figure()
ax = fig.gca(projection='3d')

x1_plot = np.arange(0, 1, 0.01)
x2_plot = np.arange(0, 1, 0.01)
x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)
plugged_in_var = np.array([-3.30135769, -2.47045723, -5.53939177, -5.1731324,   4.89775938,  6.11583218,
                            1.27749701, 1.2204531,  0.3626939, -0.37017004,  0.48979913,  0.81539142,  0.12358006,  0.26733293,  
                            0.13225388,  0.44698984, 0.00739228, -0.08666898,  0.11820637, 0.04019551, 0.65817032,])
y_plot = yi_pred_plot(x1_plot,x2_plot,plugged_in_var)
ax.plot_wireframe(x1_plot, x2_plot, y_plot, color="green")
ax.plot_surface(x1_plot_old, x2_plot_old, y_plot_old, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(x1_scatter, x2_scatter, y_scatter, c='red')

plt.show()
