#Paolo Tamagnini - 1536242
#paolotamag@gmail.com

import random
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

N = 6
c = 0.5

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


def y_pred(omega,v,b,x_val):
    som = 0
    for j in range(0,N):
        suma = 0
        for i in range(0,len(x_val)):
            suma += omega[j,i]*x_val[i]
        suma += b[j]
        som += v[j] * g(c,suma)
    return som

def error(var):
    omega = np.zeros((N,2))
    for i in range(0,N):
        omega[i,0] = var[i]
        omega[i,1] = var[i+N]
    v = var[2*N:3*N]
    b = var[3*N:4*N]
    P = len(x_train)
    somu = 0
    for i in range(0,P):
        somu += (y_pred(omega,v,b,x_train[i])-y_train[i])**2
    return somu/2

def error_test(var):
    omega = np.zeros((N,2))
    for i in range(0,N):
        omega[i,0] = var[i]
        omega[i,1] = var[i+N]
    v = var[2*N:3*N]
    b = var[3*N:4*N]
    P = len(x_train)
    somu = 0
    for i in range(0,n-P):
        somu += (y_pred(omega,v,b,x_test[i])-y_test[i])**2
    return somu/2


def g(c,t):
    return 1 / (1+ np.exp(-c*t))

def print_result(ress):
	print
	print "w1"
	print ress[:N]
	print
	print "w2"
	print ress[N:N*2]
	print
	print "v"
	print ress[N*2:N*3]
	print
	print "b"
	print ress[N*3:N*4]
	print



def print_res_info(res):
	print
	print "result:"
	r = res.x
	print_result(r)
	print "final error over train:"
	print error(res.x)
	print "divided by # of istances in train"
	print error(res.x)/P
	print "over test:"
	print error_test(res.x)
	print "divided by # of istances in test"
	print error_test(res.x)/(n-P)
	print

def yi_noNoise(x1,x2):
    n = len(x1)
    y = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            y[i,j] = f([x1[i,j],x2[i,j]]) 
    return y

def yi_pred_plot(x1,x2,var):
    omega = np.zeros((N,2))
    for i in range(0,N):
        omega[i,0] = var[i]
        omega[i,1] = var[i+N]
    v = var[2*N:3*N]
    b = var[3*N:4*N]
    n = len(x1)
    y = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            x_val = [x1[i,j],x2[i,j]]
            y[i,j] = y_pred(omega,v,b,x_val)
    return y


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
print
print "Initial errror:"
var = np.zeros(N*4)
for i in range(0,N*4):
    var[i] = random.random()
print error(var)
print

x1_scatter = []
x2_scatter = []
for a in x_train:
    x1_scatter.append(a[0])
    x2_scatter.append(a[1])
y_scatter = y_train

print("Scatter plots with f(x)..")

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


print("Plotting approximation function found..")
fig = plt.figure()
ax = fig.gca(projection='3d')

x1_plot = np.arange(0, 1, 0.01)
x2_plot = np.arange(0, 1, 0.01)
x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)

plugged_in_var = np.array([ -24.39359506,  15.40313868, -24.29766189,  -3.59030463,  -3.94143586,
  15.66531799, -5.73277184, -16.06174653,  -4.66968074,  15.54316305,  17.21038357,
 -13.95922685, -6.00324357,  11.22223938,   6.21618339,  -4.07314761,   4.13608812,
    -11.8798632, 14.65662368,  -9.18767862,  13.62281412,  -3.10855269,  -1.46299136,
    -10.12076475])

y_plot = yi_pred_plot(x1_plot,x2_plot,plugged_in_var)
ax.plot_wireframe(x1_plot, x2_plot, y_plot, color="green")
ax.plot_surface(x1_plot_old, x2_plot_old, y_plot_old, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(x1_scatter, x2_scatter, y_scatter, c='red')

print "Error train:"
print error(plugged_in_var)
print
print "Error test:"
print error_test(plugged_in_var)
print
print_result(plugged_in_var)

plt.show()


