#Paolo Tamagnini - 1536242
#paolotamag@gmail.com

import random
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


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


    errorTR = error(res.x)

    errorTE = error_test(res.x)


    return errorTR,errorTE

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

N_list = range(1,11)
c_list = [0.1,0.5,1,2,5,10]

maxN = np.max(N_list)
omega_var = np.zeros((maxN,2))
for i in range(0,maxN):
    for j in range(0,2):
        omega_var[i,j] = random.random()
b_var = np.zeros(maxN)
for i in range(0,maxN):
    b_var[i] = random.random()
v_var = np.zeros(maxN)
for i in range(0,maxN):
    v_var[i] = random.random()

best_error_Train = 1000
best_error_Test = 1000
best_for_Train = []
best_for_Test = []
for N in N_list:
    for c in c_list:
        print "----------(N:",N,"| c:",c,")----------"
        omega_var_now = omega_var[:N,:]
        b_var_now = b_var[:N]
        v_var_now = v_var[:N]
        concat1 = np.append(omega_var_now[:,0],omega_var_now[:,1])
        concat2 =np.append(b_var_now,v_var_now)
        var = np.append(concat1,concat2)
        resBFGS = minimize(error, var, method='BFGS')
        errTrain,errTest = print_res_info(resBFGS)
        if (best_error_Train > errTrain):
            print
            print "------------------------"
            print " found better for train!"
            print errTrain,"(",N,",",c,")"
            print "------------------------"
            best_error_Train = errTrain
            best_for_Train = [N,c,resBFGS,errTest]
            print
        if (best_error_Test > errTest):
            print
            print "------------------------"
            print "  found better for test!"
            print errTest,"(",N,",",c,")"
            print "------------------------"
            best_error_Test = errTest
            best_for_Test = [N,c,resBFGS,errTrain]
            print  
        print

print "the best N and c for train are:"
N_best_train = best_for_Train[0]
c_best_train = best_for_Train[1]
var_best_train = best_for_Train[2].x
print "N: ",N_best_train
print "c: ",c_best_train
print "with train error:", best_error_Train
print "with test error:", best_for_Train[3]
omega = np.zeros((N_best_train,2))
for i in range(0,N_best_train):
    omega[i,0] = var_best_train[i]
    omega[i,1] = var_best_train[i+N_best_train]
v = var_best_train[2*N_best_train:3*N_best_train]
b = var_best_train[3*N_best_train:4*N_best_train]
print "v: ",v
print "w: ",omega
print "b: ",b
print
print "--------------------------------------"
print
print "the best N and c for test are:"
N_best_test = best_for_Test[0]
c_best_test = best_for_Test[1]
var_best_test = best_for_Test[2].x
print "N: ",N_best_test
print "c: ",c_best_test
print "with train error:", best_for_Test[3]
print "with test error:", best_error_Test
omega = np.zeros((N_best_test,2))
for i in range(0,N_best_test):
    omega[i,0] = var_best_test[i]
    omega[i,1] = var_best_test[i+N_best_test]
v = var_best_test[2*N_best_test:3*N_best_test]
b = var_best_test[3*N_best_test:4*N_best_test]
print "v: ",v
print "w: ",omega
print "b: ",b
