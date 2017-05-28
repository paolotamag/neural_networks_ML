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

def y_pred(omega,x_val):
    som = 0
    for j in range(0,N):
        som += omega[j]*phi(c[j],x_val,sigma)

    return som

def error(var):
    omega = var[:N]
    P = len(x_train)
    somu = 0
    for i in range(0,P):
        somu += (y_pred(omega,x_train[i])-y_train[i])**2
    somu += ro*np.linalg.norm(omega)**2
    return somu/2
    
def error_test(var):
    omega = var[:N]
    P = len(x_train)
    somu = 0
    for i in range(0,n-P):
        somu += (y_pred(omega,x_test[i])-y_test[i])**2
    somu += ro*np.linalg.norm(omega)**2
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

    n = len(x1)
    y = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            x_val = [x1[i,j],x2[i,j]]
            y[i,j] = y_pred(omega,x_val)
    return y

def print_result(ress):
    print
    print "w: "
    print ress[:N]
    print



def print_res_info(res):

    errorTR = error(res.x)

    errorTE = error_test(res.x)


    return errorTR,errorTE



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

ro = 0.0001


N_list = range(1,21)
sigma_list = [0.5,1]
maxN = np.max(N_list)

omega_var = np.zeros(maxN)
for i in range(0,maxN):
    omega_var[i] = random.random()

print "Drawing at random ", maxN, " c from x_train.."
c_index = np.random.choice(range(0,P), maxN, replace=False)
print "chosen xs indexes : "
print
print c_index
print
c_var_fixed = x_train[c_index,:]
print "chosen centers : "
print
print c_var_fixed
print



best_error_Train = 1000
best_error_Test = 1000
best_for_Train = []
best_for_Test = []
for N in N_list:
    for sigma in sigma_list:
        print "----------(N:",N,"| sigma:",sigma,")----------"

        var = omega_var[:N]
        c = c_var_fixed[:N,:]
        resBFGS = minimize(error, var, method='BFGS')
        errTrain,errTest = print_res_info(resBFGS)
        if (best_error_Train > errTrain):
            print
            print "------------------------"
            print " found better for train!"
            print errTrain,"(",N,",",sigma,")"
            print "------------------------"
            best_error_Train = errTrain
            best_for_Train = [N,sigma,resBFGS,errTest]
            print
        if (best_error_Test > errTest):
            print
            print "------------------------"
            print "  found better for test!"
            print errTest,"(",N,",",sigma,")"
            print "use xs: ",c_index
            print "------------------------"
            best_error_Test = errTest
            best_for_Test = [N,sigma,resBFGS,errTrain]
            print  
        print

N_best_train = best_for_Train[0]
sigma_best_train = best_for_Train[1]
var_best_train = best_for_Train[2].x
print "the best N and sigma for train are:"
print "N: ",N_best_train
print "c: ",sigma_best_train
print
print "with train error:", best_error_Train
print "with test error:", best_for_Train[3]
print
omega = var_best_train[:N_best_train]
print "w: ",omega
print
print "c for best train:"
print c_var_fixed[:N_best_train]
print c_index[:N_best_train]
print
print "--------------------------------------"
print
print "the best N and sigma for test are:"
N_best_test = best_for_Test[0]
c_best_test = best_for_Test[1]
var_best_test = best_for_Test[2].x
print "N: ",N_best_test
print "c: ",c_best_test
print
print "with train error:", best_for_Test[3]
print "with test error:", best_error_Test
print
omega = var_best_test[:N_best_test]
print "w: ",omega
print
print "c for best test:"
print c_var_fixed[:N_best_test]
print c_index[:N_best_test]
