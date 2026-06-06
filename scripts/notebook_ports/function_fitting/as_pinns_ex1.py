"""Python port generated from the original AS-PINNs notebook.

Source notebook: notebooks/function_fitting/as_pinns_ex1.ipynb

This file keeps the original executable cell order for traceability. Reusable, tested project code lives under src/as_pinns/.
"""


# %% [cell 1]
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sympy import integrate, exp, sin, log, oo, pi,symbols
import deepxde as dde
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from deepxde.backend import tf
import datetime
import os
from pylab import mpl
from scipy.signal import chirp, spectrogram
mpl.rcParams['font.sans-serif']=['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus']=False
from matplotlib.pyplot import MultipleLocator
import xlrd
import xlwt
from sympy import *
import sympy as sp

import runpy
from pathlib import Path
globals().update(runpy.run_path(str(Path(__file__).with_name('solution_ex1.py'))))

def Plot_dis():
    import re
    lines = open("s1-s2.dat", "r").readlines()
    S = np.array(
        [
            np.fromstring(
                min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                sep=",",
            )
            for line in lines
        ]
    )
    markersize = 10
    markevery = 50
    plt.figure()
    l = S.shape[0]
    width, height, dpi = 4, 2.7, 150
    plt.figure(figsize=(width,height),dpi = dpi)
    markersize = 2
    markevery = 20
    plt.plot(range(0, 200 * l, 200), S[:,0:1], linestyle='dashed',linewidth=0.7, \
             marker='^',markersize=markersize,markevery=markevery, label="s1",color='orange')
    plt.plot(range(0, 200 * l, 200), S[:,1:2], linestyle='dashed',linewidth=0.7, \
             marker='+',markersize=markersize,markevery=markevery, label="s2",color='#884B8B')
    plt.plot(range(0, 200 * l, 200), S[:,2:3], linestyle='dashed',linewidth=0.7, \
             marker='1',markersize=markersize,markevery=markevery, label="s3",color='#004B8B')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.4,linewidth=0.7, label="exact",color='green')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.9,linewidth=0.7, label="exact",color='lightcoral')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+2,linewidth=0.7, label="exact",color='#900000')
    plt.tick_params(width=0.5, labelsize=6)
    plt.xlabel('epoch', fontsize=8)
    plt.legend(fontsize=6,frameon=False,bbox_to_anchor=(0.92, 0.7))
    plt.show()
    data_s = np.hstack((np.array([range(0, 200 * l, 200)]).reshape(-1,1),S))
    print(S[-1])

# %% [cell 2]
np.random.seed(1234)
tf.set_random_seed(1234)

s1 = tf.math.abs(tf.Variable(0.1, trainable=True, dtype=tf.float32))
s2 = tf.math.abs(tf.Variable(0.2, trainable=True, dtype=tf.float32))
s3 = tf.math.abs(tf.Variable(0.3, trainable=True, dtype=tf.float32))

c1 = (tf.Variable(0, trainable=True, dtype=tf.float32))
c2 = (tf.Variable(0, trainable=True, dtype=tf.float32))
c3 = (tf.Variable(0, trainable=True, dtype=tf.float32))

num_domain = 100
epochs=20000
lr = 1e-3
u_nodenum=20
u_layer=2

activation_func1 = tf.nn.tanh
L = 3
def f1(x_in):
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0), tf.less(x_in, 0.4))
    cond2 = tf.logical_and(tf.greater_equal(x_in, 0.4), tf.less(x_in, 0.9))
    cond3 = tf.logical_and(tf.greater_equal(x_in, 0.9), tf.less(x_in, 2))
    f1 = x_in*0+4
    f2 = x_in*-2+4.8
    f3 = 0*x_in+3
    f4 = 0*x_in+2
    f = tf.where(cond1, f1, tf.where(cond2, f2, tf.where(cond3, f3, f4)))
    return f

def f2(x_in):
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0.4), tf.less(x_in, 0.9))
    f1 = x_in*0-2
    f2 = x_in*0
    f = tf.where(cond1, f1, f2)
    return f

def beampde(x, y):
    x_in = x[:,0:1]
    fai = y[:,0:4]
    M = y[:,4:8]

    M0_dNN = dde.grad.jacobian(fai, x, i =0, j=0)
    M1_dNN = dde.grad.jacobian(fai, x, i =1, j=0)
    M2_dNN = dde.grad.jacobian(fai, x, i =2, j=0)
    M3_dNN = dde.grad.jacobian(fai, x, i =3, j=0)

    V0_dNN = dde.grad.jacobian(M, x, i =0, j=0)
    V1_dNN = dde.grad.jacobian(M, x, i =1, j=0)
    V2_dNN = dde.grad.jacobian(M, x, i =2, j=0)
    V3_dNN = dde.grad.jacobian(M, x, i =3, j=0)

    f1_sig = 10*tf.keras.activations.relu(x_in+s1)*tf.keras.activations.relu(-x_in+s1)
    f2_sig = 10*tf.keras.activations.relu(x_in-s1)*tf.keras.activations.relu(-x_in+s2)
    f3_sig = 10*tf.keras.activations.relu(x_in-s2)*tf.keras.activations.relu(-x_in+s3)
    f4_sig = 10*tf.keras.activations.relu(x_in-s3)*tf.keras.activations.relu(-x_in+L+(L-s3))

    return [(M[:,0:1]-M0_dNN)*f1_sig+(M[:,1:2]-M1_dNN)*f2_sig+(M[:,2:3]-M2_dNN)*f3_sig+(M[:,3:4]-M3_dNN)*f4_sig,\
            (M[:,0:1]-f1(x_in))*f1_sig+(M[:,1:2]-f1(x_in))*f2_sig+(M[:,2:3]-f1(x_in))*f3_sig+(M[:,3:4]-f1(x_in))*f4_sig,\
           ((V0_dNN-f2(x_in))*f1_sig+(V1_dNN-f2(x_in))*f2_sig+(V2_dNN-f2(x_in))*f3_sig+(V3_dNN-f2(x_in))*f4_sig)]

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)
geom = dde.geometry.Interval(0, L)

data = dde.data.TimePDE(geom,beampde,[], num_domain=num_domain)
net = dde.maps.FNN([1]  + [50]*3 + [8], "tanh", "Glorot uniform")

def modify_output(X, y):
    x = X[:,0:1]
    fai = y[:,0:4]
    M = y[:,4:8]
    final_output = tf.concat([fai[:,0:1]*x*(x-s1)+10*c1*x/s1,
                              fai[:,1:2]*(x-s1)*(x-s2)+10*c1*(x-s2)/(s1-s2)+10*c2*(x-s1)/(s2-s1),
                              fai[:,2:3]*(x-s2)*(x-s3)+10*c2*(x-s3)/(s2-s3)+10*c3*(x-s2)/(s3-s2),
                              fai[:,3:4]*(x-s3)+10*c3,
                              M], axis=1)
    return final_output

net.apply_output_transform(modify_output)
model = dde.Model(data, net)
model.compile("adam", lr,loss_weights=[1,1,1])
variable = dde.callbacks.VariableValue([s1,s2,s3,c1,c2,c3], period=200, filename="s1-s2.dat")#200为epochs之间的间隔
losshistory, train_state = model.train(epochs=epochs, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
Plot_dis()
list_point = []
list_reslut = []

# %% [cell 3]
for i in range(7):
    import re
    lines = open("s1-s2.dat", "r").readlines()
    S = np.array(
        [
            np.fromstring(
                min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                sep=",",
            )
            for line in lines
        ]
    )
    print(S[-1])
    interv1 = (X1>=0)&(X1<S[:,0:1][-1][0])
    interv2 = (X1>=S[:,0:1][-1][0])&(X1<S[:,1:2][-1][0])
    interv3 = (X1>=S[:,1:2][-1][0])&(X1<S[:,2:3][-1][0])
    interv4 = (X1>=S[:,2:3][-1][0])&(X1<=3)
    x0 = (X1[interv1]).reshape(-1,1)
    x1 = (X1[interv2]).reshape(-1,1)
    x2 = (X1[interv3]).reshape(-1,1)
    x3 = (X1[interv4]).reshape(-1,1)
    X2 = X1
    plt.rcParams.update({'font.size': 10.5})
    markersize=5
    markevery=300
    plt.figure(figsize=(16,4),dpi=150)
    plt.subplot(1,3,1)
    fai1 = model.predict(x0)[:,0:1]
    fai2 = model.predict(x1)[:,1:2]
    fai3 = model.predict(x2)[:,2:3]
    fai4 = model.predict(x3)[:,3:4]

    M1_dNN = model.predict(x0,operator=lambda x,y:dde.grad.jacobian(y,x,i=0,j=0))
    M2_dNN = model.predict(x1,operator=lambda x,y:dde.grad.jacobian(y,x,i=1,j=0))
    M3_dNN = model.predict(x2,operator=lambda x,y:dde.grad.jacobian(y,x,i=2,j=0))
    M4_dNN = model.predict(x3,operator=lambda x,y:dde.grad.jacobian(y,x,i=3,j=0))
    M_dNN = np.vstack((M1_dNN,M2_dNN,M3_dNN,M4_dNN))
    fai = np.vstack((fai1,fai2,fai3,fai4))
    plt.plot(X1,fai, label='PINN',color='green',marker='o',markersize=markersize,markevery=markevery,linewidth=0.7)
    plt.ylabel('u')
    plt.legend()

    plt.subplot(1,3,2)
    M1 = model.predict(x0)[:,4:5]
    M2 = model.predict(x1)[:,5:6]
    M3 = model.predict(x2)[:,6:7]
    M4 = model.predict(x3)[:,7:8]
    M = np.vstack((M1,M2,M3,M4))
    plt.plot(x0,M1, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
    plt.plot(x1,M2, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
    plt.plot(x2,M3, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
    plt.plot(x3,M4, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
    plt.ylabel('du')
    plt.legend()
    plt.subplot(1,3,3)
    V1 = model.predict(x0,operator=lambda x,y:dde.grad.jacobian(y,x,i=4,j=0))
    V2 = model.predict(x1,operator=lambda x,y:dde.grad.jacobian(y,x,i=5,j=0))
    V3 = model.predict(x2,operator=lambda x,y:dde.grad.jacobian(y,x,i=6,j=0))
    V4 = model.predict(x3,operator=lambda x,y:dde.grad.jacobian(y,x,i=7,j=0))
    V  = np.vstack((V1,V2,V3,V4))
    plt.plot(x0, V1, label='PINN',color='green',marker='^',markersize=markersize,\
             markevery=markevery,linewidth=0.7)
    plt.plot(x1, V2, label='PINN',color='green',marker='^',markersize=markersize,\
             markevery=markevery,linewidth=0.7)
    plt.plot(x2, V3, label='PINN',color='green',marker='^',markersize=markersize,\
             markevery=markevery,linewidth=0.7)
    plt.plot(x3, V4, label='PINN',color='green',marker='^',markersize=markersize,\
             markevery=markevery,linewidth=0.7)
    plt.plot(X1,ddWW1)
    plt.ylabel('ddu')
    plt.legend()
    ####真解
    dis = 0.75
    interv1 = (X1>=0)&(X1<0.4)
    interv2 = (X1>=0.4)&(X1<0.9)
    interv3 = (X1>=0.9)&(X1<2)
    interv4 = (X1>=2)&(X1<=3)
    x_vals_w0 = X1[interv1].reshape(-1,1)
    x_vals_w1 = X1[interv2].reshape(-1,1)
    x_vals_w2 = X1[interv3].reshape(-1,1)
    x_vals_w3 = X1[interv4].reshape(-1,1)
    # plt.figure(figsize=(18,7),dpi=150)
    plt.subplot(1, 3, 1)
    dW1 = WW1[interv1].reshape(-1,1)
    dW2 = WW1[interv2].reshape(-1,1)
    dW3 = WW1[interv3].reshape(-1,1)
    dW4 = WW1[interv4].reshape(-1,1)
    plt.plot(x_vals_w0, dW1, label="w1'", linestyle='dashed', color='purple')
    plt.plot(x_vals_w1, dW2, label="w2'", linestyle='dashed', color='blue')
    plt.plot(x_vals_w2, dW3, label="w3'", linestyle='dashed', color='orange')
    plt.plot(x_vals_w3, dW4, label='w4',linestyle='dashed', color='red')
    # plt.title('First Derivative')
    plt.legend()

    plt.subplot(1, 3, 2)
    ddW1 = dWW1[interv1].reshape(-1,1)
    ddW2 = dWW1[interv2].reshape(-1,1)
    ddW3 = dWW1[interv3].reshape(-1,1)
    ddW4 = dWW1[interv4].reshape(-1,1)
    plt.plot(x_vals_w0, ddW1, label="w1'", linestyle='dashed', color='purple')
    plt.plot(x_vals_w1, ddW2, label="w2'", linestyle='dashed', color='blue')
    plt.plot(x_vals_w2, ddW3, label="w3'", linestyle='dashed', color='orange')
    plt.plot(x_vals_w3, ddW4, label='w4',linestyle='dashed', color='red')
    # plt.title('First Derivative')
    plt.legend()
    plt.savefig('function fitting.png')

    norm_Fai = np.linalg.norm(fai-WW1,ord=2) / np.linalg.norm(WW1,ord=2)

    norm_M2 = np.linalg.norm(M-M_dNN,ord=2) / np.linalg.norm(M_dNN,ord=2)

    norm_M = np.linalg.norm(M-dWW1,ord=2) / np.linalg.norm(dWW1,ord=2)
    norm_V = np.linalg.norm(V-ddWW1,ord=2) / np.linalg.norm(ddWW1,ord=2)
    norm_Reslut = np.array([norm_Fai,norm_M,norm_V,norm_M2])
    norm_Reslut2 = np.array([norm_M2,norm_M,norm_V])
    print('norm_Fai,norm_M,norm_V',norm_Reslut)
    Weight = norm_Reslut2/np.min(norm_Reslut2)

    print('Weight',Weight)
    list_reslut.append(norm_Reslut)
    model.compile("adam", 1e-4,loss_weights=Weight)
    X = np.random.uniform(0,L,10000).reshape(-1,1)
    err_total = np.abs(model.predict(X, operator=beampde))  #计算前6项残差
    err_eq = np.sum(err_total,axis=0)
    x_ids = np.argsort(-err_eq,axis=0 )[:5]
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        list_point.append(X[elem][0][0])
        data.add_anchors(X[elem])
    losshistory, train_state = model.train(epochs=epochs, callbacks=[variable])
    # losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    Plot_dis()
import re
lines = open("s1-s2.dat", "r").readlines()
S = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
print(S[-1])
interv1 = (X1>=0)&(X1<S[:,0:1][-1][0])
interv2 = (X1>=S[:,0:1][-1][0])&(X1<S[:,1:2][-1][0])
interv3 = (X1>=S[:,1:2][-1][0])&(X1<S[:,2:3][-1][0])
interv4 = (X1>=S[:,2:3][-1][0])&(X1<=3)
x0 = (X1[interv1]).reshape(-1,1)
x1 = (X1[interv2]).reshape(-1,1)
x2 = (X1[interv3]).reshape(-1,1)
x3 = (X1[interv4]).reshape(-1,1)
X2 = X1
plt.rcParams.update({'font.size': 10.5})
markersize=5
markevery=300
plt.figure(figsize=(16,4),dpi=150)
plt.subplot(1,3,1)
fai1 = model.predict(x0)[:,0:1]
fai2 = model.predict(x1)[:,1:2]
fai3 = model.predict(x2)[:,2:3]
fai4 = model.predict(x3)[:,3:4]

M1_dNN = model.predict(x0,operator=lambda x,y:dde.grad.jacobian(y,x,i=0,j=0))
M2_dNN = model.predict(x1,operator=lambda x,y:dde.grad.jacobian(y,x,i=1,j=0))
M3_dNN = model.predict(x2,operator=lambda x,y:dde.grad.jacobian(y,x,i=2,j=0))
M4_dNN = model.predict(x3,operator=lambda x,y:dde.grad.jacobian(y,x,i=3,j=0))
M_dNN = np.vstack((M1_dNN,M2_dNN,M3_dNN,M4_dNN))

fai = np.vstack((fai1,fai2,fai3,fai4))
plt.plot(X1,fai, label='PINN',color='green',marker='o',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('u')
plt.legend()

plt.subplot(1,3,2)
M1 = model.predict(x0)[:,4:5]
M2 = model.predict(x1)[:,5:6]
M3 = model.predict(x2)[:,6:7]
M4 = model.predict(x3)[:,7:8]
M = np.vstack((M1,M2,M3,M4))
plt.plot(x0,M1, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.plot(x1,M2, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.plot(x2,M3, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.plot(x3,M4, label='PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
# plt.plot(x,model.predict(x,operator=lambda x,y:dde.grad.jacobian(y,x,i=1,j=0)), label='dy_xx',marker='.',markersize=markersize,markevery=40,linewidth=0.7)
# plt.plot(x,x*0-20,'-.',linewidth=0.7)
plt.ylabel('du')
plt.legend()
plt.subplot(1,3,3)
V1 = model.predict(x0,operator=lambda x,y:dde.grad.jacobian(y,x,i=4,j=0))
V2 = model.predict(x1,operator=lambda x,y:dde.grad.jacobian(y,x,i=5,j=0))
V3 = model.predict(x2,operator=lambda x,y:dde.grad.jacobian(y,x,i=6,j=0))
V4 = model.predict(x3,operator=lambda x,y:dde.grad.jacobian(y,x,i=7,j=0))
V  = np.vstack((V1,V2,V3,V4))
plt.plot(x0, V1, label='PINN',color='green',marker='^',markersize=markersize,\
         markevery=markevery,linewidth=0.7)
plt.plot(x1, V2, label='PINN',color='green',marker='^',markersize=markersize,\
         markevery=markevery,linewidth=0.7)
plt.plot(x2, V3, label='PINN',color='green',marker='^',markersize=markersize,\
         markevery=markevery,linewidth=0.7)
plt.plot(x3, V4, label='PINN',color='green',marker='^',markersize=markersize,\
         markevery=markevery,linewidth=0.7)
plt.plot(X1,ddWW1)
# plt.plot(x,model.predict(x,operator=lambda x,y:dde.grad.jacobian(y,x,i=1,j=0)), label='dy_xx',marker='.',markersize=markersize,markevery=40,linewidth=0.7)
# plt.plot(x,x*0-20,'-.',linewidth=0.7)
plt.ylabel('ddu')
plt.legend()
####
dis = 0.75
interv1 = (X1>=0)&(X1<0.4)
interv2 = (X1>=0.4)&(X1<0.9)
interv3 = (X1>=0.9)&(X1<2)
interv4 = (X1>=2)&(X1<=3)
x_vals_w0 = X1[interv1].reshape(-1,1)
x_vals_w1 = X1[interv2].reshape(-1,1)
x_vals_w2 = X1[interv3].reshape(-1,1)
x_vals_w3 = X1[interv4].reshape(-1,1)
# plt.figure(figsize=(18,7),dpi=150)
plt.subplot(1, 3, 1)
dW1 = WW1[interv1].reshape(-1,1)
dW2 = WW1[interv2].reshape(-1,1)
dW3 = WW1[interv3].reshape(-1,1)
dW4 = WW1[interv4].reshape(-1,1)
plt.plot(x_vals_w0, dW1, label="w1'", linestyle='dashed', color='purple')
plt.plot(x_vals_w1, dW2, label="w2'", linestyle='dashed', color='blue')
plt.plot(x_vals_w2, dW3, label="w3'", linestyle='dashed', color='orange')
plt.plot(x_vals_w3, dW4, label='w4',linestyle='dashed', color='red')
# plt.title('First Derivative')
plt.legend()

plt.subplot(1, 3, 2)
ddW1 = dWW1[interv1].reshape(-1,1)
ddW2 = dWW1[interv2].reshape(-1,1)
ddW3 = dWW1[interv3].reshape(-1,1)
ddW4 = dWW1[interv4].reshape(-1,1)
plt.plot(x_vals_w0, ddW1, label="w1'", linestyle='dashed', color='purple')
plt.plot(x_vals_w1, ddW2, label="w2'", linestyle='dashed', color='blue')
plt.plot(x_vals_w2, ddW3, label="w3'", linestyle='dashed', color='orange')
plt.plot(x_vals_w3, ddW4, label='w4',linestyle='dashed', color='red')
# plt.title('First Derivative')
plt.legend()
plt.savefig('function fitting.png')
