"""Python port generated from the original AS-PINNs notebook.

Source notebook: notebooks/force_discontinuity/as_pinns_ex2.ipynb

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
#import datetime
import time
import os
from pylab import mpl
from scipy.signal import chirp, spectrogram
mpl.rcParams['font.sans-serif']=['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus']=False
from matplotlib.pyplot import MultipleLocator
# import xlrd
# import xlwt
from sympy import *
import sympy as sp
# import torch

# %% [cell 2]
def Output(X):
    import re
    lines = open("s1.dat", "r").readlines()
    S = np.array([np.fromstring(min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line),\
                                key=len),sep=",",) for line in lines ])
    dis = S[:,0:num_dis+2][-1]
    interv = [ (X>=dis[i])&(X<dis[i+1]) for i in range(0,dis.shape[0]-1) ]
    out_all = model.predict(X)
    Y = out_all[:, 2: 4*num_NN-num_NN2]
    Out = Y[:,0::num_NN][interv[0].ravel()]
    for i in range(1,num_NN):
        Out = np.vstack((Out, Y[:,i::num_NN][interv[i].ravel()]))
    Out = np.hstack((out_all[:,0:2],Out))
    return [Out[:,i:i+1] for i in range(0,4)]

def Output_dNN(X, j = 0):
    import re
    lines = open("s1.dat", "r").readlines()
    S = np.array([np.fromstring(min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line),\
                                key=len),sep=",",) for line in lines ])
    dis = S[:,0:num_dis+2][-1]
    if j == 0:
        Out_dNN = model.predict(X, operator=lambda x,y:dde.grad.jacobian(y,x, i=j,j=0))
        return -Out_dNN
    if j ==1:
        Out_dNN = model.predict(X, operator=lambda x,y:dde.grad.jacobian(y,x, i=j,j=0))
        return Out_dNN
    interv = [ (X>=dis[i])&(X<dis[i+1]) for i in range(0,dis.shape[0]-1) ]
    Out_dNN = model.predict(X[interv[0].ravel()], operator=lambda x,y:dde.grad.jacobian(y,x, i=num_NN*j-num_NN2,j=0))
    for i in range(1,num_NN):
        Out_dNN = np.vstack((Out_dNN,model.predict(X[interv[i].ravel()], operator=\
                                               lambda x,y:dde.grad.jacobian(y,x, i=num_NN*j+i-num_NN2,j=0)) ))
    if j == 0 or j == 3: #u,fai,M,V   fai_dNN,dfai_dNN,XXX,V_dNN,q_dNN
        Out_dNN = -Out_dNN
    return Out_dNN

def Norm_exact(P = 0):
    import re
    lines = open("s1.dat", "r").readlines()
    S = np.array([np.fromstring(min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line),\
                                key=len),sep=",",) for line in lines ])
    dis = S[:,0:num_dis+2][-1]
    X = X1
    W,fai,M,V = Output(X)
    Q = [Output_dNN(X, j = i) for i in [3]][0]
    EI_exact = NP_EI(X)
    EI = EI_exact
    norm_w = np.linalg.norm(W-WW1,ord=2) / np.linalg.norm(WW1,ord=2)
    norm_fai = np.linalg.norm(fai-dWW1,ord=2) / np.linalg.norm(dWW1,ord=2)
    norm_M = np.linalg.norm(M-ddWW1,ord=2) / np.linalg.norm(ddWW1,ord=2)
    norm_V = np.linalg.norm(V-dddWW1,ord=2) / np.linalg.norm(dddWW1,ord=2)
    norm_Q = np.linalg.norm(Q-ddddWW1,ord=2) / np.linalg.norm(ddddWW1,ord=2)
    norm_EI = np.linalg.norm(EI-EI_exact,ord=2) / np.linalg.norm(EI_exact,ord=2)
    norm = np.hstack((norm_w,norm_fai,norm_M,norm_V,norm_Q))
    Norm = np.hstack((np.array(dis[1:num_dis+1]),norm))
    print(Norm)
    if P == 1:
        return Norm, [W,fai,M,V,Q]
    return Norm

def Norm_NN( X, P = 0):
    u,fai,M,V = Output(X)
    fai_dNN,dfai_dNN,V_dNN,Q_dNN = [Output_dNN(X, j = i) for i in [0,1,2,3]]
    Q = Q_dNN
    M_dNN = dfai_dNN*EI
    EI_exact = NP_EI(X)
    norm_fai_dNN = np.linalg.norm(fai-fai_dNN,ord=2) / np.linalg.norm(fai_dNN,ord=2)
    norm_M_dNN = np.linalg.norm(M-M_dNN,ord=2) / np.linalg.norm(M_dNN,ord=2)
    norm_V_dNN = np.linalg.norm(V-V_dNN,ord=2) / np.linalg.norm(V_dNN,ord=2)
    norm_Q_dNN = np.linalg.norm(Q_dNN-NP_Q(X),ord=2) / np.linalg.norm(NP_Q(X),ord=2)
    norm_EI = np.linalg.norm(EI-EI_exact,ord=2) / np.linalg.norm(EI_exact,ord=2)
    Norm = np.hstack((norm_fai_dNN,norm_M_dNN,norm_V_dNN,norm_Q_dNN))
    print('norm_fai_dNN,norm_M_dNN,norm_V_dNN,norm_Q_dNN')
    print(Norm)
    if P == 1:
        return Norm,[u,fai,M,V,Q],[fai_dNN,dfai_dNN,V_dNN,Q_dNN]
    return Norm

def Plot_dis():
    import re
    lines = open("s1.dat", "r").readlines()
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
    width, height, dpi = 3.35, 2.1, 200
    plt.figure(figsize=(width,height),dpi = dpi)
    markersize = 2
    markevery = 20
    plt.plot(range(0, 200 * l, 200), S[:,1:2], linestyle='dashed',alpha=0.7,linewidth=0.7, \
             marker='^',markersize=markersize,markevery=markevery, label="$s^1$",color='orange')
    plt.plot(range(0, 200 * l, 200), S[:,2:3], linestyle='dashed',alpha=0.7,linewidth=0.7, \
             marker='v',markersize=markersize,markevery=markevery, label="$s^2$",color='lightcoral')
    # plt.plot(range(0, 200 * l, 200), S[:,3:4], linestyle='dashed',alpha=0.7,linewidth=0.7, \
    #          marker='o',markersize=markersize,markevery=markevery, label="$s^3$",color='c')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+1/3,linewidth=0.7, label="$exact$",color='green')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.75,linewidth=0.7, label="$exact$",color='#004B8B')
    # plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.65,linewidth=0.7, label="$exact$",color='purple')
    plt.tick_params(width=0.5, labelsize=6)
    plt.xlabel('Epoch', fontsize=7)
    plt.legend(fontsize=6,frameon=False,bbox_to_anchor=(0.5, 0.35), ncol=2)
    plt.savefig('ex3_s.png',bbox_inches='tight', dpi=800)
    plt.show()
    data_s = np.hstack((np.array([range(0, 200 * l, 200)]).reshape(-1,1),S))
    print(S[-1])
import runpy
from pathlib import Path
globals().update(runpy.run_path(str(Path(__file__).with_name('solution_ex2.py'))))

# %% [cell 3]
d1 = 1/3
d2 = 0.75
s0 = tf.Variable(-0.1, trainable=False, dtype=tf.float32)
s1 = tf.Variable(0.1, trainable=True, dtype=tf.float32)
s2 = tf.Variable(0.75, trainable=False, dtype=tf.float32)
s_end = tf.Variable(L+0.1, trainable=False, dtype=tf.float32)
SS = [s0,s1,s2,s_end]

SS_fig = [-SS[1],s1,s2,2*L-SS[-2]]

num_dis = len(SS)-2
num_NN = num_dis+1

num_NN2 = 2*(num_NN-1)

CC = []
for i in range(3):
    CC.append(tf.Variable(0, trainable=True, dtype=tf.float32))
VV = SS.copy()
VV.extend(CC)
c1 = CC[0]
c2 = CC[1]
c3 = CC[2]

# %% [cell 4]
K = 1234
np.random.seed(K)
tf.set_random_seed(K)

num_domain = 0
epochs = 10000
lr = 1e-3
u_nodenum=20
u_layer=2

activation_func1 = tf.nn.tanh
activation_func2 = tf.nn.elu

l = 0.75
a = 0.25
A = 0.05*0.012
I = 0.012*0.05**3/12
E = 1e4/I*0.001
EI = E*I
EI_real = EI
L = 1.
F= 100.
anchors_x = np.linspace(0,L,201).reshape(-1,1)

def f_sig0(x_in,i):
    cond1 = tf.logical_and(tf.greater_equal(x_in, SS[i]), tf.less(x_in, SS[i+1]))
    f1 = tf.ones_like(x_in)
    f2 = tf.zeros_like(x_in)
    f = tf.where(cond1, f1, f2)
    return f

def f(x_in):
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0), tf.less(x_in, 1/3))
    f1 = 30 - 30 * x_in
    cond2 = tf.logical_and(tf.greater_equal(x_in, 1/3), tf.less(x_in, 3/4))
    f2 = 20 * tf.ones_like(x_in)
    cond3 = tf.logical_and(tf.greater_equal(x_in, 3/4), tf.less_equal(x_in, 1))
    f3 = tf.zeros_like(x_in)
    f = tf.where(cond1, f1, tf.where(cond2, f2, f3))
    return f

def g(x_in):
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0), tf.less(x_in, 1/3))
    f1 = - 30*tf.ones_like(x_in)
    cond2 = tf.logical_and(tf.greater_equal(x_in, 1/3), tf.less_equal(x_in, 1))
    f2 = tf.zeros_like(x_in)
    f = tf.where(cond1, f1, f2)
    return f

def NP_Q(x_in):
    cond1 = np.logical_and(np.greater_equal(x_in, 0), np.less(x_in, 1/3))
    f1 = 30 - 30 * x_in
    cond2 = np.logical_and(np.greater_equal(x_in, 1/3), np.less(x_in, 3/4))
    f2 = 20 * np.ones_like(x_in)
    cond3 = np.logical_and(np.greater_equal(x_in, 3/4), np.less_equal(x_in, 1))
    f3 = np.zeros_like(x_in)
    f = np.where(cond1, f1, np.where(cond2, f2, f3))
    return f

def TF_EI(x_in):
    f1 = 10*tf.ones_like(x_in)
    return f1

def NP_EI(x_in):
    f1 = 10*np.ones_like(x_in)
    return f1

def beampde_err(x, y):
    x_in = x[:,0:1]
    u    = y[:,0  :1]
    Fai  = y[:,1  :2]
    M    = y[:,num_NN*0+2:num_NN*1+2]
    V    = y[:,num_NN*1+2:num_NN*2+2]

    Fai_dNN = []
    M_dNN = []
    V_dNN = []
    Q_dNN = []
    DEI_dNN = []
    dQ_dNN = []

    fai_dNN = -dde.grad.jacobian(u, x, i = 0, j=0)
    dfai_dNN =  dde.grad.jacobian(Fai, x, i = 0, j=0)
    m_dNN = EI_real * dfai_dNN

    for i in range(num_NN):
        v_dNN = dde.grad.jacobian(M, x, i = i, j=0)
        q_dNN = -dde.grad.jacobian(V, x, i = i, j=0)
        dq_dNN = dde.grad.jacobian(q_dNN, x, i = 0, j=0)
        # Fai_dNN.append(fai_dNN)
        # M_dNN.append(m_dNN)
        V_dNN.append(v_dNN)
        Q_dNN.append(q_dNN)
        dQ_dNN.append(dq_dNN)
    f_sig = []
    for i in range(num_NN):
        f_sig.append( 10*tf.keras.activations.relu((x_in-SS_fig[i])) \
                     * tf.keras.activations.relu((-x_in+SS_fig[i+1])) )
    # f_sig.append(10*tf.keras.activations.relu(-x_in+s1))
    # f_sig.append(10*tf.keras.activations.relu(x_in-s1))
    loss_Fai = 0
    loss_M = 0
    loss_V = 0
    loss_Q = 0
    loss_dQ = 0
    loss_Fai = loss_Fai + (Fai - fai_dNN)
    for i in range(num_NN):
        loss_M = loss_M + (M[:,i:i+1]-m_dNN   )               * f_sig0(x_in,i)
        loss_V = loss_V + (V[:,i:i+1]-V_dNN[i])               * f_sig0(x_in,i)
        loss_Q = loss_Q + (f(x_in)-Q_dNN[i])                  * f_sig0(x_in,i)
        loss_dQ = loss_dQ + (g(x_in)-dQ_dNN[i])               * f_sig[i]
    return [loss_Fai,
            loss_M,\
            loss_V,
            loss_Q,loss_dQ]

def beampde2(x, y):
    x_in = x[:,0:1]
    u    = y[:,0  :1]
    Fai  = y[:,1  :2]
    M    = y[:,num_NN*0+2:num_NN*1+2]
    V    = y[:,num_NN*1+2:num_NN*2+2]

    Fai_dNN = []
    M_dNN = []
    V_dNN = []
    Q_dNN = []
    DEI_dNN = []
    dQ_dNN = []

    fai_dNN = -dde.grad.jacobian(u, x, i = 0, j=0)
    dfai_dNN =  dde.grad.jacobian(Fai, x, i = 0, j=0)
    m_dNN = EI_real * dfai_dNN

    for i in range(num_NN):
        v_dNN = dde.grad.jacobian(M, x, i = i, j=0)
        q_dNN = -dde.grad.jacobian(V, x, i = i, j=0)
        dq_dNN = dde.grad.jacobian(q_dNN, x, i = 0, j=0)
        # Fai_dNN.append(fai_dNN)
        # M_dNN.append(m_dNN)
        dQ_dNN.append(dq_dNN)
        V_dNN.append(v_dNN)
        Q_dNN.append(q_dNN)
    f_sig = []
    for i in range(num_NN):
        f_sig.append( 10*tf.keras.activations.relu((x_in-SS_fig[i])) \
                     * tf.keras.activations.relu((-x_in+SS_fig[i+1])) )
    # f_sig.append(10*tf.keras.activations.relu(-x_in+s1))
    # f_sig.append(10*tf.keras.activations.relu(x_in-s1))
    loss_Fai = 0
    loss_M = 0
    loss_V = 0
    loss_Q = 0
    loss_dQ = 0

    loss_Fai = loss_Fai + (Fai - fai_dNN)
    for i in range(num_NN):
        loss_M = loss_M + (M[:,i:i+1]-m_dNN   )               * f_sig[i]
        loss_V = loss_V + (V[:,i:i+1]-V_dNN[i])               * f_sig[i]
        loss_Q = loss_Q + (f(x_in)-Q_dNN[i])                  * f_sig[i]
        loss_dQ = loss_dQ + (g(x_in)-dQ_dNN[i])               * f_sig[i]
    return [loss_Fai,
            loss_M,\
            loss_V,
            loss_Q,
            loss_dQ]

geom = dde.geometry.Interval(0, L)
data = dde.data.TimePDE(geom,beampde2,[], num_domain=num_domain,anchors=anchors_x)

net = dde.maps.FNN([1]  + [50]*3 + [num_NN*2+2], "tanh", "Glorot uniform")

def modify_output(X, y):
    x = X[:,0:1]
    u    = y[:,0  :1]
    Fai  = y[:,1  :2]
    M    = y[:,num_NN*0+2:num_NN*1+2]
    V    = y[:,num_NN*1+2:num_NN*2+2]

    final_output = tf.concat([u[:,0:1]*x*(x-l),\
                              Fai[:,0:1]*x, \
                              M[:,0:1]*(x-s1) +  10*c1, \
                              M[:,1:2]*(x-s1)*(x-s2) + 10*c1*(x-s2)/(s1-s2)+10*c2*(x-s1)/(s2-s1),\
                              M[:,2:3]*(x-s2)*(x-L) + 10*c2*(x-L)/(s2-L),\
                              10*V[:,0:1]*(x-s1)+10*c3,\
                              10*V[:,1:2]*(x-s1)+10*c3,\
                              10*V[:,2:3]*(x-L)+100], axis=1)
    return final_output

net.apply_output_transform(modify_output)
variable = dde.callbacks.VariableValue(VV, period=200, filename="s1.dat")#200为epochs之间的间隔
model = dde.Model(data, net)

loss = []
Weight = np.ones(5)
Weight[-1] = 1
for i in range(5):
    loss.append('MSE')

model.compile("adam", lr, loss_weights=list(Weight.ravel()), loss = loss)
losshistory, train_state = model.train(epochs=10000, callbacks=[variable])
list_reslut = Norm_exact()
List_weight = []
list_point = []
list_Norm_dNN = []
list_NN =[]
list_dNN =[]
Norm_dNN,[u,fai,M,V,Q],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
list_Norm_dNN.append(Norm_dNN)
list_NN.append(np.array([u,fai,M,V,Q]).reshape(5,-1).T)
list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
np.savetxt('1.txt', list_reslut)
Plot_dis()

# %% [cell 5]
for i in range(3):
    Weight = np.hstack((Norm_dNN/np.max(Norm_dNN),1))
    print('Weight',Weight)
    model.compile("adam", 1e-4, loss_weights=list(Weight.ravel()), loss = loss)
    X = np.random.uniform(0,L,10000)
    X = np.sort(X,axis = 0).reshape(-1,1)
    err_total = np.abs(model.predict(X, operator=beampde_err))  #计算前6项残差
    err_eq = np.sum(err_total,axis=0)             #每项残差除以均值进行缩放，每个点，残差求和
    print("Mean residual: %.3e" % (np.mean(err_total)))
    x_ids = np.argsort(-err_eq,axis=0 )[:20]
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        list_point.append(X[elem][0][0])
        data.add_anchors(X[elem])
    losshistory, train_state = model.train(epochs=10000, callbacks=[variable])
    Norm_dNN,[u,fai,M,V,Q],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
    list_Norm_dNN.append(Norm_dNN)
    list_NN.append(np.array([u,fai,M,V,Q]).reshape(5,-1).T)
    list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
    list_reslut = np.vstack(( list_reslut, Norm_exact() ))
    np.savetxt('1.txt', list_reslut)
    Plot_dis()

# %% [cell 6]
for i in range(6):
    Weight = np.hstack((Norm_dNN/np.max(Norm_dNN),1))
    print('Weight',Weight)
    # model.compile("adam", 1e-4, loss_weights=list(Weight.ravel()), loss = loss)
    X = np.random.uniform(0,L,10000)
    X = np.sort(X,axis = 0).reshape(-1,1)
    err_total = np.abs(model.predict(X, operator=beampde_err))  #计算前6项残差
    err_eq = np.sum(err_total,axis=0)             #每项残差除以均值进行缩放，每个点，残差求和
    print("Mean residual: %.3e" % (np.mean(err_total)))
    x_ids = np.argsort(-err_eq,axis=0 )[:20]
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        list_point.append(X[elem][0][0])
        data.add_anchors(X[elem])
    losshistory, train_state = model.train(epochs=10000, callbacks=[variable])
    Norm_dNN,[u,fai,M,V,Q],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
    list_Norm_dNN.append(Norm_dNN)
    list_NN.append(np.array([u,fai,M,V,Q]).reshape(5,-1).T)
    list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
    list_reslut = np.vstack(( list_reslut, Norm_exact() ))
    np.savetxt('1.txt', list_reslut)
    Plot_dis()

# %% [cell 8]
Norm,[W,Fai,M,V,Q] = Norm_exact(P =1)
plt.rcParams.update({'font.size': 10.5})
markersize=8
markevery=150
X2 = X1
plt.figure(figsize=(18,7),dpi=150)
plt.subplot(2,3,1)
plt.plot(X2, W, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('W')
plt.legend()

plt.subplot(2,3,2)
plt.plot(X2, fai, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('fai')
plt.legend()

plt.subplot(2,3,3)
plt.plot(X2, dfai_dNN, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('dfai')
plt.legend()

plt.subplot(2,3,4)
plt.plot(X2, M, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('M')
plt.legend()

plt.subplot(2,3,5)
plt.plot(X2, V, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('V')
plt.legend()

plt.subplot(2,3,6)
plt.plot(X2, Q, label='MD-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('Q')
plt.legend()



plt.subplot(2,3,1)
# plt.vlines(0.75,-0.05,0.15)
# plt.hlines(0,0.6,0.8)
plt.plot(X1,WW1,label="W", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(2,3,2)
plt.plot(X1,dWW1,label="fai", linestyle='dashdot', color='orange')
plt.legend()

plt.subplot(2,3,3)
plt.plot(X1,dfai_real,label="Dfai", linestyle='dashdot', color='orange')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(X1,ddWW1,label="M", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(2, 3, 5)
plt.plot(X1,dddWW1,label="V", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(2, 3, 6)
plt.plot(X1,ddddWW1,label="Q", linestyle='dashdot', color='orange')
plt.legend()
