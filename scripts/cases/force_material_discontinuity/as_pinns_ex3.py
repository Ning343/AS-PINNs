"""Executable AS-PINNs Python case script.

Case: force_material_discontinuity/as_pinns_ex3.
Reusable, tested project code lives under src/as_pinns/.
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
# import time
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
    Y = out_all[:, 0: 5*num_NN]
    Out = Y[:,0::num_NN][interv[0].ravel()]
    for i in range(1,num_NN):
        Out = np.vstack((Out, Y[:,i::num_NN][interv[i].ravel()]))
    return [Out[:,i:i+1] for i in range(0,5)]

def Output_dNN(X, j = 0):
    import re
    lines = open("s1.dat", "r").readlines()
    S = np.array([np.fromstring(min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line),\
                                key=len),sep=",",) for line in lines ])
    dis = S[:,0:num_dis+2][-1]
    interv = [ (X>=dis[i])&(X<dis[i+1]) for i in range(0,dis.shape[0]-1) ]
    Out_dNN = model.predict(X[interv[0].ravel()], operator=lambda x,y:dde.grad.jacobian(y,x, i=num_NN*j,j=0))
    for i in range(1,num_NN):
        Out_dNN = np.vstack((Out_dNN,model.predict(X[interv[i].ravel()], operator=\
                                               lambda x,y:dde.grad.jacobian(y,x, i=num_NN*j+i,j=0)) ))
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
    W,fai,M,V,EI = Output(X)
    Q = [Output_dNN(X, j = i) for i in [3]][0]
    EI_exact = NP_EI(X)
    norm_w = np.linalg.norm(W-WW1,ord=2) / np.linalg.norm(WW1,ord=2)
    norm_fai = np.linalg.norm(fai-dWW1,ord=2) / np.linalg.norm(dWW1,ord=2)
    norm_M = np.linalg.norm(M-ddWW1,ord=2) / np.linalg.norm(ddWW1,ord=2)
    norm_V = np.linalg.norm(V-dddWW1,ord=2) / np.linalg.norm(dddWW1,ord=2)
    norm_Q = np.linalg.norm(Q-ddddWW1,ord=2) / np.linalg.norm(ddddWW1,ord=2)
    norm_EI = np.linalg.norm(EI-EI_exact,ord=2) / np.linalg.norm(EI_exact,ord=2)
    norm = np.hstack((norm_w,norm_fai,norm_M,norm_V,norm_Q,norm_EI))
    Norm = np.hstack((np.array(dis[1:num_dis+1]),norm))
    print(Norm)
    if P == 1:
        return Norm, [W,fai,M,V,Q,EI]
    return Norm

def Norm_NN( X, P = 0):
    u,fai,M,V,EI = Output(X)
    fai_dNN,dfai_dNN,V_dNN,Q_dNN = [Output_dNN(X, j = i) for i in [0,1,2,3]]
    Q = Q_dNN
    M_dNN = dfai_dNN*EI
    EI_exact = NP_EI(X)
    norm_fai_dNN = np.linalg.norm(fai-fai_dNN,ord=2) / np.linalg.norm(fai_dNN,ord=2)
    norm_M_dNN = np.linalg.norm(M-M_dNN,ord=2) / np.linalg.norm(M_dNN,ord=2)
    norm_V_dNN = np.linalg.norm(V-V_dNN,ord=2) / np.linalg.norm(V_dNN,ord=2)
    norm_Q_dNN = np.linalg.norm(Q_dNN-NP_Q(X),ord=2) / np.linalg.norm(NP_Q(X),ord=2)
    norm_EI = np.linalg.norm(EI-EI_exact,ord=2) / np.linalg.norm(EI_exact,ord=2)
    Norm = np.hstack((norm_fai_dNN,norm_M_dNN,norm_V_dNN,norm_Q_dNN,norm_EI))
    print('norm_fai_dNN,norm_M_dNN,norm_V_dNN,norm_Q_dNN,norm_EI')
    print(Norm)
    if P == 1:
        return Norm,[u,fai,M,V,Q,EI],[fai_dNN,dfai_dNN,V_dNN,Q_dNN]
    return Norm
import runpy
from pathlib import Path
globals().update(runpy.run_path(str(Path(__file__).with_name('solution_ex3.py'))))

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
    plt.plot(range(0, 200 * l, 200), S[:,3:4], linestyle='dashed',alpha=0.7,linewidth=0.7, \
             marker='o',markersize=markersize,markevery=markevery, label="$s^3$",color='c')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.35,linewidth=0.7, label="$exact$",color='green')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.5,linewidth=0.7, label="$exact$",color='#004B8B')
    plt.plot(range(0, 200 * l, 200), S[:,0:1]*0+0.65,linewidth=0.7, label="$exact$",color='purple')
    plt.tick_params(width=0.5, labelsize=6)
    plt.xlabel('Epoch', fontsize=7)
    plt.legend(fontsize=6,frameon=False,bbox_to_anchor=(0.5, 0.35), ncol=2)
    plt.show()

# %% [cell 3]
s0 = tf.Variable(-0.1, trainable=False, dtype=tf.float32)
s1 = tf.Variable(0.1, trainable=True, dtype=tf.float32)
s2 = tf.Variable(0.2, trainable=True, dtype=tf.float32)
s3 = tf.Variable(0.3, trainable=True, dtype=tf.float32)
s_end = tf.Variable(L+0.1, trainable=False, dtype=tf.float32)
SS = [s0,s1,s2,s3,s_end]

SS_fig = [-s1,s1,s2,s3,2*L-s3]

num_dis = len(SS)-2
num_NN = num_dis+1

CC = []
for i in range(num_dis*6):
    CC.append(tf.Variable(0, trainable=True, dtype=tf.float32))
VV = SS.copy()
VV.extend(CC)
c1 = CC[0]
c2 = CC[1]
c3 = CC[2]
c4 = CC[3]
c5 = CC[4]
c6 = CC[5]
c7 = CC[6]
c8 = CC[7]
c9 = CC[8]
c10 = CC[9]
c11 = CC[10]
c12 = CC[11]

# %% [cell 4]
K = 1234
np.random.seed(K)
tf.set_random_seed(K)

num_domain = 0
epochs = 10000
lr = 1e-3
u_nodenum=50
u_layer=2

activation_func1 = tf.nn.tanh
activation_func2 = tf.nn.elu

L = 1
anchors_x = np.random.uniform(0,L,200).reshape(-1,1)

def f(x_in):
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0), tf.less(x_in, 0.35))
    f1 = tf.zeros_like(x_in)
    cond2 = tf.logical_and(tf.greater_equal(x_in, 0.35), tf.less(x_in, 0.65))
    f2 = 20 * tf.ones_like(x_in)
    f3 = tf.zeros_like(x_in)
    f = tf.where(cond1, f1, tf.where(cond2, f2, f3))
    return f
def NP_Q(x_in):
    cond1 = np.logical_and(x_in >= 0, x_in < 0.35)
    cond2 = np.logical_and(x_in >= 0.35, x_in < 0.65)

    f1 = np.zeros_like(x_in)
    f2 = 20 * np.ones_like(x_in)
    f3 = np.zeros_like(x_in)

    f = np.where(cond1, f1, np.where(cond2, f2, f3))
    return f
def TF_EI(x_in):
    I = 0.012*0.05**3/12
    E = 1e4/I*0.001
    cond1 = tf.logical_and(tf.greater_equal(x_in, 0), tf.less(x_in, 0.5))
    f1 = E*I*tf.ones_like(x_in)
    f2 = 0.5*E*I*tf.ones_like(x_in)
    f = tf.where(cond1, f1, f2)
    return f

def NP_EI(x_in):
    I = 0.012*0.05**3/12
    E = 1e4/I*0.001
    cond1 = np.logical_and(np.greater_equal(x_in, 0), np.less(x_in, 0.5))
    f1 = E*I*np.ones_like(x_in)
    f2 = 0.5*E*I*np.ones_like(x_in)
    f = np.where(cond1, f1, f2)
    return f

def beampde2(x, y):
    x_in = x[:,0:1]
    u    = y[:,0:num_NN]
    Fai  = y[:,num_NN*1:num_NN*2]
    M    = y[:,num_NN*2:num_NN*3]
    V    = y[:,num_NN*3:num_NN*4]
    EI   = y[:,num_NN*4:num_NN*5]

    Fai_dNN = []
    M_dNN = []
    V_dNN = []
    Q_dNN = []
    DEI_dNN = []

    for i in range(num_NN):
        fai_dNN = -dde.grad.jacobian(u, x, i = i, j=0)
        dfai_dNN =  dde.grad.jacobian(Fai, x, i = i, j=0)
        m_dNN = EI[:,i:i+1] * dfai_dNN
        v_dNN = dde.grad.jacobian(M, x, i = i, j=0)
        q_dNN = -dde.grad.jacobian(V, x, i = i, j=0)
        dEI_dNN = dde.grad.jacobian(EI, x, i = i, j=0)

        Fai_dNN.append(fai_dNN)
        M_dNN.append(m_dNN)
        V_dNN.append(v_dNN)
        Q_dNN.append(q_dNN)
        DEI_dNN.append(dEI_dNN)

    f_sig = []
    for i in range(num_NN):
        f_sig.append( 10*tf.keras.activations.relu((x_in-SS_fig[i])) \
                     * tf.keras.activations.relu((-x_in+SS_fig[i+1])) )
    loss_Fai = 0
    loss_M = 0
    loss_V = 0
    loss_Q = 0
    loss_EI = 0
    loss_dEI = 0

    for i in range(num_NN):
        loss_Fai = loss_Fai + (Fai[:,i:i+1]-Fai_dNN[i])       * f_sig[i]
        loss_M = loss_M + (M[:,i:i+1]-M_dNN[i])               * f_sig[i]
        loss_V = loss_V + (V[:,i:i+1]-V_dNN[i])               * f_sig[i]
        loss_Q = loss_Q + (f(x_in)-Q_dNN[i])                  * f_sig[i]
        loss_EI = loss_EI + (TF_EI(x_in)-EI[:,i:i+1])         * f_sig[i]
        loss_dEI = loss_dEI + DEI_dNN[i]                      * f_sig[i]
    return [loss_Fai,
            loss_M,\
            loss_V,
            loss_Q,
            loss_EI,
            loss_dEI]

geom = dde.geometry.Interval(0, L)
data = dde.data.TimePDE(geom,beampde2,[], num_domain=num_domain,anchors=anchors_x)

net = dde.maps.FNN([1]  + [50]*3 + [num_NN*5], "tanh", "Glorot uniform")

def modify_output(X, y):
    x = X[:,0:1]
    u    = y[:,num_NN*0:num_NN*1]
    Fai  = y[:,num_NN*1:num_NN*2]
    M    = y[:,num_NN*2:num_NN*3]
    V    = y[:,num_NN*3:num_NN*4]
    EI   = y[:,num_NN*4:num_NN*5]

    final_output = tf.concat([0.01*(x*u[:,0:1]*(x-s1)+10*c1*x/s1),\
                              0.01*(u[:,1:2]*(x-s1)*(x-s2)+10*c1*(x-s2)/(s1-s2)+10*c2*(x-s1)/(s2-s1)),\
                              0.01*(u[:,2:3]*(x-s2)*(x-s3)+10*c2*(x-s3)/(s2-s3)+10*c3*(x-s2)/(s3-s2)),\
                              0.01*(u[:,3:4]*(x-L)*(x-s3)+10*c3*(x-L)/(s3-L)),\
                              0.01*(Fai[:,0:1]*(x-s1)+10*c4), \
                              0.01*(Fai[:,1:2]*(x-s1)*(x-s2)+10*c4*(x-s2)/(s1-s2)+10*c5*(x-s1)/(s2-s1)),\
                              0.01*(Fai[:,2:3]*(x-s2)*(x-s3)+10*c5*(x-s3)/(s2-s3)+10*c6*(x-s2)/(s3-s2)),\
                              0.01*(Fai[:,3:4]*(x-s3)+10*c6), \
                              x*M[:,0:1]*(x-s1)+10*c7*x/s1,\
                              M[:,1:2]*(x-s1)*(x-s2)+10*c7*(x-s2)/(s1-s2)+10*c8*(x-s1)/(s2-s1),\
                              M[:,2:3]*(x-s2)*(x-s3)+10*c8*(x-s3)/(s2-s3)+10*c9*(x-s2)/(s3-s2),\
                              M[:,3:4]*(x-L)*(x-s3)+10*c9*(x-L)/(s3-L),\
                              V[:,0:1]*(x-s1)+10*c10,\
                              V[:,1:2]*(x-s1)*(x-s2)+10*c10*(x-s2)/(s1-s2)+10*c11*(x-s1)/(s2-s1),\
                              V[:,2:3]*(x-s2)*(x-s3)+10*c11*(x-s3)/(s2-s3)+10*c12*(x-s2)/(s3-s2),\
                              V[:,3:4]*(x-s3)+10*c12,\
                              EI], axis=1)
    return final_output

net.apply_output_transform(modify_output)
variable = dde.callbacks.VariableValue(VV, period=200, filename="s1.dat")#200为epochs之间的间隔
model = dde.Model(data, net)

loss = []
Weight = np.ones(6)
for i in range(6):
    loss.append('MSE')

model.compile("adam", lr, loss_weights=list(Weight.ravel()), loss = loss)
losshistory, train_state = model.train(epochs=20000, callbacks=[variable])
list_reslut = Norm_exact()
List_weight = []
list_point = []
list_Norm_dNN = []
list_NN =[]
list_dNN =[]
Norm_dNN,[u,fai,M,V,Q,EI],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
list_Norm_dNN.append(Norm_dNN)
list_NN.append(np.array([u,fai,M,V,Q,EI]).reshape(6,-1).T)
list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
Plot_dis()

# %% [cell 5]
for i in range(2):
    Weight = np.hstack((Norm_dNN/np.min(Norm_dNN), 1))
    print('Weight',Weight)
    model.compile("adam", 1e-3, loss_weights=list(Weight.ravel()), loss = loss)
    X = np.random.uniform(0,L,2000)
    X = np.sort(X,axis = 0).reshape(-1,1)
    err_total = np.abs(model.predict(X, operator=beampde2))  #计算前6项残差
    err_eq = np.sum(err_total,axis=0)             #每项残差除以均值进行缩放，每个点，残差求和
    print("Mean residual: %.3e" % (np.mean(err_total)))
    x_ids = np.argsort(-err_eq,axis=0 )[:20]
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        list_point.append(X[elem][0][0])
        data.add_anchors(X[elem])
    losshistory, train_state = model.train(epochs=20000, callbacks=[variable])
    Norm_dNN,[u,fai,M,V,Q,EI],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
    list_Norm_dNN.append(Norm_dNN)
    list_NN.append(np.array([u,fai,M,V,Q,EI]).reshape(6,-1).T)
    list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
    list_reslut = np.vstack(( list_reslut, Norm_exact() ))
    np.savetxt('1.txt', list_reslut)
    Plot_dis()

# %% [cell 6]
Weight = np.hstack((Norm_dNN/np.min(Norm_dNN), 1))
for i in range(4):
    print('Weight',Weight)
    if i ==0:
        model.compile("adam", 1e-4, loss_weights=list(Weight.ravel()), loss = loss)
    X = np.random.uniform(0,L,2000)
    X = np.sort(X,axis = 0).reshape(-1,1)
    err_total = np.abs(model.predict(X, operator=beampde2))  #计算前6项残差
    err_eq = np.sum(err_total,axis=0)             #每项残差除以均值进行缩放，每个点，残差求和
    print("Mean residual: %.3e" % (np.mean(err_total)))
    x_ids = np.argsort(-err_eq,axis=0 )[:20]
    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        list_point.append(X[elem][0][0])
        data.add_anchors(X[elem])
    losshistory, train_state = model.train(epochs=20000, callbacks=[variable])
    Norm_dNN,[u,fai,M,V,Q,EI],[fai_dNN,dfai_dNN,V_dNN,Q_dNN] = Norm_NN(X1, P = 1)
    list_Norm_dNN.append(Norm_dNN)
    list_NN.append(np.array([u,fai,M,V,Q,EI]).reshape(6,-1).T)
    list_dNN.append(np.array([fai_dNN,dfai_dNN,V_dNN,Q_dNN]).reshape(4,-1).T)
    list_reslut = np.vstack(( list_reslut, Norm_exact() ))
    np.savetxt('1.txt', list_reslut)
    Plot_dis()

# %% [cell 7]
Norm,[W,Fai,M,V,Q,EI] = Norm_exact(P =1)
plt.rcParams.update({'font.size': 10.5})
markersize=8
markevery=150
X2 = X1
plt.figure(figsize=(18,10),dpi=150)
plt.subplot(3,3,1)
plt.plot(X2, W, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('W')
plt.legend()

plt.subplot(3,3,2)
plt.plot(X2, fai, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('fai')
plt.legend()

plt.subplot(3,3,3)
plt.plot(X2, dfai_dNN, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('dfai')
plt.legend()

plt.subplot(3,3,4)
plt.plot(X2, M, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('M')
plt.legend()

plt.subplot(3,3,5)
plt.plot(X2, V, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('V')
plt.legend()

plt.subplot(3,3,6)
plt.plot(X2, Q, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
plt.ylabel('Q')
plt.legend()
plt.subplot(3, 3, 7)
plt.plot(X2, EI/EI_real, label='AS-PINN',color='green',marker='^',markersize=markersize,markevery=markevery,linewidth=0.7)
# plt.ylim(0,2)
plt.legend()



plt.subplot(3,3,1)
# plt.vlines(0.75,-0.05,0.15)
# plt.hlines(0,0.6,0.8)
plt.plot(X1,WW1,label="W", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(3,3,2)
plt.plot(X1,dWW1,label="fai", linestyle='dashdot', color='orange')
plt.legend()

plt.subplot(3,3,3)
plt.plot(X1,dfai_real,label="Dfai", linestyle='dashdot', color='orange')
plt.legend()

plt.subplot(3, 3, 4)
plt.plot(X1,ddWW1,label="M", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(3, 3, 5)
plt.plot(X1,dddWW1,label="V", linestyle='dashdot', color='orange')
plt.legend()
plt.subplot(3, 3, 6)
plt.plot(X1,ddddWW1,label="Q", linestyle='dashdot', color='orange')
plt.legend()

plt.subplot(3, 3, 7)
plt.plot(X2, NP_EI(X2)/EI_real, label="EI", linestyle='dashdot', color='orange')
# plt.ylim(0,2)
plt.legend()
