"""Python port generated from the original AS-PINNs notebook.

Source notebook: notebooks/function_fitting/solution_ex1.ipynb

This file keeps the original executable cell order for traceability. Reusable, tested project code lives under src/as_pinns/.
"""


# %% [cell 1]
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import *

l1 = 0.4
l2 = 0.9
l3 = 2
L =3
x_sym = sp.symbols('x')
# 定义函数
fai1_sym = x_sym*0+4
fai2_sym = -2*x_sym+4.8
fai3_sym = x_sym*0+3
fai4_sym = -x_sym*0+2

y1_sym = integrate(fai1_sym, (x_sym, 0, x_sym))
y2_sym = integrate(fai1_sym, (x_sym, 0, l1)) + integrate(fai2_sym,(x_sym,l1,x_sym))
y3_sym = integrate(fai1_sym, (x_sym, 0, l1)) + integrate(fai2_sym,(x_sym,l1,l2)) + integrate(fai3_sym,(x_sym,l2,x_sym))
y4_sym = integrate(fai1_sym, (x_sym, 0, l1)) + integrate(fai2_sym,(x_sym,l1,l2)) + integrate(fai3_sym,(x_sym,l2,l3))\
          +integrate(fai4_sym,(x_sym,l3,x_sym))
M1_sym = sp.diff(fai1_sym,x_sym)
M2_sym = sp.diff(fai2_sym,x_sym)
M3_sym = sp.diff(fai3_sym,x_sym)
M4_sym = sp.diff(fai4_sym,x_sym)

fai1_func = sp.lambdify(x_sym, fai1_sym, 'numpy')
fai2_func = sp.lambdify(x_sym, fai2_sym, 'numpy')
fai3_func = sp.lambdify(x_sym, fai3_sym, 'numpy')
fai4_func = sp.lambdify(x_sym, fai4_sym, 'numpy')
y1_func = sp.lambdify(x_sym, y1_sym, 'numpy')
y2_func = sp.lambdify(x_sym, y2_sym, 'numpy')
y3_func = sp.lambdify(x_sym, y3_sym, 'numpy')
y4_func = sp.lambdify(x_sym, y4_sym, 'numpy')
M1_func = sp.lambdify(x_sym, M1_sym, 'numpy')
M2_func = sp.lambdify(x_sym, M2_sym, 'numpy')
M3_func = sp.lambdify(x_sym, M3_sym, 'numpy')
M4_func = sp.lambdify(x_sym, M4_sym, 'numpy')
x1 = np.arange(0,l1,0.001).reshape(-1,1)
x2 = np.arange( l1,l2,0.001).reshape(-1,1)
x3 = np.arange( l2,l3,0.001).reshape(-1,1)
x4 = np.arange( l3,L,0.001).reshape(-1,1)
X1  = np.arange(0,L,0.001).reshape(-1,1)
dWW1 = x1*0+fai1_func(x1)
dWW2 = x2*0+fai2_func(x2)
dWW3 = x3*0+fai3_func(x3)
dWW4 = x4*0+fai4_func(x4)
WW1 = y1_func(x1)
WW2 = y2_func(x2)
WW3 = y3_func(x3)
WW4 = y4_func(x4)
ddWW1 = x1*0+M1_func(x1)
ddWW2 = x2*0+M2_func(x2)
ddWW3 = x3*0+M3_func(x3)
ddWW4 = x4*0+M4_func(x4)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(X1,np.vstack((WW1,WW2,WW3,WW4)))
plt.subplot(1,3,2)
plt.plot(X1,np.vstack((dWW1,dWW2,dWW3,dWW4)))
plt.subplot(1,3,3)
plt.plot(X1,np.vstack((ddWW1,ddWW2,ddWW3,ddWW4)))

# %% [cell 2]
dWW1 = np.vstack((dWW1,dWW2,dWW3,dWW4))
WW1 = np.vstack((WW1,WW2,WW3, WW4))
ddWW1 = np.vstack((ddWW1,ddWW2,ddWW3, ddWW4))
