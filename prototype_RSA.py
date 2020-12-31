# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:00:27 2020
Kode Python
@author: aluthfian
"""
import sympy as sp
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def remainDiv(M,e_Val,nVal):
    e_Val = bin(e_Val)[2:]
    C = 1
    for i in range(len(e_Val)):
        C = (C**2)%nVal
        if e_Val[i] == '1':
            C = (C*M)%nVal
    return C

def Find_E_Value(p,q,d):
    phi = (p-1)*(q-1)
    e_Val = modInverse(d, phi)
    return e_Val
    
    
def modInverse(a, m):
  # function by Nikita Tiwari in geeksforgeeks.org
  # a < m
    m0 = m 
    y = 0
    x = 1
  
    if (m == 1): 
        return 0
  
    while (a > 1): 
  
        # q is quotient 
        q = a // m 
  
        t = m 
  
        # m is remainder now, process 
        # same as Euclid's algo 
        m = a % m 
        a = t 
        t = y 
  
        # Update x and y 
        y = x - q * y 
        x = t 
  
    # Make x positive 
    if (x < 0): 
        x = x + m0 
  
    return x

# %% The system generates two values for the public key
primeVal = [i for i in sp.primerange(1000, 5000)]
idx_p = random.randint(1, high=500)
p = primeVal[idx_p]
idx_q = random.randint(1, high=500)
q = primeVal[idx_q]
primeVal = [i for i in sp.primerange(max(p,q), 10000)]
idx_d = random.randint(1, high=500)
dVal = primeVal[idx_d]

# public keys
nVal = p*q
e_Val = Find_E_Value(p,q,dVal)

# %% A message
msg = r'Jokowi is actually a good leader'

# %% Encryption
num_msg = [ord(elem) for elem in msg]
C_msg = [remainDiv(elem,e_Val,nVal) for elem in num_msg]

# %% image plotting
sz_img = np.int(np.rint(np.sqrt(len(msg))))
x_dot = np.arange(sz_img)
y_dot = np.arange(sz_img)
X, Y = np.meshgrid(x_dot, y_dot)
pad_size = sz_img**2 - len(C_msg)
img_col = np.pad(np.remainder(C_msg,256), (0, pad_size),\
                 'constant', constant_values=0)
img_col = np.reshape(img_col, (sz_img, sz_img))
extent = np.min(x_dot), np.max(x_dot), np.min(y_dot), np.max(y_dot)
fig = plt.figure(frameon=False)
im1 = plt.imshow(img_col, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)
# %% Decryption
num_dec = [remainDiv(elem,dVal,nVal) for elem in C_msg]
msg_dec = ''.join([chr(elem) for elem in num_dec])