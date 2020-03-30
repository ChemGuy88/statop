#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    hw1_1
'''
import csv, json, re, requests, string, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial as spt

# hw1.1
n = 5
m = 5
r = 2
Y = np.random.rand(n*m)
Y = Y.reshape((n,m))
X = np.random.rand(n*r)
X = X.reshape((n,r))
M = Y.T.dot(X)
U, d, V = sp.linalg.svd(M, full_matrices=False)
D = np.diag(d)
m2 = np.hstack((M,np.zeros((n,m-r))))
u1, d1, v1 = sp.linalg.svd(m2, full_matrices=False)
t = u1.dot(v1.T)
# t = u1.T.dot(v1)
error = np.linalg.norm(Y-X.dot(t[:,:r].T))
print(error)
