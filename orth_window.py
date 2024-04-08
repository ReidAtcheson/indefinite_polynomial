import numpy as np
import scipy.linalg as la
import scipy.sparse as sp 
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import legvander
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import poly


seed=239478
rng=np.random.default_rng(seed)

m = 1024
bands = [0,1,32]

A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A = 0.5*(A+A.T)
A = A / spla.norm(A,ord=np.inf)

v = rng.uniform(-1,1,size=m)
k=5
V = np.zeros((m,k+1))
V[:,0] = v/np.linalg.norm(v)
for i in range(k):
    V[:,i+1] = A @ V[:,i]


V[:,0:4],_ = la.qr(V[:,0:4],mode="economic")
V[:,4:],_ = la.qr(V[:,4:],mode="economic")

Vk = V[:,0:5]

print(np.linalg.norm(Vk.T @ Vk - np.eye(Vk.shape[1])))
