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

def make_intervals(intervals : list[tuple],m=100):
    values=[]
    for a,b in intervals:
        values.append(np.linspace(a,b,m))
    return np.concatenate(values)



m=300

it=0
for alpha in [-0.8,-0.7,-0.6,-0.5,-0.4,-0.3]:
    intervals = [(-1.0,alpha),(-0.2,-0.1),(0.1,0.2),(0.6,1.0)]
    xs=make_intervals(intervals,m=m)
    #xs=np.linspace(-1,1,m)

    k=20
    V=poly.poly_basis(xs,k)
    y = (1.0/xs)
    p = V @ (V.T @ y)

    for i in range(0,xs.size,m):
        ibeg=i
        iend=i+m
        plt.plot(xs[ibeg:iend],np.abs(y[ibeg:iend]-p[ibeg:iend]),color='blue')
    #plt.plot(xs,np.abs(y-p))
    for a,b in intervals:
        plt.axvspan(a,b,color='yellow',alpha=0.3)
    plt.ylim(0,3)
    plt.savefig(f"approx/p_{str(it).zfill(3)}.svg")
    plt.close()
    it=it+1
