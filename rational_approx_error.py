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
for alpha in [-0.5,-0.4,-0.3,-0.2]:
    intervals = [(-1.0,-0.5),(-0.1,0.3),(0.6,1.0)]
    xs=make_intervals(intervals,m=m)
    #xs=np.linspace(-1,1,m)

    k=40
    V=poly.poly_basis(xs,k)
    #print(np.linalg.norm(V.T @ V - np.eye(k)))

    Q,R,p=la.qr(V.T,mode='economic',pivoting=True)


    #print(np.linalg.norm(Q@R - V[p,:].T))
    plt.scatter(xs[p[0:k]],np.zeros(k),s=4)
    for a,b in intervals:
        plt.axvspan(a,b,color='yellow',alpha=0.3)
    plt.savefig("rrqr_subset.svg")
