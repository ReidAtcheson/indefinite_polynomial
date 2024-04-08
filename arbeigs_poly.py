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
m = 1000

eps=1/100

def clustered_eigs(outer,inner,rng=rng,mineig=None):
    centers = list(rng.uniform(-1,1,size=outer))
    if mineig:
        centers = centers + [mineig]

    eigs=[]
    for center in centers:
        eigs += list(rng.normal(center,0.2,size=inner))
    return np.array(eigs)

#eigs = rng.uniform(-1,1,size=m)
#eigs = np.array(list(rng.uniform(-1,-eps,size=m//2)) + list(rng.uniform(eps,1,size=m//2)))
eigs = clustered_eigs(10,10)




def findorder(eigs,rho=0.5,maxorder=1000):
    maxorder = min(maxorder,eigs.size)
    for order in range(3,maxorder):
        xs,p,tau = poly.arb_eigs_poly_l2(order,eigs)
        if tau<rho:
            return order
    return 0

minorder=findorder(eigs)
print(min(abs(eigs)),1/min(abs(eigs)),minorder)

plt.scatter(eigs,np.zeros_like(eigs))
plt.savefig("eigs.svg")


#plt.plot(xs,p)
#plt.savefig("arbeigs.svg")
