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

m = 100000
bands = [0,1,32]
#es = list(np.linspace(-0.9,-1e-5,m//2)) + list(np.linspace(1e-5,0.9,m//2)) 
#Q,_ = la.qr(rng.uniform(-1,1,size=(m,m)))
#A = Q @ np.diag(es) @ Q.T
A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A = 0.5*(A+A.T)
A = A / spla.norm(A,ord=np.inf)
#b = Q @ np.ones(m)
b = rng.uniform(-1,1,size=m)
#xe = la.solve(A,b)

def callback(A,b):
    res=[]
    def f(xk):
        r = b - A@xk
        res.append(np.linalg.norm(r))
    return res,f

maxiter=40000
inner=1000

res0,f0 = callback(A,b)
spla.minres(A,b,maxiter=maxiter,callback=f0,tol=1e-14)


res,f = callback(A,b)
xs=[]
x = np.zeros(m)
for _ in range(maxiter//inner):
    x,info = spla.minres(A,b,maxiter=inner,x0=x,callback=f,tol=1e-14)
    xs.append(x)
    V = np.concatenate([xi.reshape(-1,1) for xi in xs],axis=1)
    e,_,_,_ = la.lstsq(A@V,b)
    x = V @ e


plt.semilogy(res0,label="plain")
plt.semilogy(res,label="history restart")
plt.legend()
plt.savefig("res.svg")
