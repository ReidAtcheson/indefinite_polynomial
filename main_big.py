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

m = 10000
bands = [0,1,32]

#es = list(np.linspace(-0.9,-0.1,m//2)) + list(np.linspace(0.1,0.9,m//2)) 
#Q,_ = la.qr(rng.uniform(-1,1,size=(m,m)))
#A = Q @ np.diag(es) @ Q.T

A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A = 0.5*(A+A.T)
A = A / spla.norm(A,ord=1)
luA = spla.splu(A)
norminvA = spla.onenormest(spla.LinearOperator((m,m), matvec = luA.solve, rmatvec = lambda x : luA.solve(x,trans='T')))
print(f"log10(cond(A)) = {np.log10(norminvA)}")

xe = rng.uniform(-1,1,size=m)
b = A @ xe


#if isinstance(A,np.ndarray):
#    print(np.min(np.abs(la.eigvalsh(A))))
#    print(np.max(np.abs(la.eigvalsh(A))))
#else:
#    print(np.min(np.abs(la.eigvalsh(A.toarray()))))
#    print(np.max(np.abs(la.eigvalsh(A.toarray()))))


order=200
eps=1e-4
#p = poly.indefinite_damping(order,eps,n=1000,max_value=1.0)
p = poly.symmetric_damping(order,eps,n=1000,max_value=1.0)
def callback(A,b,label=""):
    res=[]
    it = 0
    def f(xk):
       nonlocal it
       r = b - A@xk
       res.append(np.linalg.norm( (xk - xe)/xe, ord=np.inf ))
       #res.append(np.linalg.norm(r))
       #plt.close()
       #plt.plot(es,Q.T @ r)
       #plt.savefig(f"plots/{label}_{str(it).zfill(3)}.svg")
       #it=it+1
       #plt.close()
       #res.append(np.linalg.norm(r))
    return res,f

maxiter = 10000
inner = 200
res0,f = callback(A,b,label="plain")
spla.minres(A,b,callback=f,maxiter=maxiter,tol=1e-13)

res1,f = callback(A,b,label="deflated")
x = np.zeros_like(b)
for i in range(maxiter//inner):
    r = b - A@x
    #x = x + poly.poly_eval(A,r,p)
    x = x + poly.poly_iter(A,r,p,20)
    x,_ = spla.minres(A,b,x0=x,maxiter=inner,tol=1e-13,callback=f)
    #f(x)

#res2,f = callback(A,b,label="deflated_par")
#x = np.zeros_like(b)
#for i in range(maxiter//inner):
#    r = b - A@x
#    y,_ = spla.minres(A,b,x0=x,maxiter=inner,tol=1e-13,callback=f)
#    x = x + poly.poly_eval(A,r,p)
#    V=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
#    AV = A@V
#    e,_,_,_=la.lstsq(AV,r)
#    x = x + V@e    
#    f(x)




#def precon(b):
#    inner_iter=1
#    x = np.zeros_like(b)
#    for i in range(inner_iter):
#        r = b - A@x
#        x = x + poly.poly_eval(A,r,p)
#    return x
#
#
#res2,f = callback(A,b,label="precon")
#spla.gmres(A,b,callback=f,M = spla.LinearOperator((m,m),matvec=precon),maxiter=maxiter,callback_type="x",restart=restart,tol=1e-13)
plt.semilogy(res0,label="minres")
plt.semilogy(res1,label="deflated minres")
#plt.semilogy(res2,label="deflated_par minres")
#plt.semilogy(res2,label="precon gmres")






plt.legend()
plt.savefig("res.svg")





