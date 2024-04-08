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

m = 100
bands = [0,1,32]
es = list(np.linspace(-0.9,-0.1,m//2)) + list(np.linspace(0.1,0.9,m//2)) 

Q,_ = la.qr(rng.uniform(-1,1,size=(m,m)))
A = Q @ np.diag(es) @ Q.T
#A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
#A = 0.5*(A+A.T)
#A = A / spla.norm(A,ord=np.inf)
b = Q @ np.ones(m)
xe = la.solve(A,b)


if isinstance(A,np.ndarray):
    print(np.min(np.abs(la.eigvalsh(A))))
    print(np.max(np.abs(la.eigvalsh(A))))
else:
    print(np.min(np.abs(la.eigvalsh(A.toarray()))))
    print(np.max(np.abs(la.eigvalsh(A.toarray()))))


order=10
eps=1e-4
#p = poly.indefinite_damping(order,eps,n=1000,max_value=1.0)
p = poly.symmetric_damping(order,eps,n=1000,max_value=1.0)
def callback(A,b,label=""):
    res=[]
    it = 0
    def f(xk):
       nonlocal it
       r = b - A@xk
       #res.append(np.linalg.norm( (xk - xe)/xe, ord=np.inf ))
       res.append(np.linalg.norm(r))
       plt.close()
       plt.plot(es,Q.T @ r)
       plt.savefig(f"plots/{label}_{str(it).zfill(3)}.svg")
       it=it+1
       plt.close()
       #res.append(np.linalg.norm(r))
    return res,f

maxiter = 100
inner=10
res0,f = callback(A,b,label="plain")
spla.minres(A,b,callback=f,maxiter=maxiter,tol=1e-13)

res1,f = callback(A,b,label="deflated")
x = np.zeros_like(b)
for i in range(maxiter//inner):
    r = b - A@x
    #x = x + poly.poly_eval(A,r,p)
    x = x + poly.poly_iter(A,r,p,5)
    x,_ = spla.minres(A,b,x0=x,maxiter=inner,tol=1e-13,callback=f)
    #f(x)

res2,f = callback(A,b,label="poly_iter")
x = np.zeros_like(b)
for i in range(maxiter//inner):
    r = b - A@x
    #y,_ = spla.minres(A,b,x0=x,maxiter=inner,tol=1e-13,callback=f)
    x = x + poly.poly_eval(A,r,p)
    #V=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
    #AV = A@V
    #e,_,_,_=la.lstsq(AV,r)
    #x = x + V@e    
    f(x)




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
plt.semilogy(res2,label="poly iter")
#plt.semilogy(res2,label="deflated_par minres")
#plt.semilogy(res2,label="precon gmres")






plt.legend()
plt.savefig("res.svg")





#Plot residual polynomials
plt.close()
nsamples=1000
xs = np.linspace(-1.0,1.0,nsamples)
V = legvander(xs,order)
ys = 1.0 - np.diag(xs) @ V @ p
plt.plot(xs,ys)
plt.savefig("out.svg")
#Plot residuals from polynomial evaluation
plt.close()
D = sp.diags([xs],[0])
plt.plot(xs,1.0 - D @ poly.poly_eval(D,np.ones(nsamples),p))
plt.savefig("out2.svg")
plt.close()
#Plot residuals when applied to AA
e = Q @ np.ones(m)
x = poly.poly_eval(A,e,p)
#x,info = spla.gmres(A,e,maxiter=1)
r = Q.T @ (e - A@x)
plt.plot(es,r)
plt.savefig("spectral_res.svg")


# A = Q D Q^T, P(A) = Q P(D) Q^T
# P(D) @ e, 
# Q @ P(D) @ Q^T (Q @ e)

