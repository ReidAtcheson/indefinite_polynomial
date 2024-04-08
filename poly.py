import numpy as np
import cvxpy as cp
import scipy.linalg as la
from numpy.polynomial.legendre import legvander


def poly_basis(xs,k):
    m=xs.size
    V=np.zeros((m,k))
    V[:,0]=1.0/np.sqrt(m)
    for i in range(1,k):
        w = xs*V[:,i-1]
        h = V[:,0:i].T @ w
        f = w - V[:,0:i] @ h
        s = V[:,0:i].T @ f
        h = h + s
        w = w - V[:,0:i] @ h
        w = w/np.linalg.norm(w)
        V[:,i]=w
    return V


def arb_eigs_poly_l2(order,eigs):
    k=order+1
    assert(0 not in list(eigs))
    xs=np.array([0]+list(eigs))
    m=xs.size

    #Construct an orthonormal polynomial basis
    #on the discrete set `xs`
    V=np.zeros((m,k))
    V[:,0]=np.ones(m)/np.sqrt(m)
    for i in range(1,k):
        w = xs*V[:,i-1]

        h = V[:,0:i].T @ w
        f = w - V[:,0:i] @ h
        s = V[:,0:i].T @ f
        h = h + s

        w = w - V[:,0:i] @ h
        w = w/np.linalg.norm(w)
        V[:,i]=w


    e = np.zeros(m)
    e[0]=1
    p = V.T @ e
    Vp = V@p
    Vp = Vp/Vp[0]
    tau = max(abs(Vp[1:]))


    ids=np.argsort(xs)

    return xs[ids],(Vp)[ids],tau






def arb_eigs_poly(order,eigs):
    k=order+1
    assert(0 not in list(eigs))
    xs=np.array([0]+list(eigs))
    m=xs.size

    #Construct an orthonormal polynomial basis
    #on the discrete set `xs`
    V=np.zeros((m,k))
    V[:,0]=np.ones(m)/np.sqrt(m)
    for i in range(1,k):
        w = xs*V[:,i-1]

        h = V[:,0:i].T @ w
        f = w - V[:,0:i] @ h
        s = V[:,0:i].T @ f
        h = h + s

        w = w - V[:,0:i] @ h
        w = w/np.linalg.norm(w)
        V[:,i]=w
    p = cp.Variable(k)
    tau = cp.Variable(1)

    problem = cp.Problem(cp.Minimize(tau),
            [
                tau >= 0,
                V[1:,:] @ p <= tau,
                -(V[1:,:] @ p) <= tau,
                V[0,:] @ p == 1
                ])
    problem.solve(solver=cp.CLARABEL)
    #print(tau.value)
    #print(Vneg @ p.value)
    ids=np.argsort(xs)

    return xs[ids],(V@p.value)[ids],tau.value







#Constructs a polynomial P such that
#1. |P(x)| small as possible for x in [-1.0, -eps]
#2. P(0) = 1
#3. |P(x)| <= 1 for x in [0.0,1.0]
def indefinite_damping(order,eps,n=100,max_value=1.0):
    assert(eps>0)
    #Formulate this as a linear program:
    #min tau
    #s.t. 
    # -tau <= 0
    # Vneg*p - tau <= 0
    #-Vneg*p - tau <= 0
    #V0*p = 1
    # Vpos*p <= 1
    # -Vpos*p <= 1
    xneg = np.linspace(-1.0,-eps,n)
    xpos = np.linspace(eps,1.0,n)
    Vneg = legvander(xneg,order)
    Vpos = legvander(xpos,order)
    V0 = legvander([0.0],order)
    p = cp.Variable(order + 1)
    tau = cp.Variable(n)
    c = np.ones(n)
    Pneg = np.ones(n) - np.diag(xneg) @ (Vneg @ p)
    Ppos = np.ones(n) - np.diag(xpos) @ (Vpos @ p)
    problem = cp.Problem(cp.Minimize(c.T @ tau),
            [
                tau >= 0,
                Pneg <= tau,
                -Pneg <= tau,
                Ppos <= max_value,
                -Ppos <= max_value
                ]
            )
    problem.solve(solver=cp.CLARABEL)
    #print(tau.value)
    #print(Vneg @ p.value)
    return p.value

#Constructs a polynomial P such that
#1. |P(x)| small as possible for x in [-1.0, -eps] , [eps, 1.0]
#2. P(0) = 1
def symmetric_damping(order,eps,n=100,max_value=1.0):
    assert(eps>0)
    #Formulate this as a linear program:
    #min tau
    #s.t. 
    # -tau <= 0
    # Vneg*p - tau <= 0
    #-Vneg*p - tau <= 0
    #V0*p = 1
    # Vpos*p <= 1
    # -Vpos*p <= 1
    xneg = np.linspace(-1.0,-eps,n)
    xpos = np.linspace(eps,1.0,n)
    Vneg = legvander(xneg,order)
    Vpos = legvander(xpos,order)
    V0 = legvander([0.0],order)
    p = cp.Variable(order + 1)
    tau = cp.Variable(n)
    c = np.ones(n)
    Pneg = np.ones(n) - np.diag(xneg) @ (Vneg @ p)
    Ppos = np.ones(n) - np.diag(xpos) @ (Vpos @ p)
    problem = cp.Problem(cp.Minimize(c.T @ tau),
            [
                tau >= 0,
                Pneg <= tau,
                -Pneg <= tau,
                Ppos <= tau,
                -Ppos <= tau
                ]
            )
    problem.solve(solver=cp.CLARABEL)
    #print(tau.value)
    #print(Vneg @ p.value)
    return p.value



#Evaluate P(A) * x
# xn = x0 + s(A)r0
# rn = (I - As(A))r0
# P(A) = I - As(A)
# s(A) = inv(A)*(P(A)-I)
def poly_eval(A,r,p):
    order = p.size-1
    assert(p.size>2)
    out = np.zeros_like(r)

    P0 = r.copy()
    P1 = A @ r

    out += p[0] * P0
    out += p[1] * P1
    for n in range(1,order):
        Pn = ((2*n+1)/(n+1))*A @ P1 - (n/(n+1)) * P0
        P0 = P1
        P1 = Pn
        out += p[n+1]*Pn
    return out



def poly_iter(A,r,p,maxiter):
    b=r.copy()
    assert(maxiter>0)
    x=np.zeros_like(r)
    for n in range(maxiter):
        x = x + poly_eval(A,r,p)
        r = b - A@x
    return x




