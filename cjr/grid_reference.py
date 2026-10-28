"""Référence numérique Hawkes P=1, alpha=0 : générateur sur grille + Howard.

Décroissance par schéma amont, sauts d'une unité exacts lorsque 1/h est entier.
La mémoire est plafonnée à L : approximation contrôlée par raffinement et extension
séparés. Ce n'est pas une solution exacte du Hawkes non borné.
"""
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import bicgstab, LinearOperator


def solve(p,L=8.,h=.25,policy=None,tol=1e-9):
    if p.C!=1 or p.eps_plus or p.eps_minus:
        raise ValueError('Référence réservée à P=1, alpha=0')
    step=int(round(1/h))
    if abs(step*h-1)>1e-10: raise ValueError('1/h doit être entier')
    m=int(round(L/h))+1
    iq,ia,ib=np.indices((2*p.K+1,m,m))
    q=(iq-p.K).ravel(); ea=(ia*h).ravel(); eb=(ib*h).ravel()
    n=len(q); ids=np.arange(n)
    def index(qi,ai,bi): return ((qi*m+ai)*m+bi).ravel()
    current=ids
    decay_targets=[index(iq,np.maximum(ia-1,0),ib),index(iq,ia,np.maximum(ib-1,0))]
    decay_rates=[p.betas[0]*ea/h,p.betas[0]*eb/h]
    nofill=[index(iq,np.minimum(ia+step,m-1),ib),index(iq,ia,np.minimum(ib+step,m-1))]
    filled=[index(np.maximum(iq-1,0),np.minimum(ia+step,m-1),ib),
            index(np.minimum(iq+1,2*p.K),ia,np.minimum(ib+step,m-1))]
    E=np.column_stack([ea,eb])
    lam=np.asarray(p.mu)[None]+E@np.asarray(p.a)[:,0,:].T
    allowed=np.column_stack([q>-p.K,q<p.K])
    V=np.zeros(n)
    given=policy(q,E) if policy is not None else None
    for it in range(100):
        if given is None:
            diff=np.column_stack([V[filled[s]]-V[nofill[s]] for s in (0,1)])
            d=np.maximum(1/p.kappa+p.fee-diff,0)
        else: d=given
        d=np.where(allowed,d,np.inf)
        prob=p.A*np.exp(-p.kappa*d)
        reward=-p.psi(q.astype(float))+np.sum(lam*prob*np.where(np.isfinite(d),d-p.fee,0),axis=1)
        targets=list(decay_targets); rates=list(decay_rates)
        for s in (0,1):
            targets.extend([nofill[s],filled[s]])
            rates.extend([lam[:,s]*(1-prob[:,s]),lam[:,s]*prob[:,s]])
        diag=sum(rates)
        Q=coo_matrix((np.concatenate(rates+[-diag]),
                      (np.tile(current,len(rates)+1),np.concatenate(targets+[current]))),shape=(n,n)).tocsr()
        matrix=p.r*eye(n,format='csr')-Q
        diagonal=matrix.diagonal()
        preconditioner=LinearOperator((n,n),matvec=lambda x:x/diagonal)
        new,info=bicgstab(matrix,reward,x0=V,rtol=1e-11,atol=1e-12,maxiter=20000,M=preconditioner)
        if info != 0 or np.max(np.abs(matrix@new-reward))>1e-7:
            raise RuntimeError(f'Système de politique grille non convergé : {info}')
        err=float(np.max(np.abs(new-V)))
        V=new
        if given is not None or err<tol:
            return dict(V=V.reshape(2*p.K+1,m,m),d=d.reshape(2*p.K+1,m,m,2),
                        q=q,E=E,L=L,h=h,iterations=it+1)
    raise RuntimeError('Howard Hawkes non convergé')
