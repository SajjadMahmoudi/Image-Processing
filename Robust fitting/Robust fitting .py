import pylab as plt
import numpy as np
from scipy.linalg import inv, eigh, solve


def _find_max_eigval(S):
    a = S[0,0]
    b = S[0,1]
    c = S[0,2]
    d = S[1,1]
    e = S[1,2]
    f = S[2,2]

    _a = -4
    _b = 4 * (c - d)
    _c = a * f - 4 * b * e + 4 * c * d - c * c
    _d = a * d * f - b * b * f - a * e * e + 2 * b * c * e - c  * c * d

    x2, x1, x0 = sorted(np.roots([_a, _b, _c, _d] ))
    return x0

def _find_max_eigvec(S):
    l = _find_max_eigval(S)
 
    a11 = S[0,0]
    a12 = S[0,1]
    a13 = S[0,2]
    a22 = S[1,1]
    a23 = S[1,2]
 
    u = np.array([
        a12 * a23 - (a13  - 2*l) * (a22 + l),
        a12 * (a13  - 2*l) - a23 * a11,
        a11 * (a22 + l) - a12 * a12
    ])
 
    c = 4 * u[0] * u[2] - u[1] * u[1]
 
    return l, u/np.sqrt(c)

def fit_ellipse(X):
    x = X[:,0]
    y = X[:,1]
 
    # building the design matrix
    D = np.vstack([ x*x, x*y, y*y, x, y, np.ones(X.shape[0])]).T
    S = np.dot(D.T, D)
 
    S11 = S[:3][:,:3]
    S12 = S[:3][:,3:]
    S22 = S[3:][:,3:]
 
    S22_inv = inv(S22)
    S22_inv_S21 = np.dot(inv(S22), S12.T)
 
    Sc =  S11 - np.dot(S12, S22_inv_S21)
    l, a = _find_max_eigvec(Sc)
 
    b = - np.dot(S22_inv_S21, a)
 
    return np.hstack([a,b])

def create_ellipse(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
 
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    X = np.vstack([ np.cos(angles) * r[0], np.sin(angles) * r[1]]).T
    return np.dot(X,R.T) + xc
 
def get_parameters(x):
    a,b,c,d,e,f = x
 
    A = np.array([
        [ a, b/2 ],
        [b/2, c  ]
    ])    
 
    B = np.array([d,e])    
 
    w,u = eigh(A)
 
    Xc = solve(-2*A,B)
    r2 = -0.5 * np.inner(Xc,B) - f
 
    rr2 = r2 / w
 
    alpha = np.arccos(u[0,0])
    if alpha > np.pi/2:
        alpha = alpha - np.pi
 
    alpha *= np.sign(u[0,1])
 
    return tuple(np.sqrt(rr2)), tuple(Xc), alpha

r = (1.0, 0.5)
xc = (0,1.5)
alpha = np.pi/4

#noise 
X = create_ellipse(r,xc,alpha)
Xn = create_ellipse(r,xc,alpha, angle_range=(-np.pi/1, np.pi/1), n=300)
#gaussian noise
Xn += 0.2 * min(r) * np.random.randn(*Xn.shape)
 
a = fit_ellipse(Xn)
rf, xcf, alpha_f = get_parameters(a)
 

Xf = create_ellipse(rf,xcf,alpha_f)
 
 
plt.plot(*Xn.T, '.', color='yellow')
plt.plot(*X.T, label="source ellipse", color='blue')
plt.plot(*Xf.T, label="guessed ellipse", color='red')
plt.legend(loc="lower right")
plt.show()


