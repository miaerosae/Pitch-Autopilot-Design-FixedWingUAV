import numpy as np
import math as m

from spec_data import *
from mass_dist import mass_dist
from state_eq import get_rho
from aero_model import aero_model

def Dyn_nonlinear(t,X):
    VT      = X[0,0]
    gamma   = X[1,0]
    h       = X[2,0]
    alpha   = X[3,0]
    Q       = X[4,0]

    # coefficients
    eta     = np.vstack([0,0])
    J, *_   = mass_dist(eta)
    rho     = ger_rho(h)

    # lateral coefficients
    V = 0; P = 0; R = 0; dela = 0; delr = 0;
    bet = m.asin(V/VT)

    # aerodynamic force and moment
    _, _, _, _, Cm, _, CL, CD = aero_model(alpha, bet, VT, h, eta, P, Q, R, dela, dele, delr)
    qbar    = 0.5*rho*VT**2
    L       = qbar*CL*S
    D       = qbar*CD*S
    M       = qbar*Cm*S*cbar

    # Thrust
    S_prop  = 0.2027
    C_prop  = 1.0
    K_motor = 80
    T       = 0.5*rho*S_prop*((K_motor*delT)**2 - VT**2)

    # Dynamics
    VT_dot  = (T*m.cos(alpha) - D)/m - g*m.sin(gamma)
    gamma_dot   = (L + T*m.sin(alpha))/m/VT - g*m.cos(gamma)/VT
    h_dot = VT*m.sin(gamma)
    alpha_dot = Q - gamma_dot
    Q_dot = M/J[1,1]

    return np.vstack([VT_dot, gamma_dot, h-dot, alpha_dot, Q_dot])


def Dyn_LQR(t,X, ic, K):
    d2r = m.pi/180

    VT, gamma, h, alpha, Q = X

    VT0, gamma0, h0, alpha0, Q0, delT0, dele0 = ic

    # coefficients
    eta     = np.vstack([0,0])
    J, *_   = mass_dist(eta)
    rho     = get_rho(h)

    # lateral coefficients
    V = 0.0; P = 0.0; R = 0.0; dela = 0.0; delr = 0.0;
    bet = np.arcsin(V/VT)

    # gain K
    xd  = np.vstack([VT-VT0, gamma-gamma0, h-h0, alpha-alpha0, Q-Q0])
    u   = -K.dot(xd)

    dele = u[0,0] + dele0
    delT = u[1,0] + delT0

    # constraints in elevator angle
    if dele >= 10*d2r:
        dele = 10*d2r
    elif dele <= -10*d2r:
        dele = -10*d2r
    
    # constraints in thrust
    if delT >= 1:
        delT = 1
    elif delT <= 0:
        delT = 0

    # aerodynamic force and moment
    _, _, _, _, Cm, _, CL, CD = aero_model(alpha, bet, VT, h, eta, P, Q, R, dela, dele, delr)
    qbar    = 0.5*rho*VT**2
    Cm      = np.asscalar(Cm)
    CL      = np.asscalar(CL)
    CD      = np.asscalar(CD)
    L       = qbar*CL*S
    D       = qbar*CD*S
    M       = qbar*Cm*S*cbar

    # Thrust
    S_prop  = 0.2027
    C_prop  = 1.0
    K_motor = 80
    T       = 0.5*rho*S_prop*((K_motor*delT)**2 - VT**2)

    # Dynamics
    VT_dot  = (T*np.cos(alpha) - D)/mass - g*np.sin(gamma)
    gamma_dot   = (L + T*np.sin(alpha))/mass/VT - g*np.cos(gamma)/VT
    h_dot = VT*np.sin(gamma)
    alpha_dot = Q - gamma_dot
    Q_dot = M/J[1,1]

    return np.array([VT_dot, gamma_dot, h_dot, alpha_dot, Q_dot])
