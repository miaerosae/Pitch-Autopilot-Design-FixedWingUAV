import numpy as np
import math as m
from spec_data import *
from aero_data import *
from mass_dist import mass_dist
from aero_model import aero_model

def get_rho(altitude):
        pressure = 101325 * (1 - 2.25569e-5 * altitude)**5.25616
        temperature = 288.14 - 0.00649 * altitude
        return pressure / (287*temperature)

def quat2dcm(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    row1 = np.array([q3**2+q0**2-q1**2-q2**2, 2*(q0*q1+q2*q3), 2*(q0*q2-q1*q3)])
    row2 = np.array([2*(q0*q1-q2*q3), q3**2-q0**2+q1**2-q2**2, 2*(q1*q2+q0*q3)])
    row3 = np.array([2*(q0*q2+q1*q3), 2*(q1*q2-q0*q3), q3**2-q0**2-q1**2+q2**2])
    C = np.stack([row1, row2, row3])
    return C

def state_eq(x, u, eta):
    x = np.ravel(x)
    u = np.ravel(u)

    delT, dela, dele, delr = u

    U, V, W, P, Q, R, q0, q1, q2, q3, pN, pE, pD = x
    h = -pD

    J, p_cg = mass_dist(eta)
    x_cg = p_cg[0]
    z_cg = p_cg[2]

    v = np.vstack([U, V, W])
    omega = np.vstack([P, Q, R])
    q = np.vstack([q0, q1, q2, q3])

    # env
    VT = np.linalg.norm(v)
    alpha = np.arctan2(W,U)
    bet = np.arcsin(V/VT)

    rho = get_rho(h)
    qbar = 0.5*rho*VT**2

    # aerodynamic force and moment
    CX, CY, CZ, Cl, Cm, Cn, CL, CD = aero_model(alpha, bet, VT, h, eta, P, Q, R, dela, dele, delr)
    XA = qbar*CX*S
    YA = qbar*CY*S
    ZA = qbar*CZ*S
    
    lA = qbar*S*b*Cl + z_cg*YA
    mA = qbar*S*cbar*Cm + x_cg*ZA - z_cg*YA
    nA = qbar*S*b*Cn - x_cg*YA

    FA = np.vstack([XA, YA, ZA])
    MA = np.vstack([lA, mA, nA])

    
    # thrust force and moment

    C_prop = 1
    S_prop = 0.2027
    K_motor = 80
    
    T = 0.5*rho*S_prop*C_prop*((K_motor*delT)**2 - VT**2)

    XT = T; YT = 0; ZT = 0;
    lT = 0; mT = 0; nT = 0;
   
    FT = np.vstack([XT, YT, ZT])
    MT = np.vstack([lT, mT, nT])

    # gravitational force and moment in body coordinate
    Fg = quat2dcm(q).dot(np.vstack([0, 0, mass*g]))
    Mg = np.vstack([0, 0, 0])

    # body-axis force and moment
    F = FA + FT + Fg
    M = MA + MT + Mg

    # state-equation
    q_foreq = np.vstack([q1, q2, q3])

    v_dot = F/mass - np.cross(omega, v, axisa=0, axisb=0).T
    omega_dot = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axisa=0, axisb=0).T)
    q_dot = np.zeros((4,1))
    q_dot[0,0] =  0.5 * (-np.dot(np.ravel(omega), np.ravel(q_foreq)))
    q_dot[1::,:] = 0.5 * (omega*q0 - np.cross(omega, q_foreq, axisa=0, axisb=0).T)
    p_dot = quat2dcm(q).T.dot(v)

    return np.vstack([v_dot, omega_dot, q_dot, p_dot])

