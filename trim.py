import numpy as np
import math as m
from scipy.spatial.transform import Rotation # to make Matlab's angle2quat
from scipy.optimize import minimize, Bounds # to make minimized cost function
from state_eq import state_eq
from spec_data import *

def angle2quat(yaw, pitch, roll):
    r2d = 180/m.pi
    rot = Rotation.from_euler('xyz', [roll*r2d, pitch*r2d, yaw*r2d], degrees=True)
    quat = rot.as_quat()
    return quat

def trim_cost(z, VT, h, eta):
    alpha = z[0]
    
    U = VT*m.cos(alpha)
    V = 0
    W = VT*m.sin(alpha)

    P = 0
    Q = 0
    R = 0
    
    ph = 0
    th = alpha
    ps = 0
    q0, q1, q2, q3 = angle2quat(ph, th, ps)

    pN = 0
    pE = 0
    pD = -h

    delT = z[1]
    dela = 0
    dele = z[2]
    delr = 0

    x = np.vstack([U, V, W, P, Q, R, q0, q1, q2, q3, pN, pE, pD])
    u = np.vstack([delT, dela, dele, delr])
    dx = state_eq(x, u, eta) 
    weight = np.zeros((3,3), float)
    np.fill_diagonal(weight, [2, 1, 1000])
    dx_cost = np.vstack([dx[0,0], dx[2,0], dx[4,0]])
    cost = sum([dx_cost.T.dot(weight).dot(dx_cost)])

    # return cost
    return cost[0][0]


def trim_calc(z_guess, VT0, h0, eta0):
    d2r = m.pi/180
    z = z_guess
    bounds = Bounds([-10*d2r, 0, dele_ll],  [10*d2r, 1, dele_ul])
    # lb = np.vstack([-10*d2r, 0, dele_ll])
    # ub = np.vstack([10*d2r, 1, dele_ul])
    # bounds = Bounds(lb,ub)
    # Z = minimize(trim_cost(z_guess,VT=VT0,h=h0,eta=eta0), z_guess, method='trust-constr', bounds=bounds)
    # Z = minimize(trim_cost(z_guess,VT=VT0,h=h0,eta=eta0), z_guess, method='trust-constr', bounds=bounds)
    res = minimize(lambda z: trim_cost(z, VT=VT0, h=h0, eta=eta0), z_guess, method='trust-constr', bounds=bounds)
    Z = res.x
    Err = trim_cost(Z, VT0, h0, eta0)
    
    alpha_t = Z[0]
    U_t = VT0*m.cos(alpha_t)
    V_t = 0
    W_t = VT0*m.sin(alpha_t)

    P_t = 0
    Q_t = 0
    R_t = 0

    ph_t = 0
    th_t = alpha_t
    ps_t = 0
    q0_t, q1_t, q2_t, q3_t = angle2quat(ph_t, th_t, ps_t)

    pN_t = 0
    pE_t = 0
    pD_t = -h0
    
    delT_t = Z[1]
    dela_t = 0
    dele_t = Z[2]
    delr_t = 0

    x_t = np.vstack([U_t, V_t, W_t, P_t, Q_t, R_t, q0_t, q1_t, q2_t, q3_t, pN_t, pE_t, pD_t])
    u_t = np.vstack([delT_t, dela_t, dele_t, delr_t])

    return x_t, u_t, alpha_t, Err


if __name__ == "__main__":
    z_guess = np.random.rand(3)
    VT0 = 20
    h0 = 300
    eta0 = np.vstack([0, 0])
    res = trim_calc(z_guess, VT0, h0, eta0)
    print(res)
