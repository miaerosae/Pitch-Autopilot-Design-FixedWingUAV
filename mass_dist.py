import numpy as np

from aero_data import *
from scipy.interpolate import interpn, griddata, RegularGridInterpolator
from spec_data import *

def mass_dist(eta):
    eta1 = eta[0,0]
    eta2 = eta[1,0]

    eta1_trm = min(max(min(eta1_grd),eta1),max(eta1_grd)) 
    eta2_trm = min(max(min(eta2_grd),eta2),max(eta2_grd)) 

    # center of gravity data

    x_cg = -(np.vstack([ [183.22, 201.99, 216.61], [183.22, 206.27, 224.66], [183.22, 210.55, 232.71] ])+200-183.22)*0  # [mm]
    x_cg = x_cg/1000  # [m]

    # center of gravity interpolation

    p_ref = np.zeros((3,1)) 
    p_ref[0,0] = interpn((eta1_grd, eta2_grd), x_cg, (eta1_trm, eta2_trm))
    p_ref[1,0] = 0  # [m]
    p_ref[2,0] = 0  # [m]

    # moment of inertia data

    J_xx = 9323306930.82  # eta = 0 [g*mm^2]
    J_yy = np.array([ [96180388451.54, 96468774320.55, 97352033548.31], [96180388451.54, 96720172843.10, 98272328292.52], [96180388451.54, 97077342563.70, 99566216309.81] ])  # [g*mm^2]
    J_zz = 105244200037  # [g*mm^2]

    J_xy = -2622499.75  # eta = 0 [g*mm^2]
    J_xz = 56222833.68  # eta = 0 [g*mm^2]
    J_yz = 395245.59  # eta = 0 [g*mm^2]

    # moment of inertia interpolation

    J = np.zeros((3,3)) 
    J[0,0] = J_xx 
    J[0,1] = J_xy 
    J[0,2] = J_xz 
    J[1,0] = J_xy 
    J[1,1] = interpn((eta1_grd,eta2_grd) , J_yy, (eta1_trm,eta2_trm)) 
    J[1,2] = J_yz 
    J[2,0] = J_xz 
    J[2,1] = J_yz 
    J[2,2] = J_zz 

    J = J/(10^9)/103.47649*mass  # [kg*m^2]

    return J, p_ref
