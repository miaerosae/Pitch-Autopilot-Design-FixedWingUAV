import numpy as np
import math as m
from scipy import io

from spec_data import *

alp_grd    = np.linspace(-10, 20, 61)*m.pi/180
dele_grd   = np.array([dele_ll, 0, dele_ul])
eta1_grd    = np.array([0.0, 0.5, 1.0])
eta2_grd    = np.array([0.0, 0.5, 1.0])

mat_file = io.loadmat("aero_coeff.mat")
CL = mat_file['CL']
CD = mat_file['CD']
Cm = mat_file['CM']

CF_S    = np.vstack([ [0.840, 0.820, 0.810], 
                    [0.984, 0.962, 0.949], 
                    [1.129, 1.104, 1.087] ])  # reference planform area used in XFLR5
CF_cbar = np.vstack([ [0.288, 0.299, 0.351], 
                    [0.275, 0.286, 0.336], 
                    [0.265, 0.276, 0.325] ])  # reference mac used in XFLR5
CF_b    = np.vstack([ [3.000, 2.810, 2.354], 
                    [3.722, 3.490, 2.908], 
                    [4.446, 4.170, 3.462] ])  # reference span used in XFLR5

CD_f    = 0.4   # fuselage drag coefficient
S_f     = 0.084 # fuselage section area

CL_grd = np.transpose(CL, (3,2,0,1))
CD_grd = np.transpose(CD, (3,2,0,1)) + CD_f*S_f/CF_S[1,1]
Cm_grd = np.transpose(Cm, (3,2,0,1))

for i in range(2):
    for j in range(2):
        CL_grd[:,:,i,j] = CL_grd[:,:,i,j]/CF_S[1,1]*CF_S[i,j]
        CD_grd[:,:,i,j] = CD_grd[:,:,i,j]/CF_S[1,1]*CF_S[i,j]
        Cm_grd[:,:,i,j] = Cm_grd[:,:,i,j]/CF_S[1,1]*CF_S[i,j]/CF_cbar[1,1]*CF_cbar[i,j]


