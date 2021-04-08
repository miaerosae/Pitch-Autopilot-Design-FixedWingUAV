import numpy as np
import math as m
from scipy.interpolate import interpn

from aero_data import *

def aero_model(alpha, bet, VT, h, eta, P, Q, R, dela, dele, delr):
    eta1 = eta[0,0]
    eta2 = eta[1,0]
    
    # trimming
    alp_trm  = min(max(min(alp_grd),alpha),max(alp_grd)) 
    dele_trm = min(max(min(dele_grd),dele),max(dele_grd)) 
    eta1_trm = min(max(min(eta1_grd),eta1),max(eta1_grd)) 
    eta2_trm = min(max(min(eta2_grd),eta2),max(eta2_grd)) 

    # longitudinal coefficient interpolation
    CD = interpn((alp_grd,dele_grd,eta1_grd,eta2_grd), CD_grd, (alp_trm,dele_trm,eta1_trm,eta2_trm)) 
    CL = interpn((alp_grd,dele_grd,eta1_grd,eta2_grd), CL_grd, (alp_trm,dele_trm,eta1_trm,eta2_trm)) 
    Cm = interpn((alp_grd,dele_grd,eta1_grd,eta2_grd), Cm_grd, (alp_trm,dele_trm,eta1_trm,eta2_trm)) 
             
    # lateral coefficient
    CC = 0 
    Cl = 0 
    Cn = 0 

    # coordinate change from wind to body
    CX = m.cos(alpha)*m.cos(bet)*(-CD) - m.cos(alpha)*m.sin(bet)*(-CC) - m.sin(alpha)*(-CL) 
    CY =              m.sin(bet)*(-CD)              + m.cos(bet)*(-CC)            + 0*(-CL) 
    CZ = m.cos(bet)*m.sin(alpha)*(-CD) - m.sin(alpha)*m.sin(bet)*(-CC) + m.cos(alpha)*(-CL) 

    return CX, CY, CZ, Cl, Cm, Cn, CL, CD
