import math as m

g = 9.80665 #[m/s ^2]

# mass and geometry property

mass = 10  #  mass [kg]
S = 0.84   # reference area (nominal planform area) [m^2]
cbar = 0.288   # longitudinal reference length (nominal mean aerodynamic chord) [m]
b = 3   # lateral reference length (nominal span) [m]
bmin = 3   # [m]
bmax = 4.446   # [m]

# control surface limit

d2r = m.pi/180
dela_ll = -.5# aileron lower limit [rad]
dela_ul = .5  # aileron upper limit [rad]
dele_ll = -10*d2r  # elevator lower limit [rad]
dele_ul = 10*d2r  # elevator upper limit [rad]
delr_ll = -.5  # rudder lower limit [rad]
delr_ul = .5  # rudder upper limit [rad]

# thruster

Tmax = 50  # maximum thrust [N]
zeta = 1 
omega_n = 20  # s^-1
