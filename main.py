import numpy as np
import math as m
from scipy import integrate
import matplotlib.pyplot as plt
from sympy import *

from spec_data import *
from mass_dist import mass_dist
from trim import trim_calc
from Dyn import Dyn_LQR
from state_eq import get_rho
from fym.core import BaseEnv, BaseSystem
from fym.agents import LQR
import fym.logging

## data
d2r = m.pi/180
r2d = 180/m.pi

# specification data
e       = 0.9       # Osward efficiency factor
AR      = (b**2)/S  # wing aspect ratio

## Trim
h0      = 300               # initial altitude [m]
VT0     = 20                # initial true airspeed [m/s]
eta0    = np.vstack([0, 0]) # initial morphing parameter [span; sweep]

alp_guess   = 0
delT_guess  = 0
dele_guess  = 0
z_guess     = np.array([alp_guess, delT_guess, dele_guess])

x_t, u_t, alpha_t, Err = trim_calc(z_guess, VT0, h0, eta0)

gamma0  = 0
alpha0  = alpha_t
Q0      = x_t[4,0]
delT0   = u_t[0,0]
dele0   = u_t[2,0]

ic = np.array([VT0, gamma0, h0, alpha0, Q0, delT0, dele0])

## for VT tracking
#VT0_2   = 25
#VT0_3   = 15
#x_t2, u_t2, alpha_t2, Err2 = trim_calc(z_guess, VT0_2, h0, eta0)
#x_t3, u_t3, alpha_t3, Err3 = trim_calc(z_guess, VT0_3, h0, eta0)
#delT0_2 = u_t2[0,0]; dele0_2 = u_t2[2,0]; Q0_2 = x_t2[4,0];
#delT0_3 = u_t3[0,0]; dele0_3 = u_t3[2,0]; Q0_3 = x_t3[4,0];
#
#ic1 = np.vstack([VT0, gamma0, h0, alpha0, Q0])
#ic2 = np.vstack([VT0_2, gamma0, h0, alpha0_2, Q0_2])
#ic3 = np.vstack([VT0_3, gamma0, h0, alpha0_3, Q0_3])


J, *_ = mass_dist(eta0)

# declare symbol
VT, gamma, h, alpha, Q = symbols('VT gamma h alpha Q')
delT, dele = symbols('delT dele')

# Aerodynamic force and moment
rho     = get_rho(h0)
qbar    = 0.5*rho*VT**2

CL0         = 0.13788
CL_alpha    = 0.10055*r2d
CD0         = 0.041703
Cm0         = 0.19968
Cm_alpha    = -0.066327*r2d
Cm_dele     = -0.13254*r2d

CL = CL0 + CL_alpha*alpha
CD = CD0 + ((CL0 + CL_alpha*alpha)**2)/m.pi/e/AR
Cm = Cm0 + Cm_alpha*alpha + Cm_dele*dele

L = qbar*CL*S
D = qbar*CD*S
Myy = qbar*Cm*S*cbar

# Thrust
C_prop = 1
S_prop = 0.2027
K_motor = 80

T = 0.5*rho*S_prop*C_prop*((K_motor*delT)**2-VT**2)

# Linearization of A, B matrix near the trim point
VT_d = (T*cos(alpha)-D)/mass -g*sin(gamma) 
gamma_d = (L+T*sin(alpha))/mass/VT - g*cos(gamma)/VT 
h_d = VT*sin(gamma) 
alpha_d = Q - gamma_d 
Q_d = Myy/J[1,1] 

A = np.vstack([[diff(VT_d,VT), diff(VT_d,gamma), diff(VT_d,h), diff(VT_d,alpha), diff(VT_d,Q)], 
            [diff(gamma_d,VT), diff(gamma_d,gamma), diff(gamma_d,h), diff(gamma_d,alpha), diff(gamma_d,Q)], 
            [diff(h_d,VT), diff(h_d,gamma), diff(h_d,h), diff(h_d,alpha), diff(h_d,Q)], 
            [diff(alpha_d,VT), diff(alpha_d,gamma), diff(alpha_d,h), diff(alpha_d,alpha), diff(alpha_d,Q)], 
            [diff(Q_d,VT), diff(Q_d,gamma), diff(Q_d,h), diff(Q_d,alpha), diff(Q_d,Q)]]) 
    
B = np.vstack([[diff(VT_d,dele), diff(VT_d,delT)], 
             [diff(gamma_d,dele), diff(gamma_d,delT)], 
             [diff(h_d,dele), diff(h_d,delT)], 
             [diff(alpha_d,dele), diff(alpha_d,delT)], 
             [diff(Q_d,dele), diff(Q_d,delT)]]) 

A_trim = np.vectorize(lambda x: x.subs({VT:VT0, gamma:gamma0, h:h0, alpha:alpha0, Q:Q0, dele:dele0, delT:delT0}), otypes=[np.float32])(A)
B_trim = np.vectorize(lambda x: x.subs({VT:VT0, gamma:gamma0, h:h0, alpha:alpha0, Q:Q0, dele:dele0, delT:delT0}), otypes=[np.float32])(B)

# simulate linear dynamic system using fym, LQR
# simulate linearized equation
class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.05, max_t=50)
        
        # for level flight
        self.x = BaseSystem(np.vstack([VT0+5, gamma0, h0+30, alpha0, Q0]))

        ## for VT tracking
#        t = self.clock.get()
#        if t < 10:
#            self.x = BaseSystem(np.vstack
        self.A = A_trim
        self.B = B_trim
        
        Q = np.diag([0.003, 0.4, 0.002, 0.4, 1.7])
        R = 1.7*np.identity(2)
        self.K, *_ = LQR.clqr(self.A, self.B, Q, R)

    def reset(self):
        super().reset()

    def step(self):
        t = self.clock.get()
        x = self.x.state
        u = -self.K.dot(x)

        *_, done = self.update()
        return t, x, u, done

    def set_dot(self, t):
        x = self.x.state - np.vstack([VT0, gamma0, h0, alpha0, Q0])
        u = -self.K.dot(x)
        self.x.dot = self.A.dot(x) + self.B.dot(u)


def run():
    env = Env()
    x = env.reset()
    env.logger = fym.logging.Logger(path='data.h5')

    while True:
        env.render()
        t, x, u, done = env.step()
        env.logger.record(t=t, x=x, u=u)
        
        if done:
            break
    return env.K

    env.close()

def plot_var():
    data    = fym.logging.load('data.h5')
    fig1    = plt.figure()
    ax1     = fig1.add_subplot(5,1,1)
    ax2     = fig1.add_subplot(5,1,2)
    ax3     = fig1.add_subplot(5,1,3)
    ax4     = fig1.add_subplot(5,1,4)
    ax5     = fig1.add_subplot(5,1,5)
    
    ax1.plot(data['t'], data['x'].squeeze()[:,0])
    ax2.plot(data['t'], data['x'].squeeze()[:,1]*r2d)
    ax3.plot(data['t'], data['x'].squeeze()[:,2])
    ax4.plot(data['t'], data['x'].squeeze()[:,3]*r2d)
    ax5.plot(data['t'], data['x'].squeeze()[:,4])
    ax1.set_xlabel('t [sec]'); ax2.set_xlabel('t [sec]'); ax3.set_xlabel('t [sec]');
    ax4.set_xlabel('t [sec]'); ax5.set_xlabel('t [sec]');
    ax1.set_ylabel('VT [m/s]'); ax2.set_ylabel('gamma [deg]'); ax3.set_ylabel('h [m]');
    ax4.set_ylabel('alpha [deg]'); ax5.set_ylabel('Q')
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True); ax5.grid(True);

    plt.suptitle('Level flight : LQR Simulation for linearized system', fontsize=15)
#    plt.show()
    plt.savefig('LQR_Level_flight_linearsystem.png', dpi=200)

K = run()
plot_var()
# simulate nonlinear equation with K (calculated by linearized A, B), using RK4 method
step = 0.05
t = np.linspace(0,50,1000)
x = np.zeros((1000, 5))
x[0,:] = np.array([VT0+5, gamma0, h0+30, alpha0, Q0])

k1 = np.zeros((5,))
k2 = np.zeros((5,))
k3 = np.zeros((5,))
k4 = np.zeros((5,))

for i in range(0, 999):
    k1 = Dyn_LQR(t[i], x[i,:], ic, K)
    k2 = Dyn_LQR(t[i]+step/2, x[i,:]+step*k1/2, ic, K)
    k3 = Dyn_LQR(t[i]+step/2, x[i,:]+step*k2/2, ic, K)
    k4 = Dyn_LQR(t[i]+step, x[i,:]+step*k3, ic, K)
    x[i+1,:] = x[i,:] + (k1/6 + k2/3 + k3/3 + k4/6)*step

def plot_var_RK4():
    fig2 = plt.figure()
    ax1     = fig2.add_subplot(5,1,1)
    ax2     = fig2.add_subplot(5,1,2)
    ax3     = fig2.add_subplot(5,1,3)
    ax4     = fig2.add_subplot(5,1,4)
    ax5     = fig2.add_subplot(5,1,5)
    
    ax1.plot(t, x[:,0])
    ax2.plot(t, x[:,1])
    ax3.plot(t, x[:,2])
    ax4.plot(t, x[:,3])
    ax5.plot(t, x[:,4])
    ax1.set_xlabel('t [sec]'); ax2.set_xlabel('t [sec]'); ax3.set_xlabel('t [sec]');
    ax4.set_xlabel('t [sec]'); ax5.set_xlabel('t [sec]');
    ax1.set_ylabel('VT [m/s]'); ax2.set_ylabel('gamma [deg]'); ax3.set_ylabel('h [m]');
    ax4.set_ylabel('alpha [deg]'); ax5.set_ylabel('Q')

    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True); ax5.grid(True);
    plt.suptitle('Level flight : LQR Simulation for nonlinear system', fontsize=15);
#    plt.show()
    plt.savefig('LQR_Level_flight_nonlinearsystem.png', dpi=200)


plot_var_RK4()
