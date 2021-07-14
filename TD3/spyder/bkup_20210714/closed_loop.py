# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:27:33 2021

@author: kranthi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# process model
Kp = 2.0
taup = 5.0
def process(y,t,u,Kp,taup):
    # Kp = process gain
    # taup = process time constant
    dydt = -y/taup + Kp/taup * u
    return dydt

# specify number of steps
ns = 325
# define time points
t = np.linspace(0,ns/10,ns+1)
delta_t = t[1]-t[0]

# storage for recording values
op = np.zeros(ns+1)  # controller output
pv = np.zeros(ns+1)  # process variable
e = np.zeros(ns+1)   # error
ie = np.zeros(ns+1)  # integral of the error
dpv = np.zeros(ns+1) # derivative of the pv
P = np.zeros(ns+1)   # proportional
I = np.zeros(ns+1)   # integral
D = np.zeros(ns+1)   # derivative
sp = np.zeros(ns+1)  # set point
sp[25:] = 48

# PID (starting point)
Kc = 1.0/Kp
tauI = taup
tauD = 0.0

# PID (tuning)
Kc = 0.5 #Kc * 2
tauI = 5 #tauI / 2
tauD = 0.1 #1.0

# Upper and Lower limits on OP
op_hi = 100.0
op_lo = 0.0

# loop through time steps    
for i in range(0,ns):
    e[i] = sp[i] - pv[i]
    if i >= 1:  # calculate starting on second cycle
        dpv[i] = (pv[i]-pv[i-1])/delta_t
        ie[i] = ie[i-1] + e[i] * delta_t
    P[i] = Kc * e[i]
    I[i] = Kc/tauI * ie[i]
    D[i] = - Kc * tauD * dpv[i]
    op[i] = op[0] + P[i] + I[i] + D[i]
    if op[i] > op_hi:  # check upper limit
        op[i] = op_hi
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    if op[i] < op_lo:  # check lower limit
        op[i] = op_lo
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    #print("op:{0},err:{1},ie:{2},dpv:{3},KP:{4},Ti:{5},Td:{6}"\
    #      .format(op[i],e[i],ie[i],dpv[i],P[i],I[i],D[i]))
    y = odeint(process,pv[i],[0,delta_t],args=(op[i],Kp,taup))
    pv[i+1] = y[-1]
op[ns] = op[ns-1]
ie[ns] = ie[ns-1]
P[ns] = P[ns-1]
I[ns] = I[ns-1]
D[ns] = D[ns-1]

# plot results
plt.figure(1)

plt.plot(t,sp,'k-',linewidth=2)
plt.plot(t,pv,'b--',linewidth=3)
plt.legend(['Set Point (SP)','Process Variable (PV)'],loc='best')
plt.ylabel('Process')
plt.ylim([-0.1,60])
plt.xlabel('Time')
plt.show()