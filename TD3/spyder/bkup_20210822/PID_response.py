import numpy as np
import matplotlib.pyplot as plt
import control.matlab as cm

# Plant Model
num= [2]
dnum = [9,6,1]
delay = 0
s = cm.tf('s')
tf_sys = cm.tf(num,dnum)

# PID Model
Kp = 2
Ti = 5
Td = 2
pid = Kp*(1+ 1/(Ti*s) + Td*s)

# Closed loop
opn_tf = cm.series(pid,tf_sys)
cls_tf = cm.feedback(opn_tf,1,-1)

# Step Input
t = np.linspace(0,30,301)
u = np.ones(len(t))
u[0] = 0

# Step response of closed loop t/f
y,t,_ = cm.lsim(cls_tf, u, t)

# Ploting the response
plt.figure()
plt.plot(t,u, label="Input")
plt.plot(t,y, label="Response")
plt.legend()
plt.title("Closed loop step response")
plt.show()