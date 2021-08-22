import numpy as np
from numpy import (dtype, real, atleast_1d, atleast_2d, squeeze, asarray, zeros,
                   dot, transpose, ones, zeros_like, linspace, nan_to_num)
from scipy import integrate, interpolate, linalg
import control
from control.matlab import *
import matplotlib.pyplot as plt

def ProcessModel(num, dnum, delay, dt):
    # Plant Model
    s = tf('s')
    tf_sys = tf(num,dnum)
    # Converting Laplace domain to State space model
    ss_sys = control.tf2ss(tf_sys)
    # State, Input, Output, Disturbance matrices of state space model
    A, B, C, D = map(np.asarray, (ss_sys.A, ss_sys.B, ss_sys.C, ss_sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    x0 = zeros(n_states, ss_sys.A.dtype)
    # Building state space model
    M = np.vstack([np.hstack([A * dt, B * dt,
                                np.zeros((n_states, n_inputs))]),
                    np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                np.identity(n_inputs)]),
                    np.zeros((n_inputs, n_states + 2 * n_inputs))])
    expMT = linalg.expm(transpose(M))
    Ad = expMT[:n_states, :n_states]
    Bd1 = expMT[n_states+n_inputs:, :n_states]
    Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
    out = [Ad, Bd0, Bd1, C, D, int(delay/dt), x0]
    return out

if __name__== '__main__':
    out = ProcessModel(1,[3,2,1],1,0.1)
    print(out)
