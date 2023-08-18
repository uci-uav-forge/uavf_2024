
import os
import numpy as np
from px4_offboard_mpc.sqp_nlmpc import SQP_NLMPC, derive_quad_dynamics


def main():
    # Inertial parameters from crazyflie in MIT paper
    m = 0.027 # kg
    l = 0.040 # m
    Ixx = 2.3951 * 10**(-5)
    Iyy = 2.3951 * 10**(-5)
    Izz = 3.2347 * 10**(-5)
    
    # Aerodynamic Parameters from crazyflie in MIT paper
    kf = 0.005022
    km = 1.858 * 10**(-5)
    nl_quad_model = derive_quad_dynamics(m,l,Ixx,Iyy,Izz,kf,km)
    
    x0 = np.array([0,0,8, 0,0,0, 0,0,0, 0,0,0])
    x_set = np.array([1,0,4, 0,0,0, 0,0,0, 0,0,0])
    Q = np.diag(
        [4,4,4, 2,2,2, 1,1,1, 1,1,1]
    )
    R = 0.1 * np.diag(
        [1,1,1,1]
    )

    sqp_nlmpc = SQP_NLMPC(
        nl_quad_model, Q, R, 
        time_step=0.07, num_nodes=20, u_max=10000
    )
    print(sqp_nlmpc.next_control_and_state(
        x0=x0, x_set=x_set, visuals=True, timer=True
    ))


if __name__=='__main__':
    main()