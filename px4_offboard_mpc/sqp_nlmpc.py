''' Installing acados:
        https://docs.acados.org/installation/index.html#windows-10-wsl
    Installing python interface:
        https://docs.acados.org/python_interface/index.html
    May need to install qpOASES version 3.1 as well.
'''


from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import casadi as cs
import numpy as np
from numpy import pi
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

import atexit
import shutil
import os


class SQP_NLMPC():
    ''' SQP approximation of nonlinear MPC using Acados's OCP solver.
    '''

    def __init__(self, model, Q, R, 
        time_step=0.1, num_nodes=20, u_max=40000):
        ''' Initialize the MPC with dynamics as casadi namespace,
            Q & R cost matrices, time-step,
            number of shooting nodes (length of prediction horizon),
            square of maximum motor frequency. 
        '''
        self.DT = time_step     
        self.N = num_nodes    
        model, self.u_hover = self.get_acados_model(model)
        self.solver = self.formulate_ocp(model, Q, R, u_max)
        
        # deleting acados compiled files when script is terminated.
        atexit.register(self.delete_compiled_files)
        return


    def get_acados_model(self, model_cs):
        ''' Acados model format:
        f_imp_expr/f_expl_expr, x, xdot, u, name '''

        u_hover = model_cs.hover * np.ones(model_cs.u.shape[0])
        model_ac = AcadosModel()
        model_ac.f_expl_expr = model_cs.f_expl_expr
        model_ac.x = model_cs.x
        model_ac.xdot = model_cs.xdot
        model_ac.u = model_cs.u 
        model_ac.name = model_cs.name
        return model_ac, u_hover


    def formulate_ocp(self, model, Q, R, u_max):
        ''' Guide to acados OCP formulation: 
        https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf '''
        
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        ny = nx + nu    # combine x and u into y

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.dims.nu = nu
        ocp.dims.nx = nx
        ocp.dims.ny = ny
        ocp.dims.nbx_0 = nx
        ocp.dims.nbu = nu   
        ocp.dims.nbx = 4    # number of states being constrained
        
        # total horizon in seconds
        ocp.solver_options.tf = self.DT*self.N  

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = la.block_diag(Q, R)

        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx,:nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:,-nu:] = np.eye(nu)

        # Initialize reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(nx)

        # control input constraints (square of motor freq)
        ocp.constraints.lbu = -u_max * np.ones(nu)
        ocp.constraints.ubu = u_max * np.ones(nu)   
        ocp.constraints.idxbu = np.arange(nu)

        # state constraints: z, roll, pitch, yaw
        inf = 1000000000
        ocp.constraints.lbx = np.array([
            0, -pi/2, -pi/2, 0
        ])
        ocp.constraints.ubx = np.array([
            inf, pi/2, pi/2, 2*pi
        ])
        ocp.constraints.idxbx = np.array([
            2, 3, 4, 5
        ])

        # not sure what this is, but this paper say partial condensing HPIPM 
        # is fastest: https://cdn.syscop.de/publications/Frison2020a.pdf
        ocp.solver_options.hpipm_mode = 'SPEED_ABS'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_iter_max = 1
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.print_level = 0

        # compile acados ocp
        solver = AcadosOcpSolver(ocp)
        return solver
    

    def run_optimization(self, x0, x_set, timer) -> np.ndarray:
        ''' Set initial state and setpoint,
            then solve the optimization once. 
        '''
        if timer: st = time.time()

        assert len(x0) == 12
        assert len(x_set) == 12

        # bound x0 to initial state
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)
        
        # the reference input will be the hover input
        y_ref = np.concatenate((x_set, self.u_hover))
        for k in range(self.N): 
            self.solver.set(k, 'yref', y_ref)
        
        # solve for the next ctrl input
        self.solver.solve()
        if timer: print(time.time() - st)
        return
    

    def get_next_control(self, x0, x_set, timer=False):
        self.run_optimization(x0, x_set, timer)
        nxt_ctrl = self.solver.get(0, 'u')
        return nxt_ctrl


    def get_next_state(self, x0, x_set, timer=False, visuals=False):
        self.run_optimization(x0, x_set, timer)
        nxt_state = self.solver.get(1, 'x')

        if visuals: 
            opt_us = np.zeros((self.N, self.u_hover.shape[0]))
            opt_xs = np.zeros((self.N, x0.shape[0]))
            for k in range(self.N):
                opt_us[k] = self.solver.get(k, 'u')
                opt_xs[k] = self.solver.get(k, 'x')
            self.vis_plots(opt_us, opt_xs)
        return nxt_state


    def vis_plots(self, ctrl_inputs:np.ndarray, trajectory:np.ndarray):
        ''' Displaying the series of control inputs 
            and trajectory over prediction horizon. 
        '''
        t = self.DT * np.arange(self.N)

        u1 = ctrl_inputs[:,0]
        u2 = ctrl_inputs[:,1]
        u3 = ctrl_inputs[:,2]
        u4 = ctrl_inputs[:,3]

        x = trajectory[:,0]
        y = trajectory[:,1]
        z = trajectory[:,2]

        phi = trajectory[:,3]
        theta = trajectory[:,4]
        psi = trajectory[:,5]

        x_dot = trajectory[:,6]
        y_dot = trajectory[:,7]
        z_dot = trajectory[:,8]

        phi_dot = trajectory[:,9]
        theta_dot = trajectory[:,10]
        psi_dot = trajectory[:,11]

        fig, axs = plt.subplots(5, figsize=(12, 10))

        axs[0].set_ylabel('ctrl inputs (1/s^2)')
        axs[0].plot(t,u1, label='u1')
        axs[0].plot(t,u2, label='u2')   
        axs[0].plot(t,u3, label='u3')
        axs[0].plot(t,u4, label='u4')
        axs[0].legend()
        
        axs[1].set_ylabel('position (m)')
        axs[1].plot(t,x, label='x')
        axs[1].plot(t,y, label='y')
        axs[1].plot(t,z, label='z')
        axs[1].legend()

        axs[2].set_ylabel('orientation (rad)')
        axs[2].plot(t,phi, label='phi')
        axs[2].plot(t,theta, label='theta')
        axs[2].plot(t,psi, label='psi')
        axs[2].legend()

        axs[3].set_ylabel('velocity (m/s)')
        axs[3].plot(t,x_dot,label='x_dot')
        axs[3].plot(t,y_dot,label='y_dot')
        axs[3].plot(t,z_dot,label='z_dot')
        axs[3].legend()

        axs[4].set_ylabel('angular vel (rad/s)')
        axs[4].plot(t,phi_dot,label='phi_dot')
        axs[4].plot(t,theta_dot,label='theta_dot')
        axs[4].plot(t,psi_dot,label='psi_dot')
        axs[4].legend()

        for ax in axs.flat:
            ax.set(xlabel='time (s)')
            ax.label_outer()
        
        plt.show()
        return
    

    def delete_compiled_files(self):
        ''' Deletes the acados generated files.
        '''
        try: shutil.rmtree('c_generated_code')
        except: print('failed to delete c_generated_code') 
        
        try: os.remove('acados_ocp_nlp.json')
        except: print('failed to delete acados_ocp_nlp.json')
    
 
def derive_quad_dynamics(mass, arm_len, Ix, Iy, Iz, thrust_coeff, torque_coeff):
    ''' Returns casadi struct containing explicit dynamics,
        state, state_dot, control input, and name. 
        Nonlinear continuous-time quadcopter dynamics. 
        The cartesian states are in ENU.
    '''
    # State Variables: position, rotation, and their time-derivatives
    x = cs.SX.sym('x')
    y = cs.SX.sym('y')
    z = cs.SX.sym('z')
    phi = cs.SX.sym('phi')     # roll
    theta = cs.SX.sym('theta') # pitch
    psi = cs.SX.sym('psi')     # yaw
    x_d = cs.SX.sym('x_d')     # time-derivatives
    y_d = cs.SX.sym('y_d')
    z_d = cs.SX.sym('z_d')
    phi_d = cs.SX.sym('phi_d')
    theta_d = cs.SX.sym('theta_d')
    psi_d = cs.SX.sym('psi_d')
    # state
    X = cs.vertcat(x, y, z, phi, theta, psi,\
        x_d, y_d, z_d, phi_d, theta_d, psi_d)


    # Inertial parameters
    m = mass #1.287 # kg #1282 iris3d g #0.027 crazyfly kg
    l = arm_len #0.040 # m
    Ixx = Ix #2.3951 * 10**(-5)
    Iyy = Iy #2.3951 * 10**(-5)
    Izz = Iz #3.2347 * 10**(-5)
    

    # Aerodynamic Parameters
    kf = thrust_coeff #0.005022
    km = torque_coeff #1.858 * 10**(-5)
    Ax = 0
    Ay = 0
    Az = 0


    # rotation matrix from body frame to inertial frame
    Rx = cs.SX(np.array([
        [1,           0,            0],
        [0,    cs.cos(phi),    -cs.sin(phi)],
        [0,    cs.sin(phi),     cs.cos(phi)]
    ]))
    Ry = cs.SX(np.array([
        [cs.cos(theta),   0,  cs.sin(theta)],
        [0,            1,           0],
        [-cs.sin(theta),  0,  cs.cos(theta)]
    ]))
    Rz = cs.SX(np.array([
        [cs.cos(psi),    -cs.sin(psi),    0],
        [cs.sin(psi),     cs.cos(psi),    0],
        [0,            0,           1]
    ]))
    R = Rz @ Ry @ Rx


    # calculation of jacobian matrix that converts body frame vels to inertial frame
    W = cs.SX(np.array([ 
        [1,  0,        -cs.sin(theta)],
        [0,  cs.cos(phi),  cs.cos(theta)*cs.sin(phi)],   
        [0, -cs.sin(phi),  cs.cos(theta)*cs.cos(phi)]
    ]))
    I = np.diag([Ixx,Iyy,Izz])
    J = W.T @ I @ W


    # Coriolis matrix for defining angular equations of motion
    C11 = 0

    C12 = (Iyy-Izz)*(theta_d*cs.cos(phi)*cs.sin(phi) + psi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Izz-Iyy)*psi_d*(cs.cos(phi)**2)*cs.cos(theta) -\
        Ixx*psi_d*cs.cos(theta)

    C13 = (Izz-Iyy)*psi_d*cs.cos(phi)*cs.sin(phi)*(cs.cos(theta)**2)

    C21 = (Izz-Iyy)*(theta_d*cs.cos(phi)*cs.sin(phi) + psi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Iyy-Izz)*psi_d*(cs.cos(phi)**2)*cs.cos(theta) +\
        Ixx*psi_d*cs.cos(theta)

    C22 = (Izz-Iyy)*phi_d*cs.cos(phi)*cs.sin(phi)

    C23 = -Ixx*psi_d*cs.sin(theta)*cs.cos(theta) +\
        Iyy*psi_d*(cs.sin(phi)**2)*cs.sin(theta)*cs.cos(theta) +\
        Izz*psi_d*(cs.cos(phi)**2)*cs.sin(theta)*cs.cos(theta)

    C31 = (Iyy-Izz)*psi_d*(cs.cos(theta)**2)*cs.sin(phi)*cs.cos(phi) -\
        Ixx*theta_d*cs.cos(theta)

    C32 = (Izz-Iyy)*(theta_d*cs.cos(phi)*cs.sin(phi)*cs.sin(theta) + phi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Iyy-Izz)*phi_d*(cs.cos(phi)**2)*cs.cos(theta) +\
        Ixx*psi_d*cs.sin(theta)*cs.cos(theta) -\
        Iyy*psi_d*(cs.sin(phi)**2)*cs.sin(theta)*cs.cos(theta) -\
        Izz*psi_d*(cs.cos(phi)**2)*cs.sin(theta)*cs.cos(theta)

    C33 = (Iyy-Izz)*phi_d*cs.cos(phi)*cs.sin(phi)*(cs.cos(theta)**2) -\
        Iyy*theta_d*(cs.sin(phi)**2)*cs.cos(theta)*cs.sin(theta) -\
        Izz*theta_d*(cs.cos(phi)**2)*cs.cos(theta)*cs.sin(theta) +\
        Ixx*theta_d*cs.cos(theta)*cs.sin(theta)

    C = cs.SX(np.array([
        [C11, C12, C13], 
        [C21, C22, C23], 
        [C31, C32, C33]
    ]))


    # Control Input is square of rotor frequency
    u1 = cs.SX.sym('u1')
    u2 = cs.SX.sym('u2')
    u3 = cs.SX.sym('u3')
    u4 = cs.SX.sym('u4')
    U =  cs.vertcat(u1, u2, u3, u4)


    # actuation dynamics
    tau_beta = cs.SX(np.array([
        [l*kf*(-u2 + u4)],
        [l*kf*(-u1 + u3)],
        [km*(-u1 + u2 - u3 + u4)]
    ]))
    thrust = kf*(u1 + u2 + u3 + u4)
    g = 9.81    # acceleration due to gravity


    # continuous-time dynamics
    Xdot = cs.vertcat(
        x_d, y_d, z_d, phi_d, theta_d, psi_d,
        cs.vertcat(0,0,-g) + R @ cs.vertcat(0,0,thrust) / m,
        cs.inv(J) @ (tau_beta - C @ cs.vertcat(phi_d, theta_d, psi_d))
    )


    # store variables in casadi struct
    model_cs = cs.types.SimpleNamespace()
    model_cs.f_expl_expr = Xdot
    model_cs.x = X
    model_cs.xdot = Xdot
    model_cs.u = U
    model_cs.hover = (m*g) / (kf*U.shape[0])  # min. control for hover
    model_cs.name = 'nonlin_quadcopter'
    return model_cs

