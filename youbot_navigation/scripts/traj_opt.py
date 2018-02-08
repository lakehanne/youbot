#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

import os
import imp
import time
import copy
import rospy
import roslib
import logging
import argparse
import threading
import numpy as np
import scipy as sp

import rospkg
from nav_msgs.msg import Odometry
from rospy.rostime import Duration

from geometry_msgs.msg import Wrench
# from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchResponse, \
#                             ApplyJointEffort, ApplyJointEffortResponse, \
#                             JointRequest, BodyRequest

# import torque and wrench services                            
from scripts.services import send_joint_torques, clear_active_forces, \
                             send_body_wrench, clear_active_wrenches

from numpy.linalg import LinAlgError
import scipy.ndimage as sp_ndimage
from collections import namedtuple
from scipy.integrate import odeint, RK45

from scripts.dynamics import Dynamics
from scripts.sample import SampleList
from scripts.algorithm_utils import TrajectoryInfo

from multiprocessing import Pool, Process

import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt


roslib.load_manifest('youbot_navigation')

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
                        help='max num iterations' )
parser.add_argument('--plot_state', '-ps', action='store_true', default=False,
                        help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_false', default=True,
                        help='save plotted figures' )
parser.add_argument('--silent', '-si', action='store_true', default=False,
                        help='max num iterations' )
args = parser.parse_args()


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)

if args.silent:
    LOGGER.setLevel(logging.INFO)
else:
    LOGGER.setLevel(logging.DEBUG)
print(args)

class TrajectoryOptimization(Dynamics):
    """docstring for TrajectoryOptimization
    """
    def __init__(self, arg, rate, hyperparams):
        super(TrajectoryOptimization, self).__init__(Dynamics(Odometry, rate=rate))

        self.args            = arg
        self.pool_derivatives= Pool(processes=2)

        config                = hyperparams.config
        self.T                = config['agent']['T']
        self.dU               = config['agent']['dU']
        self.dX               = config['agent']['dX']
        self.euler_step       = config['agent']['euler_step']
        self.euler_iter       = config['agent']['euler_iter']
        self.goal_state       = config['agent']['goal_state']
        self.alpha            = config['agent']['alpha']
        self.action_penalty   = config['cost_params']['action_penalty']
        self.state_penalty    = config['cost_params']['state_penalty']

        self.hyperparams      = hyperparams
        self.wheel_rad        = self.wheel['radius']

        # backpass regularizers
        self.mu      = config['agent']['mu']
        self.delta   = config['agent']['delta']
        self.mu_min  = config['agent']['mu_min']
        self.delta_nut = config['agent']['delta_nut']

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')

        plt.ioff()
        self.fig = plt.figure()
        self._ax = self.fig.gca()

        self.traj_distr =  TrajectoryInfo(config)
        self.traj_distr.action_nominal[:,] = config['trajectory']['init_action']

    def generate_noise(self, T, dU, agent):
        """
        Generate a T x dU gaussian-distributed noise vector. This will
        approximately have mean 0 and variance 1, ignoring smoothing.

        Args:
            T: Number of time steps.
            dU: Dimensionality of actions.
        agent:
            smooth: Whether or not to perform smoothing of noise.
            var : If smooth=True, applies a Gaussian filter with this
                variance.
            renorm : If smooth=True, renormalizes data to have variance 1
                after smoothing.
        """
        smooth, var = agent['smooth_noise'], agent['smooth_noise_var']
        renorm = agent['smooth_noise_renormalize']
        noise = np.random.randn(T, dU)
        if smooth:
            # Smooth noise. This violates the controller assumption, but
            # might produce smoother motions.
            for i in range(dU):
                noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
            if renorm:
                variance = np.var(noise, axis=0)
                noise = noise / np.sqrt(variance)
        return noise

    def get_action_cost_jacs(self, t, noisy=False):
        # get body dynamics. Note that some of these are time varying parameters
        body_dynamics = self.assemble_dynamics()

        # time varying inverse dynamics parameters
        mass_matrix     = body_dynamics.M
        coriolis_matrix = body_dynamics.C
        B_matrix        = body_dynamics.B
        S_matrix        = body_dynamics.S
        qaccel          = body_dynamics.qaccel
        qvel            = body_dynamics.qvel
        q               = body_dynamics.q

        # this should be time-varying but is constant for now
        friction_vector = body_dynamics.f

        # print('q: {}, \n qvel: {} \n qaccel: {}'.format(q, qvel, qaccel))
        # calculate inverse dynamics equation
        BBT             = B_matrix.dot(B_matrix.T)
        Inv_BBT         = np.linalg.inv(BBT)
        multiplier      = Inv_BBT.dot(self.wheel_rad * B_matrix)
        inv_dyn_eq      = mass_matrix.dot(qaccel) + coriolis_matrix.dot(qvel) + \
                                B_matrix.T.dot(S_matrix).dot(friction_vector)

        # set up costs at time T
        self.traj_distr.action[t,:]        = multiplier.dot(inv_dyn_eq).squeeze()
        self.traj_distr.state[t,:]         = q.squeeze()

        """
        see a generalized ILQG paper in 43rd CDC | section V.
        the euler step multiplier takes the ode equations it to first order derivatives from 2nd order change of vars
        """
        self.traj_distr.fx[t,:,:]          = np.eye(self.dX) -self.euler_step * np.linalg.inv(mass_matrix).dot(coriolis_matrix)
        self.traj_distr.fu[t,:,:]          = -(self.euler_step * self.wheel_rad) * np.linalg.inv(mass_matrix).dot(B_matrix.T)
        rospy.logdebug("Integrating inverse dynamics equation")

        mass_inv            = -np.linalg.inv(mass_matrix)
        for n in range(1, self.euler_iter):
            self.traj_distr.nominal_state_[n,:] = self.traj_distr.nominal_state_[n-1,:] + \
                            self.euler_step * (mass_inv.dot(coriolis_matrix).dot(self.traj_distr.nominal_state[t,:])) - \
                            mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector)+ \
                            (mass_inv.dot(B_matrix.T).dot(self.traj_distr.action_nominal[t,:]))/self.wheel_rad
        self.traj_distr.nominal_state[t,:] = self.traj_distr.nominal_state_[n,:] # decode nominal state at last euler step

        if self.args.plot_state:
            self._ax.plot(self.traj_distr.nominal_state[t:,:], 'b', label='qpos', fontweight='bold')
            # self._ax.plot(tt, nominal_state[:,1], 'g', label='qpos', fontweight='bold')
            self._ax.legend(loc='best')
            self._ax.set_xlabel('time (discretized)', fontweight='bold')
            self._ax.set_ylabel('final q after integration', fontweight='bold')
            self._ax.grid()
            self._ax.gcf().set_size_inches(10,4)
            self._ax.cla()

            if self.args.save_figs:
                figs_dir = os.path.join(self.path, 'figures')
                os.mkdir(figs_dir) if not os.path.exists(figs_dir) else None
                self.fig.savefig(figs_dir + '/state_' + repr(t),
                        bbox_inches='tight',facecolor='None')

        # print('nom action: ', self.traj_distr.action_nominal[t,:].T)
        # calculate new system trajectories
        self.traj_distr.delta_state[t,:]  = self.traj_distr.state[t,:]  - self.traj_distr.nominal_state[t,:]
        self.traj_distr.delta_action[t,:] = self.traj_distr.action[t,:] - self.traj_distr.action_nominal[t,:]

        if noisy:
            noise_covar[t] = (self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:])).T.dot(\
                                self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:]))

        state_diff     = np.expand_dims(self.traj_distr.delta_state[t,:] - self.goal_state, 1)
        state_diff_nom = np.expand_dims((self.traj_distr.nominal_state[t,:] - self.goal_state), 1)

        cost_action_term = self.action_penalty[0] * np.expand_dims(self.traj_distr.delta_action[t,:], axis=0).dot(\
                            np.expand_dims(self.traj_distr.delta_action[t,:], axis=1))
        cost_state_term  = 0.5 * self.state_penalty[0] * state_diff.T.dot(state_diff)
        cost_l12_term    = np.sqrt(self.alpha + state_diff.T.dot(state_diff))

        # nominal action
        cost_nom_action_term = self.action_penalty[0] * np.expand_dims(self.traj_distr.action_nominal[t,:], axis=0).dot(\
                                                        np.expand_dims(self.traj_distr.action_nominal[t,:], axis=1))
        cost_nom_state_term  = 0.5 * self.state_penalty[0] * state_diff_nom.T.dot(state_diff_nom)
        cost_nom_l12_term    = np.sqrt(self.alpha + state_diff_nom.T.dot(state_diff_nom))

        #system cost
        l       = cost_action_term + cost_state_term + cost_l12_term
        # nominal cost about linear traj
        l_nom   = cost_nom_action_term + cost_nom_state_term + cost_nom_l12_term
        # first order derivatives
        lu      = self.action_penalty * self.traj_distr.delta_action[t,:]
        lx      = self.state_penalty[0] * state_diff + state_diff/np.sqrt(self.alpha + state_diff.T.dot(state_diff))
                                   
        luu     = np.diag(self.action_penalty)

        lxx_t1 = np.diag(self.state_penalty)
        lxx_t2 = np.eye(self.dX)
        lxx_t3 = (state_diff.T.dot(state_diff)) / ((self.alpha + state_diff.T.dot(state_diff))**3)
        lxx    = lxx_t1 +  lxx_t2 * np.eye(self.dX) - lxx_t3 * np.eye(self.dX)
        lux    = np.zeros((self.dU, self.dX))

        # generate random noise
        noise = self.generate_noise(self.T, self.dU, self.hyperparams.config['agent'])

        # print('fx: {} \n fu: {}'.format(self.traj_distr.fx[t,:,:], self.traj_distr.fu[t,:,:]))

        # squeeze dims of first order derivatives
        if lx.ndim > 1:
            lx = lx.squeeze()
        if lu.ndim > 1:
            lu = lu.squeeze()

        CostJacs = namedtuple('CostJac', ['l', 'lx', 'lu', 'lxx', 'l_nom', \
                                          'luu', 'lux', 'noise'], verbose=False)
        return CostJacs(l=l, lx=lx, lu=lu, lxx=lxx, l_nom=l_nom, luu=luu, lux=lux, noise=noise)

    def reg_sched(self, increase=False):
        # increase mu
        if increase:
            self.delta = max(self.delta_nut, self.delta * self.delta_nut)
            mu = self.mu * 1.1 #max(self.mu_min, self.mu * self.delta) #
        else:
            self.delta = min(1/self.delta_nut, self.delta/self.delta_nut)
            if self.mu * self.delta > self.mu_min:
                self.mu = self.mu * self.delta
            else:
                self.mu = 0
        self.mu = mu

    def get_new_state(self, theta, t):
        body_dynamics = self.assemble_dynamics()

        # time varying inverse dynamics parameters
        M     = body_dynamics.M
        C     = body_dynamics.C
        B     = body_dynamics.B
        S     = body_dynamics.S
        f     = body_dynamics.f
        qvel  = body_dynamics.qvel
        qaccel= body_dynamics.qaccel

        # update time-varying parameters of mass matrix
        d1, d2  =  1e-2, 1e-2

        mw, mb, r  = self.wheel['mass'], self.base['mass'], self.wheel['radius']
        I, I_b     = self.wheel['mass_inertia'][1,1], self.base['mass_inertia'][-1,-1]
         
        base_footprint_dim = 0.001  # retrieved from the box geom in gazebo
        l = np.sqrt(2* base_footprint_dim)
        l_sqr = 2* base_footprint_dim

        b, a = 0.19, 0.145 # meters as measured from the real robot
        alpha = np.arctan2(b, a)

        m13 = mb * ( d1 * np.sin(theta) + d2 * np.cos(theta) )
        m23 = mb * (-d1 * np.cos(theta) + d2 * np.sin(theta) )
        m33 = mb * (d1 ** 2 + d2 ** 2) + I_b + \
                    8 * (mw + I/(r**2)) * l_sqr * pow(np.sin(np.pi/4.0 - alpha), 2)

        # update mass_matrix
        M[0,2], M[2,0], M[1,2], M[2,1] = m13, m13, m23, m23

        # update B matrix
        B[:,:2].fill(np.cos(theta) + np.sin(theta))
        B[:,-1].fill(-np.sqrt(2)*l*np.sin(np.pi/4 - alpha))
        B[0,0] = np.sin(theta) - np.cos(theta)
        B[1,0] *= -1
        B[2,0] = np.cos(theta) - np.sin(theta)
        B[0,1]          = -1.0 * B[0,1]
        B[1,1], B[3,1]  = B[2,0], B[0,0]

        # calculate phi dor from eq 6
        Phi_coeff = -(np.sqrt(2)/r) 
        # mid matrix
        Phi_left_mat = np.ones((4, 3))
        Phi_left_mat[:,:2].fill(np.sqrt(2)/2)
        Phi_left_mat[:,2].fill(l*np.sin(np.pi/4 - alpha))
        # column 0
        Phi_left_mat[2, 0] *= -1
        Phi_left_mat[3, 0] *= -1
        # column 1
        Phi_left_mat[1, 1] *= -1
        Phi_left_mat[2, 1] *= -1

        Phi_right_mat = np.zeros((3,3))
        Phi_right_mat[0,0] = np.cos(theta)
        Phi_right_mat[1,1] = np.cos(theta)

        Phi_right_mat[0,1] = np.sin(theta)
        Phi_right_mat[1,0] = -np.sin(theta)

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        Phi_right_vector   = np.asarray([xdot, ydot, theta_dot]) 
        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left_mat.dot(Phi_right_mat).dot(Phi_right_vector)
        S = np.diag(np.sign(Phi_dot).squeeze())

        # calculate inverse dynamics equation
        BBT             = B.dot(B.T)
        Inv_BBT         = np.linalg.inv(BBT)
        multiplier      = Inv_BBT.dot(self.wheel_rad * B)
        inv_dyn_eq      = M.dot(qaccel) + C.dot(qvel) + \
                                B.T.dot(S).dot(f)

        mass_inv            = -np.linalg.inv(M)
        for n in range(1, self.euler_iter):
            self.traj_distr.nominal_state_[n,:] = self.traj_distr.nominal_state_[n-1,:] + \
                            self.euler_step * (mass_inv.dot(C).dot(self.traj_distr.nominal_state[t,:])) - \
                            mass_inv.dot(B.T).dot(S).dot(f)+ \
                            (mass_inv.dot(B.T).dot(self.traj_distr.action_nominal[t,:]))/self.wheel_rad
        new_state = self.traj_distr.nominal_state_[n,:] # decode nominal state at last euler step


        return new_state

    def do_traj_opt(self):

        rospy.loginfo('running backward pass')
        self.backward(noisy=False)

        rospy.loginfo('running forward pass')
        self.forward()

    def backward(self, noisy=False):
        T  = self.T
        dU = self.dU
        dX = self.dX

        non_pos_def = True
        while (non_pos_def):

            non_pos_def = False
            rospy.logdebug('Restarting back pass')
            for t in range (T-1, -1, -1):
                """
                get derivatives in a different execution thread. Following Todorov's
                recommendation in synthesis and stabilization paper
                """
                # self.pool_derivatives.apply_async(self.get_action_cost_jacs, \
                #                 args=(t))
                stage_jacs = self.get_action_cost_jacs(t)

                self.traj_distr.Qx[t,:]     = stage_jacs.lx
                self.traj_distr.Qu[t,:]     = stage_jacs.lu
                self.traj_distr.Qxx[t,:,:]  = stage_jacs.lxx
                self.traj_distr.Qux[t,:,:]  = stage_jacs.lux
                self.traj_distr.Quu[t,:,:]  = stage_jacs.luu
                self.traj_distr.Quu_tilde[t,:,:]  = stage_jacs.luu

                # form Q derivatives at time t first
                if t < T-1:
                    # wil be --> (3) + (3x3) x (3) ==> (3)
                    self.traj_distr.Qx[t,:] = stage_jacs.lx + self.traj_distr.fx[t,:].T.dot(self.traj_distr.Vx[t+1,:])
                    # wil be --> (4) + (4,3) x (3) ==> (4)
                    self.traj_distr.Qu[t,:] = stage_jacs.lu + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vx[t+1,:])
                    # wil be --> (3) + (3,3) x (3,3) x ((3,3)) ==> (3,3)
                    self.traj_distr.Qxx[t,:,:]  = stage_jacs.lxx + self.traj_distr.fx[t,:].T.dot(self.traj_distr.Vxx[t+1,:,:]).dot(self.traj_distr.fx[t,:])
                    # wil be --> (4, 3) + (4,3) x (3,3) x ((3,3)) ==> (4,3)
                    self.traj_distr.Qux[t,:,:]  = stage_jacs.lux + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vxx[t+1,:,:]).dot(self.traj_distr.fx[t,:])
                    # wil be --> (4, 4) + (4,3) x (3,3) x ((3,4)) ==> (4,4)
                    self.traj_distr.Quu[t,:,:]  = stage_jacs.luu + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vxx[t+1,:,:]).dot(self.traj_distr.fu[t,:])

                    # calculate the Qvals that penalize devs from the states
                    self.traj_distr.Qu_tilde[t,:] = self.traj_distr.Qu[t,:] #+ self.mu * np.eye(self.traj_distr.Qu[t,:].shape[0])
                    self.traj_distr.Quu_tilde[t,:,:] = stage_jacs.luu + self.traj_distr.fu[t,:].T.dot(\
                                    self.traj_distr.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(self.traj_distr.fu[t,:]) #+ \
                                    # self.traj_distr.Vx[t+1,:].dot(self.traj_distr.fuu[t,:,:])
                    self.traj_distr.Qux_tilde[t,:,:] = stage_jacs.lux + self.traj_distr.fu[t,:].T.dot(\
                                    self.traj_distr.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(self.traj_distr.fx[t,:]) #+ \
                                    # self.traj_distr.Vx[t+1,:].dot(self.traj_distr.fux[t,:,:])

                # symmetrize the second order moments of Q
                self.traj_distr.Quu[t,:,:] = 0.5 * (self.traj_distr.Quu[t].T + self.traj_distr.Quu[t])
                self.traj_distr.Qxx[t,:,:] = 0.5 * (self.traj_distr.Qxx[t].T + self.traj_distr.Qxx[t])
                
                # symmetrize for improved Q state improvement values too
                self.traj_distr.Quu_tilde[t,:,:] = 0.5 * (self.traj_distr.Quu_tilde[t].T + \
                            self.traj_distr.Quu_tilde[t])
                # Compute Cholesky decomposition of Q function action component.
                try:
                    U_tilde = sp.linalg.cholesky(self.traj_distr.Quu_tilde[t,:,:])
                    L_tilde = U_tilde.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u]  or Q_tilde[u, u] is not spd
                    rospy.loginfo('LinAlgError: %s', e)
                    non_pos_def = True
                    # restart the backward pass if Quu is non positive definite
                    break

                # compute open and closed loop gains.
                small_k   = -sp.linalg.solve_triangular(
                    U_tilde, sp.linalg.solve_triangular(L_tilde,
                    self.traj_distr.Qu_tilde[t, :], lower=True)
                )
                big_K = -sp.linalg.solve_triangular(
                    U_tilde, sp.linalg.solve_triangular(L_tilde,
                    self.traj_distr.Qux_tilde[t, :, :], lower=True)
                )
                # store away the gains
                self.traj_distr.gu[t, :] = small_k
                self.traj_distr.Gu[t, :, :] = big_K

                # calculate imporved value functions
                self.traj_distr.Vxx[t,:,:] = self.traj_distr.Qxx[t, :,:] + \
                                big_K.T.dot(self.traj_distr.Quu[t,:,:]).dot(big_K) + \
                                big_K.T.dot(self.traj_distr.Qux[t,:,:]) + \
                                self.traj_distr.Qux[t,:,:].T.dot(self.traj_distr.Gu[t,:,:])

                self.traj_distr.Vx[t,:] = self.traj_distr.Qx[t,:] + \
                                big_K.T.dot(self.traj_distr.Quu[t,:,:]).dot(small_k) + \
                                big_K.T.dot(self.traj_distr.Qu[t,:]) + \
                                self.traj_distr.gu[t,:].T.dot(self.traj_distr.Qux[t,:,:])

                # self.traj_distr.V[t] = 0.5 * small_k.T.dot(self.traj_distr.Quu[t,:,:]).dot(small_k) + \
                #                         small_k.T.dot(self.traj_distr).dot(self.traj_distr.Qu[t,:])
                # symmetrize quadratic Value hessian
                self.traj_distr.Vxx[t,:,:] = 0.5 * (self.traj_distr.Vxx[t,:,:] + self.traj_distr.Vxx[t,:,:].T)

            if non_pos_def: # restart the back-pass process
                old_mu = self.mu
                self.reg_sched(increase=True)
                rospy.logdebug("Hessian became non positive definite")
                rospy.logdebug('Increasing mu: {} -> {}'.format(old_mu, self.mu))
                # break

    def forward(self):
        T  = self.T
        dU = self.dU
        dX = self.dX

        # joint names of the four wheels
        wheel_joint_bl = 'wheel_joint_bl'
        wheel_joint_br = 'wheel_joint_br'
        wheel_joint_fl = 'wheel_joint_fl'
        wheel_joint_fr = 'wheel_joint_fr'
        
        start_time = Duration(secs = 0, nsecs = 0) # start asap 

        rospy.loginfo("stepped into forward pass of algorithm")
        
        reference_frame = None #'empty/world/map'
        # update the states and controls 
        for t in range(T):
            # self.pool_derivatives.apply_async(self.get_action_cost_jacs, \
            #                     args=(t))
            stage_jacs = self.get_action_cost_jacs(t)
            # update states
            self.traj_distr.delta_state[t,:]  = self.traj_distr.state[t,:]  - self.traj_distr.nominal_state[t,:]
            self.traj_distr.delta_action[t,:] = self.traj_distr.delta_action[t,:] \
                                                    + self.traj_distr.gu[t, :] \
                                                    + self.traj_distr.Gu[t, :].dot(self.traj_distr.delta_state[t,:])
            
            theta = self.traj_distr.delta_state[t,:][-1]

            rospy.logdebug('action nominal: {}'.format(self.traj_distr.action_nominal[t]))
            rospy.logdebug('delta  action : {}'.format(self.traj_distr.delta_action[t]))
            rospy.logdebug('action : {}'.format(self.traj_distr.action[t]))


            duration = Duration(secs = 5, nsecs = 0) # apply effort continuously without end duration = -1
            # update state at t+1
            if t < T-1:
                self.traj_distr.state[t+1,:]  = self.get_new_state(theta, t+1)
            torques = self.traj_distr.delta_action[t,:]
            # send in this order: 'wheel_joint_bl', 'wheel_joint_br', 'wheel_joint_fl', 'wheel_joint_fr'
            # create four different asynchronous threads for each wheel
            """
            wheel_joint_bl_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_bl_thread', 
                                        args=(wheel_joint_bl, torques[0], start_time, duration))
            wheel_joint_br_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_br_thread', 
                                        args=(wheel_joint_br, torques[1], start_time, duration))
            wheel_joint_fl_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_fl_thread', 
                                        args=(wheel_joint_fl, torques[2], start_time, duration))
            wheel_joint_fr_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_fr_thread', 
                                        args=(wheel_joint_fr, torques[3], start_time, duration))
            
            """
            wrench_bl, wrench_br, wrench_fl, wrench_fr = Wrench(), Wrench(), Wrench(), Wrench()

            wrench_bl.force.x = torques[0]
            wrench_bl.force.y = torques[0]

            wrench_br.force.x = torques[1]
            wrench_br.force.y = torques[1]

            wrench_fl.force.x = torques[2]
            wrench_fl.force.y = torques[2]

            wrench_fr.force.x = torques[3]
            wrench_fr.force.y = torques[3]

            resp_bl = send_body_wrench('wheel_link_bl', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_br = send_body_wrench('wheel_link_br', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_fl = send_body_wrench('wheel_link_fl', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_fr = send_body_wrench('wheel_link_fr', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            rospy.sleep(duration)

            # clear active wrenches
            clear_bl = clear_active_wrenches('wheel_link_bl')
            clear_br = clear_active_wrenches('wheel_link_bl')
            clear_fl = clear_active_wrenches('wheel_link_bl')
            clear_fr = clear_active_wrenches('wheel_link_bl')

            if args.silent:
                print('\n\n')

            # # https://docs.python.org/2/library/threading.html
            # wheel_joint_bl_thread.daemon = True
            # wheel_joint_br_thread.daemon = True
            # wheel_joint_fl_thread.daemon = True
            # wheel_joint_fr_thread.daemon = True

            # # send torques to robot
            # wheel_joint_fl_thread.start()
            # wheel_joint_fr_thread.start() 
            # wheel_joint_bl_thread.start()
            # wheel_joint_br_thread.start()   

            # # if t < 5:
            #     # timeout = t * 2
            # timeout=t
            """
            wait until last thread finishes before running the next time step
            this places the for loop in a blocking call
            """
            # wheel_joint_fr_thread.join(timeout=timeout) 

if __name__ == '__main__':

    from scripts import __file__ as scripts_filepath
    scripts_filepath = os.path.abspath(scripts_filepath)
    scripts_dir = '/'.join(str.split(scripts_filepath, '/')[:-1]) + '/'
    hyperparams_file = scripts_dir + 'config.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.silent:
        log_level = rospy.INFO
    else:
        log_level = rospy.DEBUG
    try:

        trajopt = TrajectoryOptimization(args, rate=30, hyperparams=hyperparams)
        rospy.init_node('trajectory_optimization', anonymous=True, log_level=log_level)

        rospy.logdebug('Started trajectory optimization node')

        while not rospy.is_shutdown():
            # optimize_trajectories = threading.Thread(
            # target=lambda: trajopt.do_traj_opt()
            # )
            # optimize_trajectories.daemon = True
            # optimize_trajectories.start()

            trajopt.do_traj_opt()


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
