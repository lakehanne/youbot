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

from geometry_msgs.msg import Wrench, Twist
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

# fix seed
np.random.seed(0)

roslib.load_manifest('youbot_navigation')

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
                        help='max num iterations' )
parser.add_argument('--plot_state', '-ps', action='store_true', default=False,
                        help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_false', default=True,
                        help='save plotted figures' )
parser.add_argument('--silent', '-si', action='store_true', default=True,
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
        self.l21_const        = config['agent']['alpha']
        self.action_penalty   = config['cost_params']['action_penalty']
        self.state_penalty    = config['cost_params']['state_penalty']
        self.terminate        = False

        self._hyperparams      = hyperparams
        self.wheel_rad        = self.wheel['radius']
        self.TOL              = config['agent']['TOL']

        # backpass regularizers
        self.mu      = config['agent']['mu']
        self.delta   = config['agent']['delta']
        self.mu_min  = config['agent']['mu_min']
        self.delta_nut = config['agent']['delta_nut']

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')
        self.pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        plt.ioff()
        self.fig = plt.figure()
        self._ax = self.fig.gca()

        self.traj_distr =  TrajectoryInfo(config)
        self.traj_distr.action_nominal[:,] = config['trajectory']['init_action']

    def generate_noise(self, T, dU, agent):
        """
        Generate a T x dU gaussian-distributed random walk noise vector. This will
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

        smooth      = agent['smooth_noise']
        var         = agent['smooth_noise_var']
        renorm      = agent['smooth_noise_renormalize']
        random_walk = agent['smooth_random_walk']

        noise       = np.random.randn(T, dU)

        if smooth:
            # Smooth noise. This violates the controller assumption, but
            # might produce smoother motions.
            for i in range(dU):
                noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)

            if random_walk:
                noise_cum = np.cumsum(noise, axis=1)

                # Determine the time evolution of the mean square distance.
                noise_cum_sq = noise_cum**2
                mean_sq_noise = np.mean(noise_cum_sq, axis=0) 

                noise = noise_cum_sq 
                      
            if renorm:
                variance = np.var(noise, axis=0)
                noise = noise / np.sqrt(variance)
        return noise

    def get_action_cost_jacs(self, noisy=True):

        T, dU, dX = self.T, self.dU, self.dX
        del_x_star = self.goal_state

        N = self._hyperparams.config['agent']['sample_length']

        u       = np.zeros((T, dU))
        x       = np.zeros((T, dX))
        fx      = np.zeros((T, dX, dX))
        fu      = np.zeros((T, dX, dU))
        fuu     = np.zeros((T, dU, dU))
        fux     = np.zeros((T, dU, dX))

        delta_u = np.zeros_like(u)
        delta_x = np.zeros_like(x)
        delta_x_star = self.goal_state

        # u_bar = self.traj_distr.action_nominal

        # see generalized ILQG Summary page
        k = range(0, T)
        K = len(k)
        delta_t = T/(K-1)
        t = [(_+1-1)*delta_t for _ in k]

        u_bar   = np.zeros((K, dU))
        x_bar   = np.zeros((K, dX))

        x_noise = self.generate_noise(T, dX, self._hyperparams.config['agent'])

        # apply u_bar to deterministic system
        for k in range(K-1):
            # get body dynamics. Note that some of these are time varying parameters
            body_dynamics = self.assemble_dynamics()

            # time varying inverse dynamics parameters
            M           = body_dynamics.M
            C           = body_dynamics.C
            B           = body_dynamics.B
            S           = body_dynamics.S
            qaccel      = body_dynamics.qaccel
            qvel        = body_dynamics.qvel
            q           = body_dynamics.q
            f           = body_dynamics.f

            # calculate the forward dynamics equation
            Minv        = np.linalg.inv(M)
            rhs         = - Minv.dot(C).dot(x_bar[k,:]) - Minv.dot(B.T).dot(S.dot(f)) \
                            + Minv.dot(B.T).dot(u_bar[k, :])/self.wheel_rad
            x_bar[k+1,:]= x_bar[k,:] +  delta_t * rhs

            # step 2.1: get linearized dynamics
            BBT         = B.dot(B.T)
            Inv_BBT     = np.linalg.inv(BBT)
            lhs         = Inv_BBT.dot(self.wheel_rad * B)
            
            # step 2.2: set up nonlinear dynamics at k
            u[k,:]      = lhs.dot(M.dot(qaccel) + C.dot(qvel) + \
                                    B.T.dot(S).dot(f)).squeeze()

            if noisy: # inject noise to the states
                x[k,:]  = q + x_noise[k, :]
            else:
                x[k,:]  = q

            rhs         = - Minv.dot(C).dot(x[k,:]) - Minv.dot(B.T).dot(S.dot(f)) \
                            + Minv.dot(B.T).dot(u[k,:])/self.wheel_rad
            x[k+1,:]    = x[k,:] +  delta_t * rhs

            # calculate the jacobians
            fx[k,:,:]   = np.eye(self.dX) - delta_t * Minv.dot(C)
            fu[k,:,:]   = -(delta_t * self.wheel_rad) * Minv.dot(B.T)  

            #step 2.3 set up state-action deviations
            delta_x[k,:] = x[k,:] - x_bar[k,:]
            delta_u[k,:] = u[k,:] - u_bar[k,:]

            # store away stage terms
            self.traj_distr.fx[k,:]            = fx[k,:]
            self.traj_distr.fu[k,:]            = fu[k,:]
            self.traj_distr.action[k,:]        =  u[k,:] 
            self.traj_distr.state[k,:]         =  x[k,:] 
            self.traj_distr.delta_state[k,:]   = delta_x[k,:]
            self.traj_distr.delta_action[k,:]  = delta_u[k,:]
            self.traj_distr.nominal_state[k,:] = x_bar[k,:]

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

        cost_action_term = 0.5 * np.sum(self.action_penalty[0] * \
                                    (np.linalg.norm(delta_u, axis=1) ** 2),
                                  axis = 0)
        cost_state_term  = 0.5 * np.sum(self.state_penalty[0] * \
                                    (np.linalg.norm(delta_x, axis=1) ** 2),
                                  axis = 0)
        cost_l12_term    = np.sqrt(self.l21_const + (delta_x - delta_x_star)**2)

        # nominal cost terms
        cost_nom_action_term = 0.5 * np.sum(self.action_penalty[0] * \
                                    (np.linalg.norm(u_bar, axis=1) ** 2),
                                    axis = 0)
        cost_nom_state_term  = 0.5 * np.sum(self.state_penalty[0] * \
                                    (np.linalg.norm(x_bar, axis=1) ** 2),
                                    axis = 0)
        cost_nom_l12_term    = np.sqrt(self.l21_const + (x_bar - delta_x_star)**2)        

        # define lx/lxx cost costants        
        final_state_diff = (x[-1,:] - self.goal_state)
        sqrt_term        = np.sqrt(self.l21_const + (final_state_diff**2))

        #system cost
        l       = cost_action_term + cost_state_term + cost_l12_term        
        # nominal cost about linear traj
        l_nom   = cost_nom_action_term + cost_nom_state_term + cost_nom_l12_term

        # first order cost terms
        lu = np.sum(self.action_penalty[0] * delta_u, axis=0) 
        lx = np.sum(self.state_penalty[0] * delta_x, axis=0)  \
                + (final_state_diff/np.sqrt(self.l21_const + final_state_diff**2)) 

        # 2nd order cost terms
        luu     = np.diag(self.action_penalty)        
        lux     = np.zeros((self.dU, self.dX))

        lxx_t1  = np.diag(self.state_penalty)
        lxx_t2_top = sqrt_term - (final_state_diff**2) /sqrt_term
        lxx_t2_bot = self.l21_const + (final_state_diff**2)

        lxx        = lxx_t1 + lxx_t2_top/lxx_t2_bot 

        # squeeze dims of first order derivatives
        lx = lx.squeeze() if lx.ndim > 1 else lx            
        lu = lu.squeeze() if lu.ndim > 1 else lu
            


        # store away stage terms
        self.traj_distr.fx            = fx
        self.traj_distr.fu            = fu
        self.traj_distr.action        = u 
        self.traj_distr.state         = x
        self.traj_distr.delta_state   = delta_x
        self.traj_distr.nominal_state = x_bar
        self.traj_distr.delta_action  = delta_u

        CostJacs = namedtuple('CostJac', ['l', 'lx', 'lu', 'lxx', 'l_nom', \
                                          'luu', 'lux'], verbose=False)
        return CostJacs(l=l, lx=lx, lu=lu, lxx=lxx, l_nom=l_nom, luu=luu, lux=lux)

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

        rospy.logdebug('running backward pass')
        self.backward(noisy=False)

        rospy.logdebug('running forward pass')
        self.forward()

    def backward(self, noisy=False):
        T  = self.T
        dU = self.dU
        dX = self.dX

        non_pos_def = True
        while (non_pos_def):

            non_pos_def = False
            rospy.logdebug('Restarting back pass')

            # retrieve the erstwhile costs for future T
            stage_jacs = self.get_action_cost_jacs()

            for t in range (T-1, -1, -1):
                """
                get derivatives in a different execution thread. Following Todorov's
                recommendation in synthesis and stabilization paper
                """
                # self.pool_derivatives.apply_async(self.get_action_cost_jacs, \
                #                 args=(t))

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

        base_link = 'base_footprint'
        
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

            # calculate the genralized force and torques
            sign_phi = np.sign(self.Phi_dot)
            bdyn = self.assemble_dynamics()

            F1 = (torques[0] - self.wheel_rad * sign_phi[0] * bdyn.f[0]) * \
                    (-(np.cos(theta) - np.sin(theta))/self.wheel_rad) + \
                 (torques[1] - self.wheel_rad * sign_phi[1] * bdyn.f[0]) * \
                    (-(np.cos(theta) + np.sin(theta))/self.wheel_rad)  + \
                 (torques[2] - self.wheel_rad * sign_phi[2] * bdyn.f[0]) * \
                    ((np.cos(theta) - np.sin(theta))/self.wheel_rad)  + \
                 (torques[3] - self.wheel_rad * sign_phi[3] * bdyn.f[0]) * \
                    ((np.cos(theta) + np.sin(theta))/self.wheel_rad)                    

            F2 = (torques[0] - self.wheel_rad * sign_phi[0] * bdyn.f[0]) * \
                    (-(np.cos(theta) + np.sin(theta))/self.wheel_rad) + \
                 (torques[1] - self.wheel_rad * sign_phi[1] * bdyn.f[0]) * \
                    (-(-np.cos(theta) + np.sin(theta))/self.wheel_rad)  + \
                 (torques[2] - self.wheel_rad * sign_phi[2] * bdyn.f[0]) * \
                    ((np.cos(theta) + np.sin(theta))/self.wheel_rad)  + \
                 (torques[3] - self.wheel_rad * sign_phi[3] * bdyn.f[0]) * \
                    ((-np.cos(theta) + np.sin(theta))/self.wheel_rad)

            F3 = np.sum(torques) * (-np.sqrt(2)*self.l * np.sin( np.pi/4 - self.alpha)/self.wheel_rad)  \
                   + (sign_phi[0] * bdyn.f[0] + sign_phi[1] * bdyn.f[0] + sign_phi[2] * bdyn.f[0] + sign_phi[3] * bdyn.f[0]) \
                        * (np.sqrt(2)* self.l * np.sin(np.pi/4 - self.alpha))

            rospy.loginfo('F1: {}, F2: {}, F3: {}'.format(F1, F2, F3))
            wrench_base = Wrench()
            wrench_base.force.x = F1
            wrench_base.force.y = F2

            base_angle = Twist()
            base_angle.angular.z = F3

            # send the torques to the base footprint
            # self.pub.publish(base_angle)
            resp_bf = send_body_wrench('base_footprint', reference_frame, 
                                            None, wrench_base, start_time, 
                                            duration )

            clear_bf = clear_active_wrenches('base_footprint')

            state_diff = self.traj_distr.delta_state[t,:] - self.goal_state

            if np.linalg.norm(state_diff) < self.TOL:
                rospy.loginfo("Successfully navigated to the goal state. \n"
                    "Current state: {} \n Goal state: {}"
                    .format(self.traj_distr.delta_state[t,:], 
                        self.goal_state))

                self.terminate = True

                break
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
            """
            # if args.silent:
            #     print('\n')
        if self.terminate:
            rospy.signal_shutdown("Achieved Navigation Goal")

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
        rospy.init_node('trajectory_optimization', 
                        disable_signals=True, anonymous=True, 
                        log_level=log_level)

        rospy.logdebug('Started trajectory optimization node')

        while not rospy.is_shutdown():
            # optimize_trajectories = threading.Thread(
            # target=lambda: trajopt.do_traj_opt()
            # )
            # optimize_trajectories.daemon = True
            # optimize_trajectories.start()

            trajopt.do_traj_opt()

            # if trajopt.terminate:
            #     break


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
