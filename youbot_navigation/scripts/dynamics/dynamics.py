#!/usr/bin/env python
from __future__ import print_function

import os
import imp
import rospy
import logging
import numpy as np
import multiprocessing
from nav_msgs.msg import Odometry
from collections import namedtuple

from scripts.dynamics.constants import MassMaker
from scripts.algorithm_utils import TrajectoryInfo, \
                            generate_noise, CostInfo

from multiprocessing.pool import ThreadPool
from geometry_msgs.msg import Twist, \
        Pose, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion

# fix random seed
np.random.seed(0)

from scripts import __file__ as scripts_filepath
scripts_filepath = os.path.abspath(scripts_filepath)
scripts_dir = '/'.join(str.split(scripts_filepath, '/')[:-1]) + '/'
hyperparams_file = scripts_dir + 'config.py'
hyperparams = imp.load_source('hyperparams', hyperparams_file)
config = hyperparams.config

LOGGER = logging.getLogger(__name__)

class Dynamics(MassMaker):
    def __init__(self, rate):
        super(Dynamics, self).__init__()
        self.rate = 30  # 10Hz

        mat_maker = MassMaker()
        mat_maker.get_mass_matrices()

        self.agent            = config['agent']
        self.T                = config['agent']['T']
        self.dU               = config['agent']['dU']
        self.dX               = config['agent']['dX']
        self.goal_state       = config['agent']['goal_state']

        self.__dict__.update(mat_maker.__dict__)
        num_samples = config['agent']['sample_length']

    def odom_cb(self, odom):
        self.odom = odom

    def listener(self):
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        sleeper = rospy.Rate(self.rate)
        sleeper.sleep()


    def assemble_dynamics(self):
        # time for accelaeration calculation
        prev_time = rospy.get_rostime().to_sec()
        self.listener()
        now_time  = rospy.get_rostime().to_sec()
        time_delta = now_time - prev_time

        # gather dynamics equation components
        mb  = self.base['mass']
        mw  = self.wheel['mass']
        r   = self.wheel['radius']

        """
        I_b is the moment of inertia of the platform about Zr' axis thru G'
        I is the mom of inertia of ith wheel about main axis
        """

        I_b = self.base['mass_inertia'][-1,-1]
        I   = self.wheel['mass_inertia'][1,1]

        # f_i are the frictional force of each of the four wheels
        f   = np.array([
                      self.wheel['friction'], self.wheel['friction'],
                      self.wheel['friction'], self.wheel['friction']
                     ])
        base_footprint_dim = 0.001  # retrieved from the box geom in gazebo
        l   = np.sqrt(2* base_footprint_dim)
        l_sqr = 2* base_footprint_dim
        b, a = 0.19, 0.145 # meters as measured from the real robot
        alpha = np.arctan2(b, a)

        x     = self.odom.pose.pose.position.x
        y     = self.odom.pose.pose.position.y

        # quaternion = [self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x,
        #                 self.odom.pose.pose.orientation.y,self.odom.pose.pose.orientation.z] 
        # see https://answers.ros.org/question/69754/quaternion-transformations-in-python/      
        quaternion = [self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x,
                        self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z]
        _, _, theta = euler_from_quaternion(quaternion, axes='sxyz')
        # theta -= np.pi/2.0  # account for diffs of frames in gazebo and paper

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        d1, d2 = 1e-2, 1e-2 # not sure of this

        # define mass matrix elements
        m11 = mb + 4 * (mw + I/(r**2))
        m22 = mb + 4 * (mw + I/(r**2))
        m13 = mb * ( d1 * np.sin(theta) + d2 * np.cos(theta) )
        m23 = mb * (-d1 * np.cos(theta) + d2 * np.sin(theta) )
        m33 = mb * (d1 ** 2 + d2 ** 2) + I_b + \
                    8 * (mw + I/(r**2)) * l_sqr * pow(np.sin(np.pi/4.0 - alpha), 2)

        # assemble mass inertia matrix
        M = np.diag([m11, m22, m33])
        M[0,2], M[2,0], M[1,2], M[2,1] = m13, m13, m23, m23

        # B matrix components
        B = np.zeros((4, 3))
        B[:,:2].fill(np.cos(theta) + np.sin(theta))
        B[:,-1].fill(-np.sqrt(2)*l*np.sin(np.pi/4 - alpha))

        B[0,0] = np.sin(theta) - np.cos(theta)
        B[1,0] *= -1
        B[2,0] = np.cos(theta) - np.sin(theta)

        B[0,1] *= -1.0
        B[1,1] = B[2,0]
        B[3,1] = B[0,0]

        xaccel = xdot/time_delta
        yaccel = ydot/time_delta
        theta_accel = theta_dot/time_delta

        LOGGER.debug("Time between motions: {}".format(time_delta))

        q       = np.array([x, y, theta]) #np.expand_dims(np.array([x, y, theta]), axis=1)
        qdot    = np.array([xdot, ydot, theta_dot]) #np.expand_dims(np.array([xdot, ydot, theta_dot]), axis=1)
        qdotdot = np.array([xaccel, yaccel, theta_accel]) #np.expand_dims(np.array([xaccel, yaccel, theta_accel]), axis=1)

        # C matrix components
        c13 = mb * theta_dot * (d1 * np.cos(theta) - d2 * np.sin(theta))
        c23 = mb * theta_dot * (d1 * np.sin(theta) + d2 * np.cos(theta))
        C = np.zeros((3,3))
        C[0,2], C[1,2] = c13, c23

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
        Phi_right_mat[2,2] = 1

        Phi_right_vector   = np.asarray([xdot, ydot, theta_dot])

        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left_mat.dot(Phi_right_mat).dot(Phi_right_vector)

        # store away future variables dot
        self.Phi_dot = Phi_dot
        self.alpha   = alpha
        self.l = l

        S = np.diag(np.sign(Phi_dot).squeeze())

        Dynamics = namedtuple('Dynamics', ['M', 'C', 'B', 'S', 'f', 'r', 'qaccel', 'qvel', 'q'],
                                verbose=False)
        body_dynamics = Dynamics(M, C, B, S, f, r, qdotdot, qdot, q)

        return body_dynamics

    def expand_array(self, array, dim):
        return np.expand_dims(array, axis=dim)


    def get_samples(self, noisy=False):
        """
            Get N samples of states and controls from the robot
        """

        T   = self.T
        dU  = self.dU
        dX  = self.dX

        traj_info =  TrajectoryInfo(config)
        cost_info =  CostInfo(config)

        # see generalized ILQG Summary page
        k = range(0, T)
        K = len(k)
        delta_t = T/(K-1)
        t = [(_+1-1)*delta_t for _ in k]

        # for n in range(N):

        # allocate space for local controls
        u       = np.zeros((T, dU))
        x       = np.zeros((T, dX))

        delta_u = np.zeros_like(u)
        delta_x = np.zeros_like(x)

        u_bar   = np.zeros_like(u) #np.random.randint(low=1, high=10, size=(T, dU))  #
        # initialize u_bar
        u_bar[:,] = config['trajectory']['init_action']
        x_bar   = np.zeros_like(x)

        fx      = np.zeros((T, dX, dX))
        fu      = np.zeros((T, dX, dU))

        x_noise = generate_noise(T, dX, self.agent)

        # costs
        l        = np.zeros((T))
        l_nom    = np.zeros((T))
        l_nlnr   = np.zeros((T))
        lu       = np.zeros((T, dU))
        lx       = np.zeros((T, dX))
        lxx      = np.zeros((T, dX, dX))
        luu      = np.zeros((T, dU, dU))
        lux      = np.zeros((T, dU, dX))

       # apply u_bar to deterministic system
        for k in range(K):
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
            Minv            = np.linalg.inv(M)
            rhs             = - Minv.dot(C).dot(x_bar[k,:]) - Minv.dot(B.T).dot(S.dot(f)) \
                              + Minv.dot(B.T).dot(u_bar[k, :])/self.wheel['radius']
            # print('rhs: ', rhs, 'xbar: ')
            if k == 0:
                x_bar[k]    = delta_t * rhs

            # step 2.1: get linearized dynamics
            BBT             = B.dot(B.T)
            Inv_BBT         = np.linalg.inv(BBT)
            lhs             = Inv_BBT.dot(self.wheel['radius'] * B)

            # step 2.2: set up nonlinear dynamics at k
            u[k,:]          = lhs.dot(M.dot(qaccel) + C.dot(qvel) + \
                                    B.T.dot(S).dot(f)).squeeze()

            # inject noise to the states
            x[k,:]          = q + x_noise[k, :] if noisy else q
            rhs             = - Minv.dot(C).dot(x[k,:]) - Minv.dot(B.T).dot(S.dot(f)) \
                                 + Minv.dot(B.T).dot(u[k,:])/self.wheel['radius']

            if k < K-1:
                x_bar[k+1,:]    = x_bar[k,:] +  delta_t * rhs
                x[k+1,:]        = x[k,:] +  delta_t * rhs

            # calculate the jacobians
            fx[k,:,:]   = np.eye(self.dX) - delta_t * Minv.dot(C)
            fu[k,:,:]   = -(delta_t * self.wheel['radius']) * Minv.dot(B.T)

            #step 2.3 set up state-action deviations
            delta_x[k,:]     = x[k,:] - x_bar[k,:]
            delta_u[k,:]     = u[k,:] - u_bar[k,:]
            delta_x_star     = self.goal_state

            # calculate cost-to-go and derivatives of nlnr stage_cost
            u_exp = self.expand_array(u[k,:], 1)
            x_exp = self.expand_array(x[k,:]-delta_x_star, 1)         
            cost_action_nlnr_term = 0.5 * self.action_penalty[0] * u_exp.T.dot(u_exp)
            cost_state_nlnr_term  = 0.5 * self.state_penalty[0]  * x_exp.T.dot(x_exp)
            cost_l12_nlnr_term    = np.sqrt(self.l21_const + (x_exp - delta_x_star).T.dot(x_exp - delta_x_star))

            # calculate cost-to-go and derivatives of stage_cost
            delu_exp = self.expand_array(delta_u[k,:], 1)
            delx_exp = self.expand_array(delta_x[k,:]-delta_x_star, 1)
            cost_action_term = 0.5 * self.action_penalty[0] * delu_exp.T.dot(delu_exp)
            cost_state_term  = 0.5 * self.state_penalty[0]  * delx_exp.T.dot(delx_exp)
            cost_l12_term    = np.sqrt(self.l21_const + (delx_exp - delta_x_star).T.dot(\
                                delx_exp - delta_x_star))


            # nominal cost terms
            ubar_exp = self.expand_array(u_bar[k,:], 1)
            xbar_exp = self.expand_array(x_bar[k,:] - delta_x_star, 1)
            cost_nom_action_term = 0.5 * self.action_penalty[0] * ubar_exp.T.dot(ubar_exp)
            cost_nom_state_term  = 0.5 * self.state_penalty[0] * xbar_exp.T.dot(xbar_exp)
            cost_nom_l12_term   = np.sqrt(self.l21_const + (xbar_exp \
                                    - delta_x_star).T.dot(xbar_exp \
                                    - delta_x_star))
            # cost_nom_l12_term    = np.sqrt(self.l21_const + self.expand_array((x_bar[k,:] - delta_x_star), 1).T.dot(self.expand_array((x_bar[k,:] - delta_x_star), 1))

            # define lx/lxx cost costants
            final_state_diff = (x[k,:] - self.goal_state)
            sqrt_term        = np.sqrt(self.l21_const + self.expand_array(final_state_diff, 1).T.dot(self.expand_array(final_state_diff, 1)))


            #system cost
            l[k]       = cost_action_term[0][0] + cost_state_term[0][0] + cost_l12_term[0][0]
            l_nom[k]   = cost_nom_action_term[0][0] + cost_nom_state_term[0][0] + cost_nom_l12_term[0][0]
            l_nlnr[k]  = cost_action_nlnr_term[0][0] + cost_state_nlnr_term[0][0] + cost_l12_nlnr_term[0][0]
            
            # first order cost terms
            lu[k] = self.action_penalty[0] * delta_u[k,:]
            lx[k] = self.state_penalty[0] * delx_exp.squeeze() \
                    + (final_state_diff/np.sqrt(self.l21_const + final_state_diff**2))

            # 2nd order cost terms
            luu[k]     = np.diag(self.action_penalty)
            lux[k]     = np.zeros((self.dU, self.dX))

            lxx_t1  = np.diag(self.state_penalty)
            lxx_t2_top = sqrt_term - (final_state_diff**2) /sqrt_term
            lxx_t2_bot = self.l21_const + (final_state_diff**2)

            lxx[k]        = lxx_t1 + lxx_t2_top/lxx_t2_bot
            # lxx[k]     = np.diag(self.state_penalty)

            # squeeze dims of first order derivatives
            lx[k] = lx[k].squeeze() if lx[k].ndim > 1 else lx[k]
            lu[k] = lu[k].squeeze() if lu[k].ndim > 1 else lu[k]

            # store away stage terms
            traj_info.fx[k,:]            = fx[k,:]
            traj_info.fu[k,:]            = fu[k,:]
            traj_info.action[k,:]        = u[k,:]
            traj_info.state[k,:]         = x[k,:]
            traj_info.delta_state[k,:]   = delta_x[k,:]
            traj_info.delta_action[k,:]  = delta_u[k,:]
            traj_info.nominal_state[k,:] = x_bar[k,:]
            traj_info.nominal_action[k,:]= u_bar[k,:]

            cost_info.l[k]   = l[k]
            cost_info.lu[k]  = lu[k]
            cost_info.lx[k]  = lx[k]
            cost_info.lux[k] = lux[k]
            cost_info.luu[k] = luu[k]
            cost_info.lxx[k] = lxx[k]
            cost_info.l_nom[k]   = l_nom[k]
            cost_info.l_nlnr[k]  = l_nlnr[k]

        sample = {'traj_info': traj_info, 'cost_info': cost_info}

        return sample
