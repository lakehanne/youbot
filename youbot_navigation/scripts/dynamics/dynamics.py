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
import matplotlib.pyplot as plt

from scripts.dynamics.constants import MassMaker
from scripts.algorithm_utils import TrajectoryInfo, \
                            generate_noise, CostInfo
from scripts.subscribers import KinectReceiver, \
                                ModelStatesReceiver
from scripts.costs import CostSum, CostAction, CostState                              

from multiprocessing.pool import ThreadPool
from geometry_msgs.msg import Twist, \
        Pose, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion

import scipy.ndimage as sp_ndimage
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
        self.dV               = config['agent']['dV']
        self.dX               = config['agent']['dX']
        self.goal_state       = None #config['agent']['goal_state']

        self.__dict__.update(mat_maker.__dict__)
        self.model_rcvr       = ModelStatesReceiver()

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
        quaternion = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                        self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
        _, _, theta  = euler_from_quaternion(quaternion, axes='sxyz')
        # theta is the diff in angle between Y_R and Y_I

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        d1, d2 = 0, 0 # set to zeros

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
        B[1,1] = np.cos(theta) - np.sin(theta)
        B[3,1] = np.sin(theta) - np.cos(theta)

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
        Phi_left = np.ones((4, 3))
        Phi_left[:,:2].fill(np.sqrt(2)/2)
        Phi_left[:,2].fill(l*np.sin(np.pi/4 - alpha))
        # column 0
        Phi_left[1, 0] *= -1
        Phi_left[2, 0] *= -1
        # column 1
        Phi_left[2, 1] *= -1
        Phi_left[3, 1] *= -1

        Phi_right = np.zeros((3,3))
        Phi_right[0,0] = -np.sin(theta)
        Phi_right[1,1] = np.sin(theta)

        Phi_right[0,1] = np.cos(theta)
        Phi_right[1,0] = np.cos(theta)
        Phi_right[2,2] = 1

        zeta               = np.asarray([xdot, ydot, theta_dot])

        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left.dot(Phi_right).dot(zeta)

        # store away future variables dot
        self.Phi_dot = Phi_dot
        self.alpha   = alpha
        self.l = l

        S = np.diag(np.sign(Phi_dot).squeeze())

        Dynamics = namedtuple('Dynamics', ['M', 'C', 'B', 'S', 'f', 'r', 'qaccel', 'qvel', 'q'],
                                verbose=False)
        body_dynamics = Dynamics(M, C, B, S, f, r, qdotdot, qdot, q)

        return body_dynamics

    def exp_arr(self, array, dim):
        return np.expand_dims(array, axis=dim)

    def get_disturbance(T, dU, gauss=True, rand_walk=True):
        if T != 1:
            assert "The disturbance function does not support multidim T as yet"
        noise = np.random.randn(T, dU)
        if gauss:
            gauss_var = 2
            noise = sp_ndimage.filters.gaussian_filter(noise, gauss_var)
        if rand_walk:
            noise_cum = np.cumsum(noise, axis=1)
            noise = noise_cum**2
        return noise

    def get_samples(self, noisy=False):
        """
            Get N samples of states and controls from the robot
        """
        T   = self.T
        dU  = self.dU
        dV  = self.dV
        dX  = self.dX

        traj_info =  TrajectoryInfo(config)
        cost_info =  CostInfo(config)

        cost_sum = CostSum(config)  

        wu       = config['all_costs']['wu']
        wv       = config['all_costs']['wv']
        wx       = config['all_costs']['wx']
        gamma    = config['all_costs']['gamma']

        ms       = self.model_rcvr.model_states
        pose     = ms.pose

        boxtacle_pose  = [pose[-1].position.x, pose[-1].position.y, pose[-1].position.z,
                         pose[-1].orientation.x, pose[-1].orientation.y, pose[-1].orientation.z,
                         pose[-1].orientation.w]
        _, _, robot_angle = euler_from_quaternion(boxtacle_pose[3:])
        self.goal_state  = np.array(boxtacle_pose[:2] + [robot_angle])
        self.goal_state[0] -= 0.456 # account for box origin so robot doesn't push box when we get to final time step

        # see generalized ILQG Summary page
        k       = range(0, T)
        K       = len(k)
        delta_t = T/(K-1)
        t       = [(_+1-1)*delta_t for _ in k]

        # allocate space for local controls
        u       = np.zeros((T, dU))
        v       = np.zeros((T, dV))
        x       = np.zeros((T, dX))

        del_u   = np.zeros_like(u)
        del_v   = np.zeros_like(v)
        del_x   = np.zeros_like(x)

        u_bar   = np.zeros_like(u) #np.random.randint(low=1, high=10, size=(T, dU))  #
        u_bar[:,] = config['trajectory']['init_action']
        v_bar   = generate_noise(T, dV, self.agent)
        v       = generate_noise(T, dV, self.agent)
        # initialize u_bar
        x_bar   = np.zeros_like(x)

        fx      = np.zeros((T, dX, dX))
        fu      = np.zeros((T, dX, dU))
        fv      = np.zeros((T, dX, dV))

        x_noise = generate_noise(T, dX, self.agent)

        # costs
        l        = np.zeros((T))
        l_nom    = np.zeros((T))
        l_nlnr   = np.zeros((T))
        lu       = np.zeros((T, dU))
        lv       = np.zeros((T, dV))
        lx       = np.zeros((T, dX))
        lxx      = np.zeros((T, dX, dX))
        lvv      = np.zeros((T, dV, dV))
        lvx      = np.zeros((T, dV, dX))
        lux      = np.zeros((T, dU, dX))
        luu      = np.zeros((T, dU, dU))
        luv      = np.zeros((T, dU, dV))

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
            Minv        = np.linalg.inv(M)
            rhs         = - Minv.dot(C).dot(x_bar[k,:]) - Minv.dot(B.T).dot(S.dot(f)) \
                              + Minv.dot(B.T).dot(u_bar[k, :] n- gamma * v_bar)/self.wheel['radius']                            
            
            if k == 0:
                x_bar[k]= delta_t * rhs

            if k < K - 1:  
                x_bar[k+1,:]= x_bar[k,:] +  delta_t * rhs

            # step 2.1: get linearized dynamics
            BBT         = B.dot(B.T)
            Inv_BBT     = np.linalg.inv(BBT)
            lhs         = Inv_BBT.dot(self.wheel['radius'] * B)

            # step 2.2: set up nonlinear dynamics at k
            u[k,:]      = lhs.dot(M.dot(qaccel) + C.dot(qvel) + \
                                    B.T.dot(S).dot(f)).squeeze() + gamma * v[k,:]
            # this is already initialized to a zero mean var 2 rand walk vector
            # v[k,:]      = u[k,:] #+ v_bar[k]#generate_noise(1, dV, self.agent)

            # inject noise to the states
            x[k,:]      = q + x_noise[k, :] if noisy else q
            rhs         = - Minv.dot(C).dot(x[k,:]) - Minv.dot(B.T).dot(S).dot(f) \
                                 + Minv.dot(B.T).dot(u[k,:] - gamma * v[k,:])/self.wheel['radius']

            if k < K-1:
                x[k+1,:]        = x[k,:] +  delta_t * rhs

            # calculate the jacobians
            fx[k,:,:]   = np.eye(self.dX) - delta_t * Minv.dot(C)
            fu[k,:,:]   = (delta_t * self.wheel['radius']) * Minv.dot(B.T)
            fv[k,:,:]   = -(delta_t * gamma * self.wheel['radius']) \
                             * Minv.dot(B.T)

            #step 2.3 set up state-action deviations
            del_x[k,:]     = x[k,:] - x_bar[k,:]
            del_u[k,:]     = u[k,:] - u_bar[k,:]
            del_v[k,:]     = v[k,:] - v_bar[k,:]
            del_x_star     = self.goal_state
               
            # get nominal state costs 
            l_nom[k] = cost_sum.eval(x=x_bar[k,:], xstar=del_x_star, \
                                     u=u_bar[k,:], v=v_bar[k,:])[0]

            # get nonlinear state cost derivatives
            l_nlnr[k], lx[k], lu[k], lv[k], lux[k], lvx[k], lxx[k], \
            luu[k], luv[k], lvv[k] = cost_sum.eval(x=x[k,:], xstar=del_x_star, \
                                                 u=u_bar[k,:], v=v_bar[k,:])
            
            # form l approximation # eq 4 in iros 18 paper
            left_mat  = np.c_[1, self.exp_arr(del_x[k], 1).T, \
                                self.exp_arr(del_u[k], 1).T, self.exp_arr(del_v[k], 1).T]               
            inner_mat = np.r_[
                            np.c_[l_nom[k], self.exp_arr(lx[k], 1).T, self.exp_arr(lu[k], 1).T, self.exp_arr(lv[k], 1).T],
                            np.c_[self.exp_arr(lx[k], 1), lxx[k], lux[k].T, lvx[k].T],
                            np.c_[self.exp_arr(lu[k], 1), lux[k], luu[k],   luv[k]],
                            np.c_[self.exp_arr(lv[k], 1), lvx[k], luv[k].T, lvv[k]]
                        ]
            right_mat  = left_mat.T    

            l[k]   = 0.5 * left_mat.dot(inner_mat).dot(right_mat)#.squeeze()

            # store away stage terms
            traj_info.fx[k,:]            = fx[k,:]
            traj_info.fu[k,:]            = fu[k,:]
            traj_info.fv[k,:]            = fv[k,:]
            traj_info.action[k,:]        = u[k,:]
            traj_info.act_adv[k,:]       = v[k,:]
            traj_info.state[k,:]         = x[k,:]
            traj_info.nom_state[k,:]     = x_bar[k,:]
            traj_info.nom_action[k,:]    = u_bar[k,:]
            traj_info.nom_act_adv[k,:]   = v_bar[k,:]
            traj_info.delta_action[k,:]  = del_u[k,:]
            traj_info.delta_act_adv[k,:] = del_v[k,:]
            traj_info.delta_state[k,:]   = del_x[k,:]

            cost_info.l[k]   = l[k]
            cost_info.lu[k]  = lu[k]
            cost_info.lv[k]  = lv[k]
            cost_info.lx[k]  = lx[k]
            cost_info.lux[k] = lux[k]
            cost_info.lvx[k] = lvx[k]
            cost_info.luu[k] = luu[k]
            cost_info.luv[k] = luv[k]
            cost_info.lvv[k] = lvv[k]
            cost_info.lxx[k] = lxx[k]
            cost_info.l_nom[k]   = l_nom[k]
            cost_info.l_nlnr[k]  = l_nlnr[k]

        sample = {'traj_info': traj_info, 'cost_info': cost_info}

        return sample
