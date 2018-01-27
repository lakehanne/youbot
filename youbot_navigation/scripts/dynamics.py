#!/usr/bin/env python
from __future__ import print_function

import os
import imp
import time
import copy
import argparse
import rospy
import scipy as sp
import logging
import threading
import numpy as np
from collections import namedtuple

import multiprocessing
from multiprocessing.pool import ThreadPool

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, \
        Pose, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion

import roslib
roslib.load_manifest('youbot_navigation')

# import sys
# print(sys.path)
from scripts.torque import MassMaker

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
                        help='max num iterations' )
parser.add_argument('--silent', '-si', action='store_true', default='False',
                        help='max num iterations' )
args = parser.parse_args()

print(args)
LOGGER = multiprocessing.log_to_stderr()
if args.silent:
    LOGGER.setLevel(logging.INFO)
else:
    LOGGER.setLevel(logging.DEBUG)

class ILQR(MassMaker):
    def __init__(self, arg, Odometry, hyperparams, rate=10):
        super(ILQR, self).__init__()
        """
        self.desired_pose = final pose that we would like the robot to reach
        self.current_pose = current pose of the robot as obtained from odometric info
        self.penalty = penalty incurred on control law
        """
        self.arg = arg
        self.rate = rate  # 10Hz

        self.hyperparams = hyperparams
        self.T = self.hyperparams['cost_params']['T']
        self.dU = self.hyperparams['cost_params']['dU']
        self.dX = self.hyperparams['cost_params']['dX']

        self.mat_maker = MassMaker()
        self.mat_maker.get_mass_matrices()
        self.body_dynamics = None

        self.__dict__.update(self.mat_maker.__dict__)

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
        mb = self.base['mass']
        mw = self.wheel['mass']
        r = self.wheel['radius']
        """
        I_b is the moment of inertia of the platform about Zr' axis thru G'
        I is the mom of inertia of ith wheel about main axis
        """
        I_b = self.base['mass_inertia'][-1,-1]
        I = self.wheel['mass_inertia'][1,1]

        # f_i are the frictional force of each of the four wheels
        f = np.array([
                      self.wheel['friction'], self.wheel['friction'],
                      self.wheel['friction'], self.wheel['friction']
                     ])
        f = np.expand_dims(f, axis=1)
        base_footprint_dim = 0.001  # retrieved from the box geom in gazebo
        l = np.sqrt(2* base_footprint_dim)
        l_sqr = 2* base_footprint_dim
        b, a = 0.19, 0.145 # meters as measured from the real robot
        alpha = np.arctan2(b, a)

        x     = self.odom.pose.pose.position.x
        y     = self.odom.pose.pose.position.y

        # note that i have switched the ordering of qx and qy due to the different conventions in ros and the paper
        quaternion = [self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.y,
                        self.odom.pose.pose.orientation.x,self.odom.pose.pose.orientation.z]
        _, _, theta = euler_from_quaternion(quaternion, axes='sxyz')
        # theta -= np.pi/2.0  # account for diffs of frames in gazebo and paper

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y

        quaternion_dot = [1.0, self.odom.twist.twist.angular.y,
                        self.odom.twist.twist.angular.x, self.odom.twist.twist.angular.z]
        _, _, theta_dot = euler_from_quaternion(quaternion_dot, axes='sxyz')
        # theta_dot -= np.pi/2.0  # account for diffs of frames in gazebo and paper  

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

        xaccel= xdot/time_delta
        yaccel = ydot/time_delta
        theta_accel = theta_dot/time_delta

        LOGGER.debug("Time between motions: {}".format(time_delta))

        q       = np.expand_dims(np.array([x, y, theta]), axis=1)
        qdot    = np.expand_dims(np.array([xdot, ydot, theta_dot]), axis=1)
        qdotdot = np.expand_dims(np.array([xaccel, yaccel, theta_accel]), axis=1)

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

        Phi_right_vector   = np.expand_dims(
                                np.asarray([xdot, ydot, theta_dot]),
                                axis=1)
        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left_mat.dot(Phi_right_mat).dot(Phi_right_vector)

        LOGGER.debug('Phi dot: {}, \n sign(Phi dot) {}'.format(Phi_dot.T, np.sign(Phi_dot).T))
        S = np.diag(np.sign(Phi_dot).squeeze())

        Dynamics = namedtuple('Dynamics', ['M', 'C', 'B', 'S', 'f', 'r', 'qaccel', 'qvel', 'q'], 
                                verbose=False)
        body_dynamics = Dynamics(M, C, B, S, f, r, qdotdot, qdot, q)

        return body_dynamics

    def do_traj_opt(self):
        T = self.hyperparams['cost_params']['T']
        U = self.hyperparams['cost_params']['U']

        # assemble Jacobian of transfer matrix and et cetera
        u          = np.zeros((T, dU, 1))
        u_bar      = np.zeros((T, dU, 1))
        u_delta    = np.zeros((T, dU, 1))

        state      = np.zeros((T, dX, 1))
        state_bar  = np.zeros((T, dX, 1)) 
        state_delta= np.zeros((T, dX, 1))  

        state_star = np.zeros((T, dX, 1)) # desired state
        fx         = np.zeros((T, dX, 1))
        fu         = np.zeros((T, dU, 1))

        stagecost_penalty    = self.hyperparams['cost_params']['penalty']  # this is the r term in the objective function
        
        # assemble stage_costs
        stage_costs     = np.zeros((T, 1))  # stage_costs
        stage_actions   = np.zeros((T, dU, 1))        
        wheel_radius    = self.wheel['radius']

        # Allocate.
        Vxx = np.zeros((T, dX, dX))
        Vx = np.zeros((T, dX))
        Qtt = np.zeros((T, dX+dU, dX+dU))
        Qt = np.zeros((T, dX+dU))

        for t in range (T-1, -1, -1):
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

            # calculate inverse dynamics equation
            BBT             = B_matrix.dot(B_matrix.T)
            Inv_BBT         = np.linalg.inv(BBT)
            multiplier      = Inv_BBT.dot(wheel_radius * B_matrix)
            inv_dyn_eq      = mass_matrix.dot(qaccel) + coriolis_matrix.dot(qvel) + \
                                    B_matrix.T.dot(S_matrix).dot(friction_vector)
            torque_vector   = multiplier.dot(inv_dyn_eq)

            # set up costs at time T
            u[t,:,:]        = torque_vector
            u_bar[t,:,:]    = 0 # assume u_bar is zero
            state[T,:]       = np.array([[q, qvel]]).T
            state_star[T,:]  = copy.copy(self.hyperparams['goal_state'])
            fx[T,:]          = -np.linalg.inv(mass_matrix).dot(coriolis_matrix)
            fu[T,:]          = -(1/wheel_radius) * np.linalg.inv(mass_matrix).dot(B_matrix.T) 

            # retrieve this from the inverse dynamics equation (20) in Todorov ILQG paper
            # stage_costs[t, :] = self.penalty * action.T.dot(action)
        

        self.final_cost = 0.5 * (diff).T.dot(diff)


if __name__ == '__main__':

    from scripts import __file__ as scripts_filepath
    scripts_filepath = os.path.abspath(scripts_filepath)
    scripts_dir = '/'.join(str.split(scripts_filepath, '/')[:-1]) + '/'
    hyperparams_file = scripts_dir + 'config.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)


    try:
        ilqr = ILQR(args, Odometry, hyperparams.config, rate=10)
        rospy.init_node('Listener')
        pool = ThreadPool(processes=2) # start 2 worker processes

        while not rospy.is_shutdown():
            """
            generate_dynamics = threading.Thread(
            target=lambda: ilqr.do_traj_opt()
            )
            generate_dynamics.daemon = True
            generate_dynamics.start()
            """

            ilqr.do_traj_opt()


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
