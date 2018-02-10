#!/usr/bin/env python
from __future__ import print_function

import rospy
import logging
import numpy as np
import multiprocessing
from nav_msgs.msg import Odometry
from collections import namedtuple

from scripts.constants import MassMaker
from multiprocessing.pool import ThreadPool
from geometry_msgs.msg import Twist, \
        Pose, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion

LOGGER = logging.getLogger(__name__)

class Dynamics(MassMaker):
    def __init__(self, Odometry, rate=10):
        super(Dynamics, self).__init__()
        self.rate = rate  # 10Hz

        mat_maker = MassMaker()
        mat_maker.get_mass_matrices()

        self.__dict__.update(mat_maker.__dict__)

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
        I = self.wheel['mass_inertia'][1,1]

        # f_i are the frictional force of each of the four wheels
        f = np.array([
                      self.wheel['friction'], self.wheel['friction'],
                      self.wheel['friction'], self.wheel['friction']
                     ])
        base_footprint_dim = 0.001  # retrieved from the box geom in gazebo
        l = np.sqrt(2* base_footprint_dim)
        l_sqr = 2* base_footprint_dim
        b, a = 0.19, 0.145 # meters as measured from the real robot
        alpha = np.arctan2(b, a)

        x     = self.odom.pose.pose.position.x
        y     = self.odom.pose.pose.position.y        

        # note that i have switched the ordering of qx and qy due to the different conventions in gazebo and the paper
        quaternion = [self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.y,
                        self.odom.pose.pose.orientation.x,self.odom.pose.pose.orientation.z]
        _, _, theta = euler_from_quaternion(quaternion, axes='sxyz')
        # theta -= np.pi/2.0  # account for diffs of frames in gazebo and paper

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        d1, d2 = 0, 0 #1e-2, 1e-2 # not sure of this

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

