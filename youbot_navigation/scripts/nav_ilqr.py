#!/usr/bin/env python
from __future__ import print_function
# computes the ilqr control law to move robot to target

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


# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# # logging.setLevel(logging.DEBUG)
# LOGGER = logging.getLogger(__name__)

LOGGER = multiprocessing.log_to_stderr()
if args.silent:
    LOGGER.setLevel(logging.INFO)
else:
    LOGGER.setLevel(logging.DEBUG)

class ILQR(MassMaker):
    """
    self.desired_pose = final pose that we would like the robot to reach
    self.current_pose = current pose of the robot as obtained from odometric info
    self.penalty = penalty incurred on control law

    """
    def __init__(self, arg, Odometry, rate=10):
        super(ILQR, self).__init__()

        self.arg = arg
        self.dU = 7 # dimension of robot pose
        self.rate = rate  # 10Hz

        self.desired_pose = np.asarray([
                        0.736239556594,
                        0.803042138661,
                        0.0,
                        -3.18874459879e-06,
                        -1.08345476892e-05,
                        0.0432270282388,
                        -0.999065275096,
                        ])

        self.desired_pose = np.expand_dims(self.desired_pose, axis=1)
        self.penalty = 0.0001  # this is the r term in the objective function
        self.current_pose = None

        self.mat_maker = MassMaker()
        self.mat_maker.get_mass_matrices()
        self.__dict__.update(self.mat_maker.__dict__)

    def odom_cb(self, odom):
        # convert current and desired to numpy arrays
        # rospy.loginfo(odom.twist.twist)
        self.odom = odom
        self.current_pose = np.asarray([odom.pose.pose.position.x,
                                        odom.pose.pose.position.y,
                                        odom.pose.pose.position.z,
                                        odom.pose.pose.orientation.x,
                                        odom.pose.pose.orientation.y,
                                        odom.pose.pose.orientation.z,
                                        odom.pose.pose.orientation.w])
        self.current_pose = np.expand_dims(self.current_pose, axis=1)


    def listener(self):
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        # print('desired_pose: ', self.desired_pose)
        # print('current_pose: ', self.current_pose)
        sleeper = rospy.Rate(self.rate)
        sleeper.sleep()

    def set_up_cost(self):

        diff = self.current_pose - self.desired_pose

        # compute stage_costs
        stage_costs = np.zeros([T-1, self.dU])  # stage_costs
        action   = np.zeros([self.dU])

        """
        for t in range (self.T-1, -1, -1):
            action  =   # retrieve this from the ib=nverse dynamics equation (20) in Todorov ILQG paper
            stage_costs[t, :] = self.penalty * action.T.dot(action)
        """


        self.final_cost = 0.5 * (diff).T.dot(diff)

    def assemble_dynamics(self):
        prev_time = rospy.get_rostime().to_sec()
        self.listener()

        # gather dynamics equation components
        mb = self.base['mass']
        mw = self.wheel['mass']
        r = self.wheel['radius']
        """
        I_b is the moment of inertia of the platform about Zr' axis thru G'
        I is the mom of inertia of ith wheel about main axis
        """
        I_b = self.base['mass_inertia'][-1,-1]
        I = self.wheel['mass_inertia'][-1,-1]

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

        """
        theta is something that we obtain from the wheel's odometry information
        theta is the rotation about the Z_R axis
        """
        quaternion = [self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x,
                        self.odom.pose.pose.orientation.y,self.odom.pose.pose.orientation.z]
        _, _, theta = euler_from_quaternion(quaternion, axes='sxyz')
        d1, d2 = 1e-2, 1e-2 # not sure of this
        alpha = np.arctan2(b, a)

        # define mass matrix elements
        m11 = mb + 4 * (mw + I/(r**2))
        m22 = mb + 4 * (mw + I/(r**2))
        m13 = mb * ( d1 * np.sin(theta) + d2 * np.cos(theta) )
        m23 = mb * (-d1 * np.cos(theta) + d2 * np.sin(theta) )
        m33 = mb * (d1 ** 2 + d2 ** 2) + I_b + \
                    8 * (mw + I/(r**2)) * l_sqr * pow(np.sin(np.pi/4.0 - alpha), 2)

        # assemble mass inertia matrix
        M = np.zeros((3,3))
        M[0,0], M[1,1], M[2,2] = m11, m22, m33
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

        # time for accelaeration calculation
        now_time  = rospy.get_rostime().to_sec()
        time_delta = now_time - prev_time

        x     = self.odom.pose.pose.position.x
        y     = self.odom.pose.pose.position.y
        theta = self.odom.pose.pose.orientation.z

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        xdotdot = xdot/time_delta
        ydotdot = ydot/time_delta
        theta_dotdot = theta_dot/time_delta

        LOGGER.warn("Time between motions: {}".format(time_delta))

        q       = np.expand_dims(np.array([x, y, theta]), axis=1)
        qdot    = np.expand_dims(np.array([xdot, ydot, theta_dot]), axis=1)
        qdotdot = np.expand_dims(np.array([xdotdot, ydotdot, theta_dotdot]), axis=1)


        # C matrix components
        c13 = mb * theta_dot * (d1 * np.cos(theta) - d2 * np.sin(theta))
        c23 = mb * theta_dot * (d1 * np.sin(theta) - d2 * np.cos(theta))
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
        Phi_right_mat[1,1] = np.cos(theta)

        Phi_right_vector   = np.expand_dims(
                                np.asarray([xdot, ydot, theta_dot]),
                                axis=1)
        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left_mat.dot(Phi_right_mat).dot(Phi_right_vector)

        LOGGER.debug('Phi dot: {}, \n sign(Phi dot) {}'.format(Phi_dot.T, np.sign(Phi_dot).T))
        S = np.diag(np.sign(Phi_dot))


        # Dynamics = namedtuple('Dynamics', ['M', 'C', 'B', 'S', 'f', 'r'], verbose=False)
        # self.body_dynamics = Dynamics(M, C, B, S, f, r)
        Dynamics = namedtuple('Dynamics', ['M', 'C', 'B', 'S', 'f', 'r' 'qaccel', 'qdot', 'q'], verbose=False)
        self.body_dynamics = Dynamics(M, C, B, S, f, r, qdotdot, qdot, q)

        # LOGGER.debug(self.body_dynamics.B)

    def get_wheel_torques(self):

        self.assemble_dynamics() # get body dynamics

        mass_matrix     = self.body_dynamics.M
        coriolis_matrix = self.body_dynamics.C
        wheel_radius    = self.body_dynamics.r
        B_matrix        = self.body_dynamics.B
        S_matrix        = self.body_dynamics.S
        friction_vector = self.body_dynamics.f
        qaccel          = self.body_dynamics.qaccel
        qvel            = self.body_dynamics.qvel
        qaccel          = self.body_dynamics.q


        BBT = B_matrix.dot(B_matrix.T)
        U = np.linalg.cholesky(BBT)
        L = U.T
        Inv_BBT =  sp.linalg.solve_triangular(
                                U, sp.linalg.solve_triangular(L, BBT, lower=True)
                                )

        multiplier          = wheel_radius * Inv_BBT.dot(B_matrix)
        lagrangian_eq       = mass_matrix.dot(qaccel) + coriolis_matrix.dot(qvel) + \
                            B.T.dot(S_matrix).dot(friction_vector)
        self.torque_vector  = multiplier.dot(lagrangian_eq)

        LOGGER.debug('torque vector: {}'.format(self.torque_vector))


if __name__ == '__main__':

    ilqr = ILQR(args, Odometry, rate=2)

    try:
        rospy.init_node('Listener')
        pool = ThreadPool(processes=2) # start 2 worker processes

        while not rospy.is_shutdown():
            generate_dynamics = ilqr.get_wheel_torques()


            # generate_dynamics = threading.Thread(
            # target=lambda: ilqr.get_wheel_torques()
            # )
            # generate_dynamics.daemon = True
            # generate_dynamics.start()

            # LOGGER.debug('system dynamics: {}'.format(generate_dynamics))

    except KeyboardInterrupt:
        LOGGER.debug("shutting down ros")
