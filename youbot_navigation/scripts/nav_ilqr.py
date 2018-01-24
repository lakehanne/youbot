#!/usr/bin/env python

# computes the ilqr control law to move robot to target

import argparse
import rospy
import MassMaker
import threading
import numpy as np

from geometry_msgs.msg import Twist, \
		Pose, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion		


parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
						help='max num iterations' )
args = parser.parse_args()

class ILQR(object):
	"""
	self.desired_pose = final pose that we would like the robot to reach
	self.current_pose = current pose of the robot as obtained from odometric info
	self.penalty = penalty incurred on control law

	"""

	def __init__(self, arg):
		super(ILQR, self).__init__()
		self.arg = arg
		self.dU = 7 # dimension of robot pose

		desired_pose = PoseStampedtheta
	    desired_pose.target_pose.header.frame_id = "base_link"
	    desired_pose.target_pose.header.stamp = rospy.Time.now();
	    desired_pose.target_pose.pose.position.x = 0.736239556594; #//0.00;
	    desired_pose.target_pose.pose.position.y = 0.803042138661; #//0.000001;
	    desired_pose.target_pose.pose.position.z = 0.0;  
	    desired_pose.target_pose.pose.orientation.x =  -3.18874459879e-06;
	    desired_pose.target_pose.pose.orientation.y = -1.08345476892e-05;
	    desired_pose.target_pose.pose.orientation.z = 0.0432270282388;
	    desired_pose.target_pose.pose.orientation.w = -0.999065275096;

	    self.desired_pose = np.asarray([
					    desired_pose.target_pose.pose.position.x;
					    desired_pose.target_pose.pose.position.y;
					    desired_pose.target_pose.pose.position.z;
					    desired_pose.target_pose.pose.orientation.x;
					    desired_pose.target_pose.pose.orientation.y;
					    desired_pose.target_pose.pose.orientation.z;
					    desired_pose.target_pose.pose.orientation.w;
					    ])

	   	self.desired_pose = np.expand_dims(self.desired_pose, axis=1)
	   	self.penalty = 0.0001

	def odom_cb(self, pose_stamped):
		self.youbot_orientation = pose_stamped.target_pose.pose.orientation
     	# convert current and desired to numpy arrays
     	self.current_pose = np.asarray([pose_stamped.target_pose.pose.position.x,
     							   		pose_stamped.target_pose.pose.position.y,
     							   		pose_stamped.target_pose.pose.position.z,
     							   		pose_stamped.target_pose.pose.orientation.x,
     							   		pose_stamped.target_pose.pose.orientation.y,
     							   		pose_stamped.target_pose.pose.orientation.z,
     							   		pose_stamped.target_pose.pose.orientation.w])
     	self.current_pose = np.expand_dims(self.current_pose, axis=1)


	def listener(self):
	    rospy.init_node('Listener')      	
      	rospy.Subscriber('/odom', PoseStamped, self.odom_cb)

     def set_up_cost(self):

     	diff = self.current_pose - self.desired_pose

     	# compute stage_costs
     	stage_costs = np.zeros([T-1, self.dU])  # stage_costs
     	action   = np.zeros([self.dU])

     	for t in range (self.T-1, -1, -1):
     		action  =   # retrieve this from the ib=nverse dynamics equation (20) in Todorov ILQG paper
	     	stage_costs[t, :] = self.penalty * action.T.dot(action)


     	self.final_cost = 0.5 * (diff).T.dot(diff)


     def assemble_dynamics(self):     	
		mat_maker = MassMaker()
		mat_maker.get_mass_matrices()

		mb = mat_maker.base['mass']
		mw = mat_maker.wheel['mass']
		r = mat_maker.wheel['radius']
		"""
		I_b is the moment of inertia of the platform about Zr' axis thru G'
		I is the mom of inertia of ith wheel about main axis
		"""
		I_b = mat_maker.base['mass_inertia'][-1,-1]
		I = mat_maker.wheel['mass_inertia'][-1,-1]

		# f_i are the frictional force of each of the four wheels
		f = np.array([
		              mat_maker.wheel['friction'], mat_maker.wheel['friction'],
		              mat_maker.wheel['friction'], mat_maker.wheel['friction']
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
		_, _, theta = euler_from_quaternion(self.youbot_orientation)
		d1, d2 = 1e-2, 1e-2 # not sure of this
		alpha = np.arctan2(b, a)
		# define matrix elements
		m11 = mb + 4 * (mw + I/(r*r))
		m22 = mb + 4 * (mw + I/(r*r))
		m13 = mb * ( d1 * np.sin(theta) + d2 * np.sin(theta) )
		m23 = mb * (-d1 * np.cos(theta) + d2 * np.sin(theta) )
		m33 = mb * (pow(d1, 2) + pow(d2, 2)) + I_b + \
					8 * (mw + I/pow(r,2)) * l_sqr * pow(np.sin(pi/4.0 - alpha), 2)

		# assemble mass inertia matrix
		M = np.zeros((3,3))
		M[0,0], M[1,1], M[2,2] = m11, m22, m33
		M[0,2], M[2,0], M[1,2], M[2,1] = m13, m13, m23, m23



if __name__ == '__main__':
	ilqr = ILQR(args)

	while not rospy.is_shutdown():
		ilqr.listener()

	    generate_matrices = threading.Thread(
	    target=lambda: ilqr.assemble_dynamics()
	    )
	    generate_matrices.daemon = True
	    generate_matrices.start()