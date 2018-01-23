#!/usr/bin/env python

# computes the ilqr control law to move robot to target

import argparse
import rospy
import numpy as np

from geometry_msgs.msg import Twist, Pose, PoseStamped


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

	def __init__(self, arg, Twist, Pose, PoseStamped):
		super(ILQR, self).__init__()
		self.arg = arg
		self.dU = 7 # dimension of robot pose

		desired_pose = PoseStamped

	    desired_pose.target_pose.header.frame_id = "base_link";
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
		current_pose = pose_stamped
     	# convert current and desired to numpy arrays
     	self.current_pose = np.asarray([current_pose.target_pose.pose.position.x,
     							   		current_pose.target_pose.pose.position.y,
     							   		current_pose.target_pose.pose.position.z,
     							   		current_pose.target_pose.pose.orientation.x,
     							   		current_pose.target_pose.pose.orientation.y,
     							   		current_pose.target_pose.pose.orientation.z,
     							   		current_pose.target_pose.pose.orientation.w])
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

ilqr = ILQR(args, Twist, Pose, PoseStamped)

while not rospy.is_shutdown():
	listener()