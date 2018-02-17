#!/usr/bin/env python
from __future__ import print_function
import rospy
import logging
import threading

from rospy.rostime import Duration
from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortResponse,\
							 ApplyBodyWrench, ApplyBodyWrenchResponse, \
							 JointRequest, BodyRequest
							 
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist

LOGGER = logging.getLogger(__name__)

def send_joint_torques(*msg):	
	rospy.wait_for_service('/gazebo/apply_joint_effort')
	try:
		send_torque = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
		rospy.loginfo('sending: {}Nm to {}'.format(msg[1], msg[0]))
		resp = send_torque(msg[0], msg[1], msg[2], msg[3])

		return ApplyJointEffortResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		LOGGER.debug("Service call failed: %s"%e)

def clear_active_forces(joint_name):	
	rospy.wait_for_service('/gazebo/clear_joint_forces')
	try:
		clear_forces = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
		LOGGER.debug('clearing forces on joint: '.format(joint))
		resp = clear_forces(joint_name)

		return JointRequestResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		LOGGER.debug("Service call failed: %s"%e)


def send_body_wrench(*msg):	
	rospy.wait_for_service('/gazebo/apply_body_wrench')
	try:
		send_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
		resp = send_wrench(msg[0], msg[1], msg[2], msg[3], msg[4], msg[5])

		return ApplyBodyWrenchResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		rospy.loginfo("Service call failed: %s"%e)		

def clear_active_wrenches(body_name):	
	rospy.wait_for_service('/gazebo/clear_body_wrenches')
	try:
		clear_forces = rospy.ServiceProxy('/gazebo/clear_body_wrenches', BodyRequest)
		rospy.logdebug('clearing wrenches on link: '.format(body_name))
		resp = clear_forces(body_name)

		# return BodyRequestResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		rospy.loginfo("Service call failed: %s"%e)
