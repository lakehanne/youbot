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


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def send_joint_torques(*msg):	
	rospy.wait_for_service('/gazebo/apply_joint_effort')
	try:
		send_torque = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
		rospy.loginfo('sending: {}Nm to {}'.format(msg[1], msg[0]))
		resp = send_torque(msg[0], msg[1], msg[2], msg[3])

		return ApplyJointEffortResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		print("Service call failed: %s"%e)

def clear_active_forces(joint_name):	
	rospy.wait_for_service('/gazebo/clear_joint_forces')
	try:
		clear_forces = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
		rospy.loginfo('clearing forces on joint: '.format(joint))
		resp = clear_forces(joint_name)

		return JointRequestResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		rospy.loginfo("Service call failed: %s"%e)


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
		rospy.loginfo('clearing wrenches on link: '.format(body_name))
		resp = clear_forces(joint_name)

		# return BodyRequestResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		rospy.loginfo("Service call failed: %s"%e)


if __name__ == "__main__":

	rospy.init_node('clients')

	joint_name = 'youbot::arm_joint_1'
	effort = 30.0
	start_time = Duration(secs = 0, nsecs = 0) # start asap	
	duration = Duration(secs = 5, nsecs = 0) # apply effort continuously without end duration = -1
	
	# response = send_joint_torques(joint_name, effort, start_time, duration)
	# response = send_joint_torques('wheel_joint_br', effort, start_time, duration)
	# response = send_joint_torques('wheel_joint_fl', effort, start_time, duration)
	# response = send_joint_torques('wheel_joint_fr', effort, start_time, duration)
	# LOGGER.debug('response: ', response)

	# ['body_name', 'reference_frame', 'reference_point', 'wrench', 'start_time', 'duration']
	wrench = Wrench()

	wrench.force.x = 40
	wrench.force.y = 40
	wrench.force.z = 40

	wrench.torque.x = 0
	wrench.torque.y = 0
	wrench.torque.y = 0

	reference_frame = None #'empty/world/map'

	resp_wrench = send_body_wrench('youbot::arm_link_2', reference_frame, 
									None, wrench, start_time, 
									duration )
	resp_wrench = send_body_wrench('wheel_link_bl', reference_frame, 
									None, wrench, start_time, 
									duration )
	resp_wrench = send_body_wrench('wheel_link_br', reference_frame, 
									None, wrench, start_time, 
									duration )
	resp_wrench = send_body_wrench('wheel_link_fl', reference_frame, 
									None, wrench, start_time, 
									duration )
	resp_wrench = send_body_wrench('wheel_link_fr', reference_frame, 
									None, wrench, start_time, 
									duration )

	clear_bl = clear_active_wrenches('wheel_link_bl')
	clear_br = clear_active_wrenches('wheel_link_bl')
	clear_fl = clear_active_wrenches('wheel_link_bl')
	clear_fr = clear_active_wrenches('wheel_link_bl')

	print(resp_wrench)
	
	# send multiple torques to four wheels
	# # right wheel
	# wheel_joint_bl = 'wheel_joint_bl'
	# wheel_joint_br = 'wheel_joint_br'
	# wheel_joint_fl = 'wheel_joint_fl'
	# wheel_joint_fr = 'wheel_joint_fr'

	# # create four different asynchronous threads for each wheel
	# wheel_joint_bl_thread = threading.Thread(group=None, target=send_joint_torques, 
	# 							name='wheel_joint_bl_thread', 
	# 							args=(wheel_joint_bl, effort, start_time, duration))
	# wheel_joint_br_thread = threading.Thread(group=None, target=send_joint_torques, 
	# 							name='wheel_joint_br_thread', 
	# 							args=(wheel_joint_br, effort, start_time, duration))
	# wheel_joint_fl_thread = threading.Thread(group=None, target=send_joint_torques, 
	# 							name='wheel_joint_fl_thread', 
	# 							args=(wheel_joint_fl, effort, start_time, duration))
	# wheel_joint_fr_thread = threading.Thread(group=None, target=send_joint_torques, 
	# 							name='wheel_joint_fr_thread', 
	# 							args=(wheel_joint_fr, effort, start_time, duration))

	# # https://docs.python.org/2/library/threading.html
	# wheel_joint_bl_thread.daemon = True
	# wheel_joint_br_thread.daemon = True
	# wheel_joint_fl_thread.daemon = True
	# wheel_joint_fr_thread.daemon = True

	# wheel_joint_bl_thread.start()
	# wheel_joint_br_thread.start()
	# wheel_joint_fl_thread.start()
	# wheel_joint_fr_thread.start()