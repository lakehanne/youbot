#!/usr/bin/env python
from __future__ import print_function
import rospy
import logging
import threading

from rospy.rostime import Duration

from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortResponse,\
						 ApplyBodyWrench, ApplyBodyWrenchResponse

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def send_joint_torques(*msg):	
	rospy.wait_for_service('/gazebo/apply_joint_effort')
	try:
		send_torque = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
		rospy.loginfo('sending: {}N to {}'.format(msg[1], msg[0]))
		resp = send_torque(msg[0], msg[1], msg[2], msg[3])

		return ApplyJointEffortResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		print("Service call failed: %s"%e)

if __name__ == "__main__":

	rospy.init_node('clients')

	joint_name = 'youbot::arm_joint_1'
	effort = 40.0
	start_time = Duration(secs = 0, nsecs = 0) # start asap	
	duration = Duration(secs = 40, nsecs = 0) # apply effort continuously without end duration = -1
	
	# response = send_joint_torques(joint_name, effort, start_time, duration)
	# LOGGER.debug('response: ', response)

	# send multiple torques to four wheels
	# right wheel
	wheel_joint_bl = 'wheel_joint_bl'
	wheel_joint_br = 'wheel_joint_br'
	wheel_joint_fl = 'wheel_joint_fl'
	wheel_joint_fr = 'wheel_joint_fr'

	# create four different asynchronous threads for each wheel
	wheel_joint_bl_thread = threading.Thread(group=None, target=send_joint_torques, 
								name='wheel_joint_bl_thread', 
								args=(wheel_joint_bl, effort, start_time, duration))
	wheel_joint_br_thread = threading.Thread(group=None, target=send_joint_torques, 
								name='wheel_joint_br_thread', 
								args=(wheel_joint_br, effort, start_time, duration))
	wheel_joint_fl_thread = threading.Thread(group=None, target=send_joint_torques, 
								name='wheel_joint_fl_thread', 
								args=(wheel_joint_fl, effort, start_time, duration))
	wheel_joint_fr_thread = threading.Thread(group=None, target=send_joint_torques, 
								name='wheel_joint_fr_thread', 
								args=(wheel_joint_fr, effort, start_time, duration))

	# https://docs.python.org/2/library/threading.html
	wheel_joint_bl_thread.daemon = True
	wheel_joint_br_thread.daemon = True
	wheel_joint_fl_thread.daemon = True
	wheel_joint_fr_thread.daemon = True

	wheel_joint_bl_thread.start()
	wheel_joint_br_thread.start()
	wheel_joint_fl_thread.start()
	wheel_joint_fr_thread.start()