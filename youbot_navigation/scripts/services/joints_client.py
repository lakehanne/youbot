#!/usr/bin/env python

import rospy

from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortResponse,\
						 ApplyBodyWrench, ApplyBodyWrenchResponse


def send_joint_torques(msg):	
	# rospy.wait_for_service('/gazebo/apply_joint_effort ')

	try:
		send_torque = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
		rospy.loginfo('sending torque: {} to  {}'.format(msg['effort'], msg['joint_name']))
		resp = send_torque(msg)

		return ApplyJointEffortResponse(resp.success, resp.status_message)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

if __name__ == "__main__":

	rospy.init_node('clients')
	msg = dict(
	joint_name = 'youbot::arm_joint_1',
	effort = 20.0,
	start_time = dict(
		secs = 0,
		nsecs = 0),
	duration = dict(secs=20,
		nsecs = 0),
	)

	response = send_joint_torques(msg)

	print('{}, \n {}'.format(response[0], response[1]))