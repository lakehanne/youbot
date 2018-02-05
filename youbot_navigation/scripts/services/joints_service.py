#!/usr/bin/env python

from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortResponse,\
						 ApplyBodyWrench, ApplyBodyWrenchResponse
import rospy

def handle_torque_control(req):
    # print "Returning [%s + %s = %s]"%(req.effort, req.b, (req.effort + req.b))
    return ApplyJointEffortResponse(req.joint, req.effort, req.time, req.duration)

def send_torque_server():
    rospy.init_node('send_torque_server')
    s = rospy.Service('/gazebo/apply_joint_effort', ApplyJointEffort, handle_torque_control)
    print "sending joint effort."
    rospy.spin()

if __name__ == "__main__":
    send_torque_server()