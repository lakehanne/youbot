import rospy
from nav_msgs.msg import Odometry

def callback(data):
	rospy.loginfo(data.twist.twist.linear)

def listener():
	rospy.init_node('test')
	rate = rospy.Rate(5)
	rate.sleep()
   	rospy.Subscriber("/odom", Odometry, callback)
	rospy.spin()

listener()