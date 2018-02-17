#!/usr/bin/env python
# from __future__ import print_function
# import rospy
# import logging

# import roslib
# roslib.load_manifest('youbot_navigation')

# # import pcl
# from sensor_msgs.msg import PointCloud2
# import sensor_msgs.point_cloud2 as pcl2

# class PointCloudsReceiver(object):
#   """docstring for PointCloudsReceiver"""
#   def __init__(self, rate):
#       super(PointCloudsReceiver, self).__init__()
#       self.rate = rate  # 10Hz
#       self.laser_pcl_topic = "/laser_cloud"

#   def pcl_cb(self, pcl_msg):

#       print(pcl_msg.data)

#       #create pcl2 clouds from points
#       header = std_msgs.msg.Header()
#       header.stamp = rospy.Time.now()
#       header.frame_id = 'cloud_map'
#       scaled_pcl = pcl2.create_cloud_xyz32(header, pcl_msg.data)  

#       print(scaled_pcl)


#   def laser_listener():
#       print('in cb listener')
#       rospy.Subscriber("/laser_cloud", PointCloud2, self.pcl_cb)
#       sleeper = rospy.Rate(self.rate)
#       sleeper.sleep()

# if __name__ == '__main__':
#   rospy.init_node("pointcloudsreceiver")
#   pcl_rcvr = PointCloudsReceiver(5)
#   try:
#       pcl_rcvr.laser_listener()
#   except Exception as e:
#       print("could not retrieve any msgs from ros core")


import rospy, math, random
import numpy as np
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection

class Lidar():
    def __init__(self, scan_topic="/scan"):
        # self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.on_scan)
        self.laser_projector = LaserProjection()

    def on_scan(self, scan):
        # print scan
        rospy.loginfo("Got scan, projecting")
        cloud = self.laser_projector.projectLaser(scan)
        print cloud
        rospy.loginfo("Printed cloud")
        for p in pc2.read_points(cloud, field_names = ("x", "y", "z"), skip_nans=True):
             print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
             
    def listen(self):
        # rospy.Subscriber("/scan", PointCloud2, self.on_scan)
        rospy.Subscriber('/scan', LaserScan, self.on_scan)

if __name__ == '__main__':
    ldr = Lidar()
    ldr.listen()