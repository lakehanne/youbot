#!/usr/bin/env python
from __future__ import print_function
import rospy
import logging
import numpy as np
import std_msgs.msg as std_msgs

import roslib
roslib.load_manifest('youbot_navigation')

# import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

class PointCloudsReceiver(object):
  """docstring for PointCloudsReceiver"""
  def __init__(self, rate):
      # super(PointCloudsReceiver, self).__init__()
      self.rate = rate  # 10Hz
      self.points = []

  def pcl_cb(self, pcl_msg):
      #create pcl2 clouds from points
      header = std_msgs.Header()
      header.stamp = rospy.Time.now()
      header.frame_id = 'cloud_map' 

      pcl_msg = pcl_msg
      for p in pcl2.read_points(pcl_msg, field_names = ("x", "y", "z"), skip_nans=True):
          # print (" x : %f  y: %f  z: %f" %(p[0],p[1],p[2]))
          self.points.append(p)

  def laser_listener(self):
      # rospy.init_node('pointcloudsreceiver')
      rospy.Subscriber("/laser_cloud", PointCloud2, self.pcl_cb)
      # sleeper = rospy.Rate(self.rate)
      # sleeper.sleep()

# if __name__ == '__main__':
#   pcl_rcvr = PointCloudsReceiver(5)

#   try:
#     while not rospy.is_shutdown():
#         pcl_rcvr.laser_listener()
#   except Exception as e:
#       print("could not retrieve any msgs from ros core")

"""
This also works but is alot slower than the C++ code
due to the projection that takes longer time

import rospy
from sensor_msgs.msg import LaserScan
import sensor_msgs.point_cloud2 as pcl2
from laser_geometry import LaserProjection

class Lidar():
    def __init__(self):
        self.laser_projector = LaserProjection()

    def on_scan(self, scan_msg):
        # print (scan_msg)
        rospy.loginfo("Got scan, projecting")
        cloud = self.laser_projector.projectLaser(scan_msg)
        for p in pcl2.read_points(cloud, field_names = ("x", "y", "z"), skip_nans=True):
             print(" x : %f  y: %f  z: %f" %(p[0],p[1],p[2]))
             
    def listen(self):
        rospy.init_node('pointcloudsreceiver')
        rospy.Subscriber('/scan', LaserScan, self.on_scan)
        sleeper = rospy.Rate(10)
        sleeper.sleep()
"""        
# if __name__ == '__main__':
#     ldr = PointCloudsReceiver(20)
#     while not rospy.is_shutdown():
#         ldr.laser_listener()