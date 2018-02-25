#!/usr/bin/env python

import time
import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image, CameraInfo

class KinectReceiver(object):

	def __init__(self, verbose=False):
		super(KinectReceiver, self).__init__()
		self.verbose = verbose

		image_sub = message_filters.Subscriber('/head_mount_kinect2/rgb/image_raw/compressed', CompressedImage)
		depth_sub = message_filters.Subscriber('/head_mount_kinect2/depth/image_raw', Image)

		self.ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		self.ts.registerCallback(self.callback)

		self.rgb_img = None
		
	def callback(self, image, depth):
		image_arr = np.fromstring(image.data, np.uint8)
		depth_arr = np.fromstring(depth.data, np.uint32)

		self.rgb_img = cv2.imdecode(image_arr, cv2.CV_LOAD_IMAGE_COLOR)
