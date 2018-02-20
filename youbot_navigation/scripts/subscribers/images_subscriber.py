#!/usr/bin/env python

import time
import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image, CameraInfo

class KinectReceiver(object):

	def __init__(self):
		super(KinectReceiver, self).__init__()
		self.verbose = False

		image_sub = message_filters.Subscriber('/head_mount_kinect2/rgb/image_raw/compressed', CompressedImage)
		depth_sub = message_filters.Subscriber('/head_mount_kinect2/depth/image_raw', Image)

		self.ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		self.ts.registerCallback(self.callback)

		self.rgb_img = None
		
	def callback(self, image, depth):
		image_arr = np.fromstring(image.data, np.uint8)
		depth_arr = np.fromstring(depth.data, np.uint32)

		self.rgb_img = cv2.imdecode(image_arr, cv2.CV_LOAD_IMAGE_COLOR)

	# def show(self):
	# 	while not rospy.is_shutdown():
	# 		image_sub = message_filters.Subscriber('/head_mount_kinect2/rgb/image_raw/compressed', CompressedImage)
	# 		depth_sub = message_filters.Subscriber('/head_mount_kinect2/depth/image_raw', Image)

	# 		self.ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
	# 		self.ts.registerCallback(self.callback)
	# 		cv2.imshow(self.rgb_img)
# 	def listen(self, rate):
# 		self.ts.registerCallback(self.callback)

# 		image_arr = np.fromstring(self.image.data, np.uint8)

# 		self.rgb_img = cv2.imdecode(image_arr, cv2.CV_LOAD_IMAGE_COLOR)
# 		#### Feature detectors using CV2 #### 
# 		# "","Grid","Pyramid" + 
# 		# "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
# 		method = "GridFAST"
# 		feat_det = cv2.FeatureDetector_create(method)
# 		tic = time.time()

# 		# convert np image to grayscale
# 		featPoints = feat_det.detect(
# 		cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY))
# 		toc = time.time()
# 		if self.verbose :
# 			print '%s detector found: %s points in: %s sec.'%(method,
# 			    len(featPoints),toc-tic)

# 		for featpoint in featPoints:
# 			x,y = featpoint.pt
# 			cv2.circle(self.rgb_img,(int(x),int(y)), 3, (0,0,255), -1)
# 		sleeper = rospy.Rate(rate)
# 		sleeper.sleep()

# rospy.init_node('image_subscriber')
# kr = KinectReceiver(False)
# kr.show()
# # kr.listen(30) if not rospy.is_shutdown() else None
# rospy.spin()