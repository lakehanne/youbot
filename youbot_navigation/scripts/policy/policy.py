#!/usr/bin/env python

import cv2
import time
import rospy
import threading
from scripts.algorithm_utils import Alex2D
from scripts.subscribers import KinectReceiver

class ProcessObservations(object):
	def __init__(self, rate=30, verbose=False):
		super(ProcessObservations, self).__init__()
		self.kr = KinectReceiver()
		self.verbose = verbose
		self.rate = rate
		self.net = Alex2D()

	def convolve_rgb(self):
		image = self.kr.rgb_img

		#### Feature detectors using CV2 #### 
		# "","Grid","Pyramid" + 
		# "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
		method = "GridFAST"
		feat_det = cv2.FeatureDetector_create(method)

		if image is not None:

			tic = time.time()
			# convert np image to grayscale
			featPoints = feat_det.detect(
							cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
			toc = time.time()

			if self.verbose :
				print '%s detector found: %s points in: %s sec.'%(method,
				    len(featPoints),toc-tic)

			for featpoint in featPoints:
				x,y = featpoint.pt
				cv2.circle(image,(int(x),int(y)), 3, (0,0,255), -1)

			cv2.imshow('rgb image', image)

			imageT = image.transpose()
			
			print(imageT.shape)

try:
	rospy.init_node('image_subscriber')

	rate = 30

	po = ProcessObservations(rate=rate, verbose=False)

	sleeper = rospy.Rate(rate)

	while not rospy.is_shutdown():
		po.convolve_rgb()	
		sleeper.sleep()

except Exception as e:
	raise e	
