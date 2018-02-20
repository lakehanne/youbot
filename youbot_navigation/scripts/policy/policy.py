#!/usr/bin/env python

import PIL
import cv2
import time
import math
import rospy
import threading
import numpy as np
from PIL import Image
from scripts.algorithm_utils import Alex2D
from scripts.subscribers import KinectReceiver, \
								ModelStatesReceiver

import sys
import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

use_cuda 	 = torch.cuda.is_available()
FloatTensor  = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
LongTensor   = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor   = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor       = FloatTensor

if use_cuda:
	LOGGER.info("Running on GPU: {}".format(torch.cuda.current_device()))

class ProcessObservations(object):
	def __init__(self, rate=30, verbose=False, test=False):
		super(ProcessObservations, self).__init__()
		self.verbose = verbose
		self.rate = rate

		self.kinect_rcvr = KinectReceiver(verbose)
		self.model_rcvr = ModelStatesReceiver()


		self.net = Alex2D()

		if not test:
			self.net.apply(self.weights_init)

		if use_cuda:
			with torch.cuda.device(0):
				self.net = self.net.cuda()

	def weights_init(self, m):
		for m in m.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				size = m.weight.size()
				fan_out = size[0] # number of rows
				fan_in = size[1] # number of columns
				variance = np.sqrt(2.0/(fan_in + fan_out))
				m.weight.data.normal_(0.0, variance)
			elif isinstance(m, nn.LSTM):
				for key in m.state_dict().keys():
					map(lambda x: init.uniform(x, 0, 1), m.state_dict().values())
			elif isinstance(m, nn.LSTMCell):
				for key in m.state_dict().keys():
					map(lambda x: init.uniform(x, 0, 1), m.state_dict().values())

	def resize_ndarray(self, array, resize_shape=(240, 240) ):
		assert array.ndim == 4 or array.ndim == 3, 'dims of array to be resized not understood'
		'array dimensions should be 3D or 4D. Got {} dims'.format(array.ndim)

		if array.ndim == 4:
			square_image = np.zeros((array.shape[0],array.shape[1])+resize_shape)
			for channel in range(array.shape[0]):
				for structs in range(array.shape[1]):
					pil_image =  PIL.Image.fromarray(array[channel, structs,:,:])
					square_image[channel, structs,:,:] = np.asarray(pil_image.resize(resize_shape, Image.NEAREST)).astype(np.float64)
		elif array.ndim == 3:
			square_image = np.zeros((array.shape[0],)+resize_shape)
			for channel in range(array.shape[0]):
				pil_image =  PIL.Image.fromarray(array[channel,:,:])
				square_image[channel, :,:] = np.asarray(pil_image.resize(resize_shape, Image.NEAREST)).astype(np.float64)
		else:
			LOGGER.critical('type of array not understood')
		return square_image

	def neatify_model_states(self, model_states):

		pose = model_states.pose
		twist = model_states.twist

		# world_pose = [pose[1].position.x, pose[1].position.y, pose[1].position.z,
		# 				pose[1].orientation.x, pose[1].orientation.y, pose[1].orientation.z,
		# 				pose[1].orientation.w]
		# world_twist = [twist[1].linear.x, twist[1].linear.y, twist[1].linear.z,
		# 				twist[1].angular.x, twist[1].angular.y, twist[1].angular.z]

		youbot_pose = [pose[-2].position.x, pose[-2].position.y, pose[-2].position.z,
						pose[-2].orientation.x, pose[-2].orientation.y, pose[-2].orientation.z,
						pose[-2].orientation.w]
		youbot_twist = [twist[-2].linear.x, twist[-2].linear.y, twist[-2].linear.z,
						twist[-2].angular.x, twist[-2].angular.y, twist[-2].angular.z]

		boxtacle_pose = [pose[-1].position.x, pose[-1].position.y, pose[-1].position.z,
						 pose[-1].orientation.x, pose[-1].orientation.y, pose[-1].orientation.z,
						 pose[-1].orientation.w]
		boxtacle_twist = [twist[-1].linear.x, twist[-1].linear.y, twist[-1].linear.z,
						twist[-1].angular.x, twist[-1].angular.y, twist[-1].angular.z]					

		# world = world_pose + world_twist
		youbot = youbot_pose + youbot_twist
		boxtacle = boxtacle_pose + boxtacle_twist

		# print(world)
		print(youbot)
		print(boxtacle)


	def convolve_rgb(self):
		# first gather robot model states
		image = self.kinect_rcvr.rgb_img
		model_states = self.model_rcvr.model_states

		model_states_list = self.neatify_model_states(model_states)

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

			# cv2.imshow('rgb image', image)
			
			square_image = self.resize_ndarray(image.transpose(), resize_shape=(240, 240))
			square_image_t = Variable(torch.from_numpy(square_image))
			print(square_image_t.size())

			output = self.net(square_image_t, None)

			print(output.shape)


# try:
rospy.init_node('image_subscriber')

rate = 30

po = ProcessObservations(rate=rate, verbose=True)

sleeper = rospy.Rate(rate)

while not rospy.is_shutdown():
	po.convolve_rgb()	
	sleeper.sleep()

# except Exception as e:
	# raise e	