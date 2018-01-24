#! /usr/bin/env python
"""
This script calculates the torque of the robot as retrieved from gazebo

# each link's coriolis mass can be accessed as a key from the class' dict object

"""
from __future__ import print_function

import time
import numpy as np

# base links
base_links = ["base_mass", "caster_mass", "wheel_mass"]

class MassMaker(object):
	"""docstring for MassMaker"""
	def __init__(self):
		super(MassMaker, self).__init__()
		
	def make_link_matrix(self, link_dict):
		temp =  np.zeros([6,6])
		temp[0,0] = link_dict['mass']
		temp[1,1] = link_dict['mass']
		temp[2,2] = link_dict['mass']

		temp[3,3] = link_dict['ixx']
		temp[3,4] = link_dict['ixy']
		temp[3,5] = link_dict['ixz']

		temp[4,3] = temp[3,4]  # symmetric
		temp[4,4] = link_dict['iyy']
		temp[4,5] = link_dict['iyz']

		temp[5,3] = temp[3,5]	# symmetric
		temp[5,4] = temp[4,5]	# symmetric
		temp[5,5] = link_dict['izz']

		return temp

	def get_constants(self):

		various_links = dict(
		base_footprint = dict(
			mass=0.1,
			ixx=1.0, ixy=0.0,
			ixz=0.0, iyy=1.0,
			iyz=0.0, izz=1.0),

		base = dict(
			mass=22,
			ixx=5.652232699207, ixy=-0.009719934438,
			ixz=1.293988226423 ,iyy=5.669473158652,
			iyz=-0.007379583694 ,izz=3.683196351726),

		caster = dict(
			mass=0.1,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=0.000004273467, izz=0.011763977943,
			friction=1.0,
			),

		caster_link_fr = dict(
			mass=0.1,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=0.000004273467, izz=0.011763977943,
			friction=1.0,
			)	,

		caster_link_br = dict(
			mass=0.1,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=0.000004273467, izz=0.011763977943,
			friction=1.0,
			)	,		

		caster_link_bl = dict(
			mass=0.1,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=0.000004273467, izz=0.011763977943,
			friction=1.0,
			)	,

		wheel = dict(
			mass=0.4,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=-0.000004273467, izz=0.011763977943,
			friction=1.0,
            radius = 0.05,
			),

		wheel_link_fl = dict(
			mass=0.4,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=-0.000004273467, izz=0.011763977943,
			friction=1.0,
			),

		wheel_link_fr = dict(
			mass=0.4,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=-0.000004273467, izz=0.011763977943,
			friction=1.0,
			)	,	

		wheel_link_bl = dict(
			mass=0.4,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=-0.000004273467, izz=0.011763977943,
			friction=1.0,
			),


		wheel_link_br = dict(
			mass=0.4,
			ixx=0.012411765597, ixy=-0.000711733678, 
			ixz=0.00050272983, iyy=0.015218160428, 
			iyz=-0.000004273467, izz=0.011763977943,
			friction=1.0,
			)	,	


		plate = dict(
			mass=1,
			ixx=0.01, ixy=0, ixz=0,
			iyy=0.01, iyz=0, izz=0.01,
			),

		link_0 = dict(
			mass=0.845,
			ixx=0.01, ixy=0, ixz=0,
			iyy=0.01, iyz=0, izz=0.01,
			),

		link_1 = dict(
			mass=2.412,
			ixx=0.003863, ixy=-0.000979, ixz=0.000000, 
			iyy=0.006196, iyz=0.000000, izz=0.006369,
			),

		link_2 = dict(
			mass=1.155,
			ixx=0.000823, ixy=0.000000, ixz=-0.000000,
			iyy=0.004447, iyz =0.000000, izz=0.004439,
			),

		link_3 = dict(
			mass=0.934,
			ixx=0.002459,  ixy=0.000000 , ixz=0.000000 , 
			iyy=0.002571,  iyz=-0.000000,  izz=0.000535, 
			),

		link_4 = dict(
			mass=0.877,
			ixx=0.000869, ixy=0.000000, ixz=-0.000000 ,
			iyy=0.001173, iyz=-0.000231, izz=0.001091,
			),

		link_5 = dict(
			mass=0.251,
			ixx=0.000280, ixy=0.000000, ixz=0.000000 ,
			iyy=0.000339, iyz=0.000000, izz=0.000119,
			),

		gripper = dict(
			mass=0.1,
			ixx=0.01, ixy=0, ixz=0,
			iyy=0.01, iyz=0, izz=0.01,
			)	,	

		gripper_finger_link_l = dict(
			mass=0.01,
			ixx=0.01, ixy=0, ixz=0,
			iyy=0.01, iyz=0, izz=0.01,
			)	,


		gripper_finger_link_r = dict(
			mass=0.01,
			ixx=0.01, ixy=0, ixz=0,
			iyy=0.01, iyz=0, izz=0.01,
			),

		hokuyo = dict(
			mass=0.16,
			ixx=0.1, ixy=0, ixz=0,
			iyy=0.1, iyz=0, izz=0.1,
			),

		kinect = dict(
			mass=0.01,
			ixx=0.001, ixy=0, ixz=0,
			iyy=0.001, iyz=0, izz=0.001,
			),

		kinect_depth = dict(
			mass=0.01,
			ixx=0.001, ixy=0, ixz=0,
			iyy=0.001, iyz=0, izz=0.001,
			)	,	

		kinect_depth_optical = dict(
			mass=0.001,
			ixx=0.0001, ixy=0, ixz=0,
			iyy=0.0001, iyz=0, izz=0.0001,
			)	,

		kinect_rgb_frame = dict(
			mass=0.001,
			ixx=0.0001, ixy=0, ixz=0,
			iyy=0.0001, iyz=0, izz=0.0001,
			)	,

		kinect_rgb_optical = dict(
			mass=0.001,
			ixx=0.0001, ixy=0, ixz=0,
			iyy=0.0001, iyz=0, izz=0.0001,
			)	
		)

		return various_links

	def get_mass_matrices(self):
		self.various_links = self.get_constants()

		for key in self.various_links.keys():
			self.various_links[key]['mass_inertia'] = self.make_link_matrix(self.various_links[key])
# 			self.various_links[key]['friction'] = self.various_links[key]['friction'] \
#                     if 'friction' in self.various_links.keys() else None
		self.__dict__.update(self.various_links)
		del self.various_links

	def get_body_jacobian_matrices(self):
		"""
		For each link with matrix m_i, we find the 
		body jacobian matrix j_i
		"""

		

