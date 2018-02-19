import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Categorical

class Alex2D(nn.Module):
	def __init__(self, beams=360, partition=72):
		r"""
		This implements the DQN algorithm by parameterizing the Q
		table with a 3D convolution of AlexNet architecture
		"""
		super(Alex2D, self).__init__()
		self.num_beam_angles = 5 #beams // partition

		self.geom_features = nn.Sequential(
			nn.Conv2d(5, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)

		self.dim_reducer = nn.Sequential(
			nn.Linear(256 * 1 * 1, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, self.num_beam_angles),
		)
		self.net_name = 'alexnet'

		self.output_num =  self.num_beam_angles
		self.hidden_size = 20
		self.input_size = 2
		self.batch_size = self.num_beam_angles


		self.fc1 = nn.Linear(self.input_size, self.hidden_size)
		self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.fc3 = nn.Linear(self.hidden_size, self.output_num)
		self.linear = nn.Linear(25, self.output_num)

	def predict(self, full_state):
		out = self.fc1(full_state)
		out = self.fc2(out)
		q_values = self.fc3(out)	  #--> will be TotalSlices x batch x beam_angles_num

		if q_values.size(0)>1:
			return q_values
		else:
			q_values =  q_values.view(-1)
			q_values  = self.linear(q_values)
			q_values = q_values.unsqueeze(1)

		return q_values


	def forward(self, *args):
		if len(args) == 2: # assemble state
			geom, Thetas = args[0], args[1]
			geom = self.geom_features(geom)

			batch, feats, filt1, filt2 = geom.size()
			geom = geom.view(batch, feats * filt1 * filt2)
			approx_geom  = self.dim_reducer(geom) # will be 65x5 of features

			# print('sizes: ', Thetas.size(), approx_geom.size())
			Thetas = Thetas.expand_as(approx_geom) # will be N x beam_num x 1
			# expand both vars along 2nd singleton dimension
			approx_geom = approx_geom.unsqueeze(2)
			Thetas      = Thetas.unsqueeze(2)
	        # concat both variables --.N x beam_angles x 2 --> [dim1 = pat_features, dim2 = beam_angles]
			full_state = torch.cat((approx_geom, Thetas), 2)
		else:
			full_state = args[0]
		q_values  = self.predict(full_state) # now 10, 1 -> 5x1

		return q_values, full_state
