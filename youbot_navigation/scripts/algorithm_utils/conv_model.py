import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Categorical

class Alex2D(nn.Module):
	def __init__(self, channels=3):
		r"""
		This implements the DQN algorithm by parameterizing the Q
		table with a 3D convolution of AlexNet architecture
		"""
		super(Alex2D, self).__init__()
		self.num_channels = channels

		self.conv_features = nn.Sequential(
			nn.Conv2d(self.num_channels, 240, kernel_size=7, stride=2, padding=0),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 117, kernel_size=5, stride=2, padding=0),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(117, 113, kernel_size=5, stride=2, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(113, 109, kernel_size=5, stride=2, padding=0),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(256, 256, kernel_size=3, padding=1),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2),
			# nn.Linear(109, 64),
		)

		self.conv1 = nn.Conv2d(self.num_channels, 240, kernel_size=7, stride=2, padding=0)
		self.conv2 = nn.Conv2d(64, 117, kernel_size=5, stride=2, padding=0)
		self.conv3 = nn.Conv2d(117, 113, kernel_size=5, stride=2, padding=0)
		self.conv4 = nn.Conv2d(113, 109, kernel_size=5, stride=2, padding=0)

		self.dim_reducer = nn.Sequential(
			nn.Linear(256 * 1 * 1, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, self.num_channels),
		)
		self.net_name = 'visuopolicy'

		self.output_num =  self.num_channels
		self.hidden_size = 20
		self.input_size = 2
		self.batch_size = self.num_channels

		self.fc1 = nn.Linear(self.input_size, 64)
		self.fc2 = nn.Linear(64, 40)
		self.fc3 = nn.Linear(40, 3)

	# def predict(self, full_state):
	# 	out = self.fc1(full_state)
	# 	out = self.fc2(out)
	# 	q_values = self.fc3(out)	  #--> will be TotalSlices x batch x beam_angles_num

	# 	if q_values.size(0)>1:
	# 		return q_values
	# 	else:
	# 		q_values =  q_values.view(-1)
	# 		q_values  = self.linear(q_values)
	# 		q_values = q_values.unsqueeze(1)

	# 	return q_values


	def forward(self, *args):
		# if len(args) == 2: # assemble state
		input_image, robot_config = args[0], args[1]
		out  = self.conv1(input_image)
		print('out: ', out)
		input_image = F.softmax(self.conv_features(input_image), dim=None)

		print('convolved image size: {}'.format(input_image.shape))

		# batch, feats, filt1, filt2 = input_image.size()
		# input_image = input_image.view(batch, feats * filt1 * filt2)
		# approx_geom  = self.dim_reducer(input_image) # will be 65x5 of features

		# print('sizes: ', robot_config.size(), approx_geom.size())
		# robot_config = robot_config.expand_as(approx_geom) # will be N x beam_num x 1
		# expand both vars along 2nd singleton dimension
		# approx_geom = approx_geom.unsqueeze(2)
		# robot_config      = robot_config.unsqueeze(2)
        # concat both variables --.N x beam_angles x 2 --> [dim1 = pat_features, dim2 = beam_angles]
		full_state = torch.cat((input_image, robot_config), 1)
		# else:
		# 	full_state = args[0]
		# q_values  = self.predict(full_state) # now 10, 1 -> 5x1

		return full_state
