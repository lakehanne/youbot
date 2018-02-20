import rospy
from gazebo_msgs.msg import ModelStates

class ModelStatesReceiver(object):
	"""docstring for ModelStatesReceiver"""
	def __init__(self):
		super(ModelStatesReceiver, self).__init__()

		self.subscriber = rospy.Subscriber("/gazebo/model_states",
					ModelStates, self.callback, queue_size = 1)

	def callback(self, model_states):
		self.model_states = model_states
		