from scripts.algorithm_utils import Alex2D
from scripts.subscribers import KinectReceiver

class ProcessObservations(object):
	def __init__(self):
		super(ProcessObservations, self).__init__(KinectReceiver)

	def convolve_rgb(self):
		image = self.rgb_img

	def show(self):
		cv2.imshow('rgb image', self.rgb_img)
		cv2.waitKey(2)


rospy.init_node('image_subscriber')
po = ProcessObservations()
po.show()		
rospy.spin()