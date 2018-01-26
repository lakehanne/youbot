import numpy as np
from tf.transformations import euler_from_quaternion


des_quat =  [1.0, 2.78940370874e-11, -4.09248993761e-11, -2.1263232087e-07]
_,_,des_theta = euler_from_quaternion(des_quat, axes='sxyz')

config = dict(

	cost_params = {
		'T': 100,
		'dU': 4,
		'dX': 3+3, # includes velocity terms
		'penalty': 0.0001
	},

	goal_state = np.asarray([[-1.42604078657e-07, 7.64165118802e-08, des_theta, 
		                                    0, 0, 0]]).T,
)

    # position: 
    #   x: -1.42604078657e-07
    #   y: 7.64165118802e-08
    #   z: 0.150000000174
    # orientation: 
    #   x: -4.09248993761e-11
    #   y: 2.78940370874e-11
    #   z: -2.1263232087e-07
    #   w: 1.0
