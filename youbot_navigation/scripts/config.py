import numpy as np
from tf.transformations import euler_from_quaternion


des_quat =  [1.0, 2.78940370874e-11, -4.09248993761e-11, -2.1263232087e-07]
_,_,des_theta = euler_from_quaternion(des_quat, axes='sxyz')

config = dict(
    num_samples =  5,
	conditions = 4,
	cost_params = {
		'action_penalty': np.array([0.0001, 0.0001, 0.0001, 0.0001]),
		'state_penalty': np.array([1, 1, 1]),
		'final_cost_weight': 1e-2,
		'stage_cost_weight': 1e-2,
	},

	agent = {
		# 'goal_state': np.asarray([[-1.42604078657e-07, 7.64165118802e-08, des_theta,
		# 	                                    0, 0, 0]]).T,
		'goal_state': np.asarray([1.25000, 0.65000, 0]).T,
		'T':  100,
		'dU': 4,
		'dV': 4,
		'dX': 3, # includes velocity terms
		'dO': 3,
		'alpha': 1.0,
		'conditions': 4,
		'delta': 1e-4, # initial value of step size used in adjusting delta
        'mu': 1e-6, # init regularizer
        'mu_min': 1e-6, # regularization min
        'delta_nut': 2,  # for regularization schedule
		'euler_step': 1e-2,  # step for the euler integrator
		'euler_iter': 50,  # step for the euler integrator
	    'smooth_noise': True,
	    'smooth_noise_var': 2.0,
	    'smooth_noise_renormalize': True,
	},

	trajectory = {
		# retrieved from gazebo [-9.891485822039117, -10.830327455656795, -8.995303183020859, -7.821391440870244]
		# 'init_action': 10,  # initial nominal toque for all four wheels of mobile base
		'init_action': np.array([9.891485822039117, 10.830327455656795, 8.995303183020859, 7.821391440870244])
		# 'init_action': np.array([-9.891485822039117, -10.830327455656795, -8.995303183020859, -7.821391440870244])
	},

	linearized_params = {
		    'x0': [np.array([0.5*np.pi, 0, 0, 0, 0, 0]),
	           np.array([0.75*np.pi, 0.5*np.pi, 0, 0, 0, 0]),
	           np.array([np.pi, -0.5*np.pi, 0, 0, 0, 0]),
	           np.array([1.25*np.pi, 0, 0, 0, 0, 0]),
	          ],

	        'u0': [
	        	   np.array([0, 0, 0, 0, 0, 0]),
		           np.array([0, 0, 0, 0, 0, 0]),
		           np.array([0, 0, 0, 0, 0, 0]),
		           np.array([0, 0, 0, 0, 0, 0]),
	          ],
		}
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
