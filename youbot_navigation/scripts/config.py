import numpy as np
from tf.transformations import euler_from_quaternion


des_quat =  [1.0, 2.78940370874e-11, -4.09248993761e-11, -2.1263232087e-07]
_,_,des_theta = euler_from_quaternion(des_quat, axes='sxyz')

config = dict(

	cost_params = {
		'action_penalty': np.array([0.0001, 0.0001, 0.0001]),
		'state_penalty': np.array([1, 1, 1]),
		'final_cost_weight': 1e-2,
		'stage_cost_weight': 1e-2,
	},

	agent = {
		# 'goal_state': np.asarray([[-1.42604078657e-07, 7.64165118802e-08, des_theta,
		# 	                                    0, 0, 0]]).T,
		'goal_state': np.asarray([-1.42604078657e-07, 7.64165118802e-08, des_theta]).T,
		'T': 100,
		'dU': 4,
		'dV': 4,
		'dX': 3, # includes velocity terms
		'dO': 3,
		'alpha': 1.0,
		'euler_step': 1e-2,  # step for the euler integrator
		'euler_iter': 50,  # step for the euler integrator
	    'smooth_noise': True,
	    'smooth_noise_var': 2.0,
	    'smooth_noise_renormalize': True,
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
