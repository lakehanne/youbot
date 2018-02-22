import os
from os.path import join, expanduser
import numpy as np
from tf.transformations import euler_from_quaternion

from scripts.costs import CostAction, CostState, CostSum

# retrieved this from /gazebo/link_states
des_quat =  [1.0,-4.09248993761e-11, 2.78940370874e-11, -2.1263232087e-07]
_,_,des_theta = euler_from_quaternion(des_quat, axes='sxyz')



config = {
    'num_samples':	5,
	'conditions':	4,
	'cost_params':	{
		'action_penalty': np.array([0.0001, 0.0001, 0.0001, 0.0001]),
		'state_penalty': np.array([1, 1, 1]),
		'final_cost_weight': 1e-2,
		'stage_cost_weight': 1e-2,
	},

	'agent': {
		'goal_state': np.asarray([1.3, 0.8, -2.752012201858897e-19]).T,
		# 'goal_state': np.asarray([0.82452343,  0.59753333,  0.07282408]).T,
		'T':  40,
		'dU': 4,
		'dV': 4,
		'dX': 3, # includes velocity terms
		'dO': 3,
		'TOL': 1e-2, # tolerance for stopping DP algorithm
		'alpha': 1.0,
		'conditions': 4,
		'sample_length': 30,  # number of samples used in cost-to-go function
		'delta': 1e-4, # initial value of step size used in adjusting delta
        'mu': 1e-6, # init regularizer
        'mu_min': 1e-6, # regularization min
        'delta_nut': 2,  # for regularization schedule
		'euler_step': 1e-2,  # step for the euler integrator
		'euler_iter': 50,  # step for the euler integrator
		'smooth_random_walk': True,
	    'smooth_noise': True,
	    'smooth_noise_var': 2.0,
	    'smooth_noise_renormalize': True,
	    'save_dir': join(expanduser('~'), 'catkin_ws', 'src', 'youbot', 
	    				'youbot_navigation', 'data'),
	},

	'trajectory': {
		'init_action': np.array([9.891485822039117, 
								10.830327455656795, 
								8.995303183020859, 
								7.821391440870244]),
		'stopping_condition': 0.01, #18.0348559119, # this is prob. dependent
		'stopping_eta': 1e3, # initial eta
		'c_zero': 1e-6, # zero term for c DP restart procedure
		'duration_length': 0.3,  # amount of time to apply control law
	},

	'all_costs': {
	    'type': CostSum,
	    'costs': [{'type': CostAction, 'wu': np.array([1, 1])}, {'type': CostState}],
	    'weights': [1e-5, 1.0],
	    'mode': 'antagonist',
	    'gamma': 1e0,
	},

	'linearized_params': {
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
}

