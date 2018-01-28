#!/usr/bin/env python

import os
import imp
import time
import copy
import rospy
import roslib
import logging
import argparse
import threading
import numpy as np
import scipy as sp
from nav_msgs.msg import Odometry
from scipy.integrate import odeint, RK45
from scripts.dynamics import Dynamics

import rospkg

import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt


roslib.load_manifest('youbot_navigation')

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
                        help='max num iterations' )
parser.add_argument('--plot_state', '-ps', action='store_true', default='True',
                        help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_true', default='True',
                        help='save plotted figures' )
parser.add_argument('--silent', '-si', action='store_true', default='False',
                        help='max num iterations' )
args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

if args.silent:
    LOGGER.setLevel(logging.INFO)
else:
    LOGGER.setLevel(logging.DEBUG)


print(args)

class TrajectoryOptimization(Dynamics):
    """docstring for TrajectoryOptimization
    rate = rate at which to call the ros engine for msgs
    """
    def __init__(self, arg, rate, hyperparams):
        super(TrajectoryOptimization, self).__init__(Dynamics(Odometry, rate=rate))

        self.args            = arg

        config              = hyperparams.config
        # self.hyperparams  = hyperparams
        self.T              = config['agent']['T']
        self.dU             = config['agent']['dU']
        self.dX             = config['agent']['dX']
        self.goal_state     = config['agent']['goal_state']
        self.cost_penalty   = config['cost_params']['penalty']
        self.wheel_rad      = self.wheel['radius']

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation') 

        plt.ioff()

    def get_nominal_state(self, res, interval, mass_matrix, coriolis_matrix, \
                             B_matrix, S_matrix, friction_vector, nom_control):   
        """
            Function that calculates the nominal trajectory of the robot
        """         
        q_init, x_init = res

        # Note that the B matrix contains q_pos
        mass_inv = -np.linalg.inv(mass_matrix)
        state_ddot = mass_inv.dot(coriolis_matrix).dot(x_init) - \
                    mass_inv.dot(B_matrix).dot(S_matrix).dot(friction_vector) + \
                    (mass_inv.dot(B_matrix).dot(nom_control))/self.wheel_rad
        diff_eq = [x_init, state_ddot]

        return diff_eq

    def get_cost_jacs(state_delta, action_delta, state_bar,
                                    action_bar):
        # assemble the cost's first order moments ell
        pass


    def do_traj_opt(self):
        T = self.T
        dU = self.dU
        dX = self.dX


        # assemble Jacobian of transfer matrix and et cetera
        action     = np.zeros((T, dU, 1))
        action_bar = np.zeros((T, dU, 1))
        action_delta    = np.zeros((T, dU, 1))

        state      = np.zeros((T, dX, 1))
        state_bar  = np.zeros((T, dX, 1)) 
        state_delta= np.zeros((T, dX, 1))  
        state_goal = np.zeros((T, dX, 1)) # desired state

        fx         = np.zeros((T, dX, dX))
        fu         = np.zeros((T, dX, dU))

        # assemble stage_costs   

        # Allocate.
        Vxx = np.zeros((T, dX, dX))
        Vx = np.zeros((T, dX))
        Qtt = np.zeros((T, dX+dU, dX+dU))
        Qt = np.zeros((T, dX+dU))

        action_zero, state_zero = np.zeros_like(action[0,:,:]), np.zeros_like(state[0,:,:])

        for t in range (T-1, -1, -1):
            # get body dynamics. Note that some of these are time varying parameters
            body_dynamics = self.assemble_dynamics() 

            # time varying inverse dynamics parameters
            mass_matrix     = body_dynamics.M       
            coriolis_matrix = body_dynamics.C       
            B_matrix        = body_dynamics.B       
            S_matrix        = body_dynamics.S       
            qaccel          = body_dynamics.qaccel  
            qvel            = body_dynamics.qvel    
            q               = body_dynamics.q      

            # this should be time-varying but is constant for now
            friction_vector = body_dynamics.f 

            # calculate inverse dynamics equation
            BBT             = B_matrix.dot(B_matrix.T)
            Inv_BBT         = np.linalg.inv(BBT)
            multiplier      = Inv_BBT.dot(self.wheel_rad * B_matrix)
            inv_dyn_eq      = mass_matrix.dot(qaccel) + coriolis_matrix.dot(qvel) + \
                                    B_matrix.T.dot(S_matrix).dot(friction_vector)

            # set up costs at time T
            action[t,:,:]    = multiplier.dot(inv_dyn_eq)
            state[t,:,:]     = q #np.r_[q, qvel]
            state_goal[t,:,:]  = self.goal_state
            fx[t,:,:]          = -np.linalg.inv(mass_matrix).dot(coriolis_matrix)
            fu[t,:,:]          = -(1/self.wheel_rad) * np.linalg.inv(mass_matrix).dot(B_matrix.T) 

            # euler integration  parameters
            init_cond = [ action_bar[t,:,:], state_bar[t,:,:] ]
            tt = np.linspace(0, 10, 101)
            mass_inv = -np.linalg.inv(mass_matrix)
            print(mass_inv.dot(coriolis_matrix).shape, state_bar[t,:,:].shape)
            state_ddot = mass_inv.dot(coriolis_matrix).dot(state_bar[t,:,:]) - \
                         mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector) + \
                        (mass_inv.dot(B_matrix.T).dot(action_bar[t,:,:]))/self.wheel_rad
            # diff_eq = [x_init, state_ddot]
            # print('state_ddot: ', state_ddot, friction_vector)

            # state_bar = odeint(self.get_nominal_state, init_cond, tt, \
            #             args=(mass_matrix, coriolis_matrix, B_matrix, \
            #                 S_matrix, friction_vector, action_bar[t,:,:]))

            if self.args.plot_state:
                plt.plt(tt, state_bar[:,0], 'b', label='qvel', fontweight='bold')
                plt.plt(tt, state_bar[:,1], 'g', label='qpos', fontweight='bold')
                plt.legend(loc='best')
                plt.xlabel('t', fontweight='bold')
                plt.ylabel('final q after integration', fontweight='bold')
                plt.grid()
                plt.gcf().set_size_inches(10,4)
                plt.cla()

                if self.args.save_figs:
                    figs_dir = os.path.join(self.path, 'figures')
                    os.mkdir(figs_dir) if not os.path.exists(figs_dir) else None                    
                    plt.savefig(figs_dir + '/state_' + repr(t), 
                            bbox_inches='tight',facecolor='None')

            # state_bar = state_bar[0][::-1] # reverse the order of states to --> [q, qdot]
            qvel_bar, qpos_bar = sum(state_bar[:,0]), sum(state_bar[:,1])
            state_bar = [qpos_bar, qvel_bar]

            # calculate new system trajectories
            state_delta[t,:,:]  = state[t,:,:] - state_bar[t,:,:]
            action_delta[t,:,:] = action[t,:,:] - action_bar[t,:,:]
            
            get_cost_jacs = self.get_cost_jacs(state_delta, action_delta, state_bar,
                                    action_bar)
            # retrieve this from the inverse dynamics equation (20) in Todorov ILQG paper
            # stage_costs[t, :] = self.penalty * action.T.dot(action)
        
    def run_backward_pass(self):
        raise NotImplementedError("Must be implemented in subclass.")
        # self.final_cost = 0.5 * (diff).T.dot(diff)


if __name__ == '__main__':

    from scripts import __file__ as scripts_filepath
    scripts_filepath = os.path.abspath(scripts_filepath)
    scripts_dir = '/'.join(str.split(scripts_filepath, '/')[:-1]) + '/'
    hyperparams_file = scripts_dir + 'config.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)  


    try:
        trajopt = TrajectoryOptimization(args, rate=30, hyperparams=hyperparams)
        rospy.init_node('trajectory_optimization')
        
        LOGGER.debug('Started trajectory optimization node')

        while not rospy.is_shutdown():
            """
            optimize_trajectories = threading.Thread(
            target=lambda: trajopt.do_traj_opt()
            )
            optimize_trajectories.daemon = True
            optimize_trajectories.start()
            """

            trajopt.do_traj_opt()


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
