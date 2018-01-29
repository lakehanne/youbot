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
import scipy.ndimage as sp_ndimage
from collections import namedtuple
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
parser.add_argument('--plot_state', '-ps', action='store_true', default=False,
                        help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_true', default=True,
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

        self.hyperparams      = hyperparams
        config                = self.hyperparams.config
        self.T                = config['agent']['T']
        self.dU               = config['agent']['dU']
        self.dX               = config['agent']['dX']
        self.euler_step       = config['agent']['euler_step']
        self.euler_iter       = config['agent']['euler_iter']
        self.goal_state       = config['agent']['goal_state']
        self.action_penalty   = config['cost_params']['action_penalty']
        self.state_penalty    = config['cost_params']['state_penalty']
        self.wheel_rad        = self.wheel['radius']

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')

        self.fig = plt.figure()
        self._ax = self.fig.gca()
        plt.ioff()

    def generate_noise(self, T, dU, agent):
        """
        Generate a T x dU gaussian-distributed noise vector. This will
        approximately have mean 0 and variance 1, ignoring smoothing.

        Args:
            T: Number of time steps.
            dU: Dimensionality of actions.
        agent:
            smooth: Whether or not to perform smoothing of noise.
            var : If smooth=True, applies a Gaussian filter with this
                variance.
            renorm : If smooth=True, renormalizes data to have variance 1
                after smoothing.
        """
        smooth, var = agent['smooth_noise'], agent['smooth_noise_var']
        renorm = agent['smooth_noise_renormalize']
        noise = np.random.randn(T, dU)
        if smooth:
            # Smooth noise. This violates the controller assumption, but
            # might produce smoother motions.
            for i in range(dU):
                noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
            if renorm:
                variance = np.var(noise, axis=0)
                noise = noise / np.sqrt(variance)
        return noise

    def get_action_cost_jacs(self, delta_state, delta_action, action_nominal):
        l       = 0.5 *  np.sum(self.action_penalty * (delta_action ** 2), axis=0) + \
                        self.stateself.goal_state
        l_nom   = 0.5 *  np.sum(self.action_penalty * (action_nominal ** 2), axis=0) # nominal cost about linear traj
        lu  = self.action_penalty * delta_action
        lx  = np.zeros((self.dX))
        luu = np.tile(np.diag(self.action_penalty), [1, 1])
        lxx = np.zeros((self.dX, self.dX))
        lux = np.zeros((self.dU, self.dX))

        # generate random noise
        noise = self.generate_noise(self.T, self.dU, self.hyperparams.config['agent'])


        # assemble the cost's first order moments ell
        # pass
        CostJacs = namedtuple('CostJac', ['l', 'lx', 'lu', 'lxx', 'l_nom', \
                                          'luu', 'lux', 'noise'], verbose=False)
        return CostJacs


    def do_traj_opt(self, noisy=False):
        T = self.T
        dU = self.dU
        dX = self.dX

        # assemble Jacobian of transfer matrix and et cetera
        action          = np.zeros((T, dU))
        action_nominal  = np.zeros((T, dU))
        delta_action    = np.zeros((T, dU))
        noise_covar     = np.zeros((T))  # covariance of dynamics brownian motion

        # state allocations
        state           = np.zeros((T, dX))
        nominal_state   = np.zeros((T, dX))
        nominal_state_  = np.zeros((self.euler_iter, dX)) # euler integration. Lordy, hope I'm right
        delta_state     = np.zeros((T, dX))
        delta_state_plus = np.zeros((T, dX))
        goal_state      = np.zeros((T, dX)) # desired state

        fx              = np.zeros((T, dX, dX))
        fu              = np.zeros((T, dX, dU))

        # assemble stage_costs

        # Allocate.
        Vxx = np.zeros((T, dX, dX))
        Vx  = np.zeros((T, dX))
        Qtt = np.zeros((T, dX+dU, dX+dU))
        Qt  = np.zeros((T, dX+dU))

        action_zero, state_zero = np.zeros_like(action[0,:]), np.zeros_like(state[0,:])

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
            action[t,:]        = multiplier.dot(inv_dyn_eq)
            state[t,:]         = q #np.r_[q, qvel]
            goal_state[t,:]    = self.goal_state

            # see a generalized ILQG paper in 43rd CDC section V.2
            fx[t,:,:]          = np.eye(self.dX) -self.euler_step * np.linalg.inv(mass_matrix).dot(coriolis_matrix)
            fu[t,:,:]          = -(self.euler_step * self.wheel_rad) * np.linalg.inv(mass_matrix).dot(B_matrix.T)

            LOGGER.debug("Integrating inverse dynamics equation")
            # compute rhs at u(0), x_bar(0)
            mass_inv            = -np.linalg.inv(mass_matrix)
            for n in range(1, self.euler_iter):
                nominal_state_[n,:] = nominal_state_[n-1,:] + \
                                self.euler_step * (mass_inv.dot(coriolis_matrix).dot(nominal_state[t,:]) - \
                                mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector) + \
                                mass_inv.dot(B_matrix.T).dot(action_nominal[t,:])/self.wheel_rad)
            nominal_state[t,:] = nominal_state_[n,:] # decode nominal state at last euler step

            if self.args.plot_state:
                self._ax.plot(nominal_state[t:,:], 'b', label='qpos', fontweight='bold')
                # self._ax.plot(tt, nominal_state[:,1], 'g', label='qpos', fontweight='bold')
                self._ax.legend(loc='best')
                self._ax.set_xlabel('time (discretized)', fontweight='bold')
                self._ax.set_ylabel('final q after integration', fontweight='bold')
                self._ax.grid()
                self._ax.gcf().set_size_inches(10,4)
                self._ax.cla()

                if self.args.save_figs:
                    figs_dir = os.path.join(self.path, 'figures')
                    os.mkdir(figs_dir) if not os.path.exists(figs_dir) else None
                    self.fig.savefig(figs_dir + '/state_' + repr(t),
                            bbox_inches='tight',facecolor='None')

            # calculate new system trajectories
            delta_state[t,:]  = state[t,:]  - nominal_state[t,:]
            delta_action[t,:] = action[t,:] - action_nominal[t,:]

            if noisy:
                noise_covar[t] = (delta_state[t,:] - np.mean(delta_state[t,:])).T.dot(\
                                    delta_state[t,:] - np.mean(delta_state[t,:]))

            stage_jacs = self.get_action_cost_jacs(delta_state[t,:], delta_action[t,:], \
                                action_nominal[t,:])

            # assemble LQG state approx and cost
            if noisy:
                delta_state_plus[t,:] = fx[t,:,:].dot(delta_state[t,:]) + \
                                        fu[t,:,:].dot(delta_action[t,:]) + \
                                        stage_jacs.noise
            else:
                delta_state_plus[t,:] = fx[t,:,:].dot(delta_state[t,:]) + \
                                        fu[t,:,:].dot(delta_action[t,:])

            # form ell(x, u) # section A, 2-player game
            ell =   delta_state.T.dot(stage_jacs.lx) + \
                    delta_action.T.dot(stage_jacs.lu) + \
                    0.5 * delta_state.T.dot(stage_jacs.lxx).dot(delta_state) + \
                    0.5 * delta_action.T.dot(stage_jacs.luu).dot(delta_action) + \
                    delta_action.T.dot(stage_jacs.lux).dot(delta_state) + \
                    stage_jacs.l_nom



    def run_backward_pass(self):
        raise NotImplementedError("Must be implemented in derived class.")
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
