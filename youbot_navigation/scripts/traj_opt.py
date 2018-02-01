#!/usr/bin/env python
# coding=utf-8

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
from numpy.linalg import LinAlgError
from nav_msgs.msg import Odometry
import scipy.ndimage as sp_ndimage
from collections import namedtuple
from scipy.integrate import odeint, RK45

from scripts.dynamics import Dynamics
from scripts.sample import SampleList
from scripts.algorithm_utils import TrajectoryInfo

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
        self.alpha            = config['agent']['alpha']
        self.action_penalty   = config['cost_params']['action_penalty']
        self.state_penalty    = config['cost_params']['state_penalty']
        self.wheel_rad        = self.wheel['radius']
        self._samples         = [[] for _ in range(config['agent']['conditions'])]

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')

        self.fig = plt.figure()
        self._ax = self.fig.gca()
        plt.ioff()

        # self.traj_distr = namedtuple('TrajDistr', ['Vx', 'Vxx', 'Qx', \
        #             'Qu', 'Qxx', 'Qux', 'Quu', \
        #             'fx', 'fu',  'action', 'action_nominal', 'delta_action', \
        #             'state', 'nominal_state', 'delta_state', 'delta_state_plus', \
        #             'gu', 'Gu'],
        #             verbose=False)
        self.traj_distr =  TrajectoryInfo(config)

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

    def get_action_cost_jacs(self,  B_matrix, t):
        state_diff     = np.expand_dims(self.traj_distr.delta_state[t,:] - self.goal_state, 1)
        state_diff_nom = np.expand_dims((self.traj_distr.nominal_state[t,:] - self.goal_state), 1)

        cost_action_term = self.action_penalty[0] * np.expand_dims(self.traj_distr.delta_action[t,:], axis=0).dot(\
                            np.expand_dims(self.traj_distr.delta_action[t,:], axis=1))
        cost_state_term  = 0.5 * self.state_penalty[0] * state_diff.T.dot(state_diff)
        cost_l12_term    = np.sqrt(self.alpha + state_diff.T.dot(state_diff))                            

        # nominal action
        cost_nom_action_term = self.action_penalty[0] * np.expand_dims(self.traj_distr.action_nominal[t,:], axis=0).dot(\
                                                        np.expand_dims(self.traj_distr.action_nominal[t,:], axis=1))
        cost_nom_state_term  = 0.5 * self.state_penalty[0] * state_diff_nom.T.dot(state_diff_nom)
        cost_nom_l12_term    = np.sqrt(self.alpha + state_diff_nom.T.dot(state_diff_nom))  

        #system cost
        l       = cost_action_term + cost_state_term + cost_l12_term
        # nominal cost about linear traj
        l_nom   = cost_nom_action_term + cost_nom_state_term + cost_nom_l12_term
        # first order derivatives
        lu      = self.action_penalty * self.traj_distr.delta_action[t,:]

        lx      = state_diff.dot(
                                    self.state_penalty + \
                                    np.tile(1/np.sqrt(self.alpha + state_diff.T.dot(state_diff)), [self.dX])
                                )
        luu     = np.diag(self.action_penalty)

        lxx_t1 = np.diag(self.state_penalty) 
        lxx_t2 = np.eye(self.dX)
        lxx_t3 = (state_diff.T.dot(state_diff)) / ((self.alpha + state_diff.T.dot(state_diff))**3) 
        lxx    = lxx_t1 +  lxx_t2 * np.eye(self.dX) - lxx_t3 * np.eye(self.dX)

        lux = np.zeros((self.dU, self.dX))

        # generate random noise
        noise = self.generate_noise(self.T, self.dU, self.hyperparams.config['agent'])


        # assemble the cost's first order moments ell
        # pass
        CostJacs = namedtuple('CostJac', ['l', 'lx', 'lu', 'lxx', 'l_nom', \
                                          'luu', 'lux', 'noise'], verbose=False)
        return CostJacs

    def do_traj_opt(self):

        self.backward(noisy=False)

    def backward(self, noisy=False):
        T  = self.T
        dU = self.dU
        dX = self.dX

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
            self.traj_distr.action[t,:]        = multiplier.dot(inv_dyn_eq).squeeze()
            self.traj_distr.state[t,:]         = q.squeeze() #np.r_[q, qvel]

            """
            see a generalized ILQG paper in 43rd CDC section V.
            the euler step multiplier takes the ode equations it to first order derivatives from 2nd order change of vars
            """
            self.traj_distr.fx[t,:,:]          = np.eye(self.dX) -self.euler_step * np.linalg.inv(mass_matrix).dot(coriolis_matrix)
            self.traj_distr.fu[t,:,:]          = -(self.euler_step * self.wheel_rad) * np.linalg.inv(mass_matrix).dot(B_matrix.T)

            LOGGER.debug("Integrating inverse dynamics equation")

            mass_inv            = -np.linalg.inv(mass_matrix)
            # print('2nd term: {}'.format(mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector).squeeze().shape))
            # print('into: {}'.format(self.traj_distr.nominal_state_[1,:].shape ))
            # print('first term: {}, 2nd term: {} all mult: {}'.format(mass_inv.dot(B_matrix.T).shape, \
            #         self.traj_distr.action_nominal[t,:].shape, ((mass_inv.dot(B_matrix.T).dot(self.traj_distr.action_nominal[t,:]))/self.wheel_rad).shape))
            for n in range(1, self.euler_iter):
                self.traj_distr.nominal_state_[n,:] = self.traj_distr.nominal_state_[n-1,:] + \
                                self.euler_step * (mass_inv.dot(coriolis_matrix).dot(self.traj_distr.nominal_state[t,:])) - \
                                mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector)+ \
                                (mass_inv.dot(B_matrix.T).dot(self.traj_distr.action_nominal[t,:]))/self.wheel_rad
            self.traj_distr.nominal_state[t,:] = self.traj_distr.nominal_state_[n,:] # decode nominal state at last euler step

            if self.args.plot_state:
                self._ax.plot(self.traj_distr.nominal_state[t:,:], 'b', label='qpos', fontweight='bold')
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
            self.traj_distr.delta_state[t,:]  = self.traj_distr.state[t,:]  - self.traj_distr.nominal_state[t,:]
            self.traj_distr.delta_action[t,:] = self.traj_distr.action[t,:] - self.traj_distr.action_nominal[t,:]

            if noisy:
                noise_covar[t] = (self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:])).T.dot(\
                                    self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:]))

            stage_jacs = self.get_action_cost_jacs(B_matrix, t)

            # assemble LQG state approx and cost
            if noisy:
                self.traj_distr.delta_state_plus[t,:] = self.traj_distr.fx[t,:,:].dot(self.traj_distr.delta_state[t,:]) + \
                                        self.traj_distr.fu[t,:,:].dot(self.traj_distr.delta_action[t,:]) + \
                                        stage_jacs.noise
            else:
                self.traj_distr.delta_state_plus[t,:] = self.traj_distr.fx[t,:,:].dot(self.traj_distr.delta_state[t,:]) + \
                                        self.traj_distr.fu[t,:,:].dot(self.traj_distr.delta_action[t,:])

            # form Q derivatives at time t first
            if t < T-1:
                # wil be --> (3) + (3x3) x (3) ==> (3)
                self.traj_distr.Qx[t,:] = stage_jacs.lx + self.traj_distr.fx[t,:].T.dot(self.traj_distr.Vx[t+1,:])
                # wil be --> (4) + (4,3) x (3) ==> (4)
                self.traj_distr.Qu[t,:] = stage_jacs.lu + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vx[t+1,:])
                # wil be --> (3) + (3,3) x (3,3) x ((3,3)) ==> (3,3)
                self.traj_distr.Qxx[t,:,:]  = stage_jacs.lxx + self.traj_distr.fx[t,:].T.dot(self.traj_distr.Vxx[t,:,:]).dot(self.traj_distr.fx[t,:])
                # wil be --> (4, 3) + (4,3) x (3,3) x ((3,3)) ==> (4,3)
                self.traj_distr.Qux[t,:,:]  = stage_jacs.lux + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vxx[t,:,:]).dot(self.traj_distr.fx[t,:])
                # wil be --> (4, 4) + (4,3) x (3,3) x ((3,4)) ==> (4,4)
                self.traj_distr.Quu[t,:,:]  = stage_jacs.luu + self.traj_distr.fu[t,:].T.dot(self.traj_distr.Vxx[t,:,:]).dot(self.traj_distr.fu[t,:])

            # symmetrize the second order moments of Q
            self.traj_distr.Quu[t,:,:] = 0.5 * (self.traj_distr.Quu[t].T + self.traj_distr.Quu[t])
            self.traj_distr.Qxx[t,:,:] = 0.5 * (self.traj_distr.Qxx[t].T + self.traj_distr.Qxx[t])

            # Compute Cholesky decomposition of Q function action component.
            try:
                U = sp.linalg.cholesky(self.traj_distr.Quu[t, :, :])
                L = U.T
            except LinAlgError as e:
                # Error thrown when Qtt[idx_u, idx_u] is not
                # symmetric positive definite.
                LOGGER.debug('LinAlgError: %s', e)
                # fail = t if self.cons_per_step else True
                break

            # compute open and closed loop gains.
            self.traj_distr.gu[t, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, self.traj_distr.Qu[t, :], lower=True)
            )
            self.traj_distr.Gu[t, :, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, self.traj_distr.Qux[t, :, :], lower=True)
            )

            # calculate value function
            self.traj_distr.Vxx[t,:,:] = self.traj_distr.Qxx[t, :,:] + self.traj_distr.Qux[t,:,:].T.dot(self.traj_distr.Gu[t,:,:])
            self.traj_distr.Vx[t,:] = self.traj_distr.Qx[t,:] + self.traj_distr.gu[t,:].T.dot(self.traj_distr.Qux[t,:,:])

            # symmetrize quadratic Value hessian
            self.traj_distr.Vxx[t,:,:] = 0.5 * (self.traj_distr.Vxx[t,:,:] + self.traj_distr.Vxx[t,:,:].T)

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
            target=lambda: trajopt.do_traj_opt(noisy=True)
            )
            optimize_trajectories.daemon = True
            optimize_trajectories.start()
            """

            trajopt.do_traj_opt()


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
