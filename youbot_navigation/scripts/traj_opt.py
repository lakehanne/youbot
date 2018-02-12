#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

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
# import pathos.pools as pp

import rospkg
from rospy.rostime import Duration

from geometry_msgs.msg import Wrench, Twist

# import torque and wrench services
from scripts.services import send_joint_torques, clear_active_forces, \
                             send_body_wrench, clear_active_wrenches

from numpy.linalg import LinAlgError
from collections import namedtuple
from scipy.integrate import odeint, RK45

from scripts.dynamics import Dynamics
from scripts.sample import SampleList
from scripts.algorithm_utils import IterationData, TrajectoryInfo, \
                            generate_noise, CostInfo

from multiprocessing import Pool, Process

# import matplotlib as mpl
# mpl.use('QT4Agg')
# import matplotlib.pyplot as plt

# fix seed
np.random.seed(0)

roslib.load_manifest('youbot_navigation')

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
                        help='max num iterations' )
parser.add_argument('--plot_state', '-ps', action='store_true', default=False,
                        help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_false', default=True,
                        help='save plotted figures' )
parser.add_argument('--silent', '-si', action='store_true', default=True,
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
    """
    def __init__(self, arg, rate, config):
        super(TrajectoryOptimization, self).__init__(Dynamics(rate))

        self.args            = arg
        # self.pool_derivatives= Pool(processes=2)

        # config                = hyperparams.config
        self.T                = config['agent']['T']
        self.dU               = config['agent']['dU']
        self.dX               = config['agent']['dX']
        self.euler_step       = config['agent']['euler_step']
        self.euler_iter       = config['agent']['euler_iter']
        self.goal_state       = config['agent']['goal_state']
        self.l21_const        = config['agent']['alpha']
        self.action_penalty   = config['cost_params']['action_penalty']
        self.state_penalty    = config['cost_params']['state_penalty']
        self.TOL              = config['agent']['TOL']
        # backpass regularizers
        self.mu               = config['agent']['mu']
        self.delta            = config['agent']['delta']
        self.mu_min           = config['agent']['mu_min']
        self.delta_nut        = config['agent']['delta_nut']
        # self.traj_info.nominal_action[:,] = config['trajectory']['init_action']

        self.config           = config


        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')
        self.pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # plt.ioff()
        # self.fig = plt.figure()
        # self._ax = self.fig.gca()
        self.linesearch_param = 1 # this is alpha in synthesis paper

    def get_state_rhs_eq(bdyn, x, u):
        M, C, B, S, f = bdyn.M, bdyn.C, bdyn.B, bdyn.S, bdyn.f
        Minv          = np.linalg.inv(M)
        rhs  = - Minv.dot(C).dot(x) - Minv.dot(B.T).dot(S.dot(f)) \
                         + Minv.dot(B.T).dot(u)/self.wheel['radius']

        return rhs

    def get_traj_cost_info(self, noisy=False):

        T, dU, dX = self.T, self.dU, self.dX

        sample        = self.get_samples(noisy)
        cur           = IterationData()

        cur.traj_info = sample['traj_info']
        cur.cost_info = sample['cost_info']

        # if self.args.plot_state:
        #     self._ax.plot(traj_info.nominal_state[t:,:], 'b', label='qpos', fontweight='bold')
        #     # self._ax.plot(tt, nominal_state[:,1], 'g', label='qpos', fontweight='bold')
        #     self._ax.legend(loc='best')
        #     self._ax.set_xlabel('time (discretized)', fontweight='bold')
        #     self._ax.set_ylabel('final q after integration', fontweight='bold')
        #     self._ax.grid()
        #     self._ax.gcf().set_size_inches(10,4)
        #     self._ax.cla()

        #     if self.args.save_figs:
        #         figs_dir = os.path.join(self.path, 'figures')
        #         os.mkdir(figs_dir) if not os.path.exists(figs_dir) else None
        #         self.fig.savefig(figs_dir + '/state_' + repr(t),
        #                 bbox_inches='tight',facecolor='None')

        return cur

    def reg_sched(self, increase=False):
        # increase mu
        if increase:
            self.delta = max(self.delta_nut, self.delta * self.delta_nut)
            mu = self.mu * 1.1 #max(self.mu_min, self.mu * self.delta) #
        else:
            mu = self.mu
            # self.delta = min(1/self.delta_nut, self.delta/self.delta_nut)
            # if mu * self.delta > self.mu_min:
            #     mu = mu * self.delta
            # else:
            #     mu = 0
            mu *= 0.09
        self.mu = mu

    def do_traj_opt(self, sample_info, stop_cond, noisy=False):
        T = self.T
        c = 0.5  # see DDP pg 113

        eta = 8e3

        while eta > stop_cond:

            prev_sample_info = self.backward(sample_info, noisy)

            # rospy.logdebug('running forward pass')
            new_sample_info      = self.forward(prev_sample_info, noisy)

            ############################################################################
            # check if integration of state diverged from previous iteration
            # see synthesis paper section D
            gu_expand = np.expand_dims(prev_sample_info.traj_info.gu, 2)
            Qu_expand = np.expand_dims(prev_sample_info.cost_info.Qu, 2)
            Quu       = prev_sample_info.cost_info.Quu

            J_t1, J_t2 = 0, 0
            for i in range(T-1):
                J_t1 += gu_expand[i,:,:].T.dot(Qu_expand[i,:,:])
                J_t2 += gu_expand[i,:,:].T.dot(Quu[i,:,:]).dot(gu_expand[i,:,:])

            alpha = self.linesearch_param
            cost_change_scale     = (alpha * J_t1) + ((alpha**2)/(2 * J_t2))
            # cause I added l and l_nom
            J_prev_traj = np.sum(prev_sample_info.cost_info.l[:-2] \
                                - prev_sample_info.cost_info.l_nom[:-2], axis=0) \
                            + prev_sample_info.cost_info.l[-1] \
                            - prev_sample_info.cost_info.l_nom[-1] # see DDP pg 113
            #############################################################################
            # calculate J for estimated cost and trajectory samples
            gu_expand = np.expand_dims(new_sample_info.traj_info.gu, 2)
            Qu_expand = np.expand_dims(new_sample_info.cost_info.Qu, 2)
            Quu       = new_sample_info.cost_info.Quu

            J_t1, J_t2 = 0, 0
            for i in range(T-1):
                J_t1 += gu_expand[i,:,:].T.dot(Qu_expand[i,:,:])
                J_t2 += gu_expand[i,:,:].T.dot(Quu[i,:,:]).dot(gu_expand[i,:,:])

            # cost_change_scale     = (alpha * J_t1) + ((alpha**2)/2 * J_t2)
            J_curr_traj = np.sum(new_sample_info.cost_info.l[:-2] \
                            - new_sample_info.cost_info.l_nom[:-2] , axis=0) \
                            + new_sample_info.cost_info.l[-1] \
                            - new_sample_info.cost_info.l_nom[-1]

            # find z
            cost_change = (J_prev_traj - J_curr_traj)/cost_change_scale

            # accept the iteration opnly if 0 < c1 < z
            if cost_change < c and cost_change > 0: # accept the trajectory

                # ros com params
                start_time = Duration(secs = 0, nsecs = 0) # start asap
                duration = Duration(secs = 6, nsecs = 0) # apply effort continuously without end duration = -1
                reference_frame = None #'empty/world/map'

                wrench_base = Wrench()
                base_angle = Twist()

                rospy.loginfo("Found suitable trajectory. Applying found control law")

                for t in range(T):
                    # send the computed torques to ROS
                    torques = new_sample_info.traj_info.action[t,:]

                    # calculate the genralized force and torques
                    bdyn = self.assemble_dynamics()
                    theta = bdyn.q[-1]

                    sign_phi = np.sign(self.Phi_dot)
                    F1 = (torques[0] - self.wheel['radius'] * sign_phi[0] * bdyn.f[0]) * \
                            (-(np.cos(theta) - np.sin(theta))/self.wheel['radius']) + \
                         (torques[1] - self.wheel['radius'] * sign_phi[1] * bdyn.f[0]) * \
                            (-(np.cos(theta) + np.sin(theta))/self.wheel['radius'])  + \
                         (torques[2] - self.wheel['radius'] * sign_phi[2] * bdyn.f[0]) * \
                            ((np.cos(theta) - np.sin(theta))/self.wheel['radius'])  + \
                         (torques[3] - self.wheel['radius'] * sign_phi[3] * bdyn.f[0]) * \
                            ((np.cos(theta) + np.sin(theta))/self.wheel['radius'])
                    F2 = (torques[0] - self.wheel['radius'] * sign_phi[0] * bdyn.f[0]) * \
                            (-(np.cos(theta) + np.sin(theta))/self.wheel['radius']) + \
                         (torques[1] - self.wheel['radius'] * sign_phi[1] * bdyn.f[0]) * \
                            (-(-np.cos(theta) + np.sin(theta))/self.wheel['radius'])  + \
                         (torques[2] - self.wheel['radius'] * sign_phi[2] * bdyn.f[0]) * \
                            ((np.cos(theta) + np.sin(theta))/self.wheel['radius'])  + \
                         (torques[3] - self.wheel['radius'] * sign_phi[3] * bdyn.f[0]) * \
                            ((-np.cos(theta) + np.sin(theta))/self.wheel['radius'])
                    F3 = np.sum(torques) * (-np.sqrt(2)*self.l * np.sin( np.pi/4 - self.alpha)/self.wheel['radius'])  \
                           + (sign_phi[0] * bdyn.f[0] + sign_phi[1] * bdyn.f[0] + sign_phi[2] * bdyn.f[0] + sign_phi[3] * bdyn.f[0]) \
                                * (np.sqrt(2)* self.l * np.sin(np.pi/4 - self.alpha))

                    rospy.loginfo('F1: {}, F2: {}, F3: {}'.format(F1, F2, F3))
                    wrench_base.force.x = F1
                    wrench_base.force.y = F2
                    base_angle.angular.z = F3

                    # send the torques to the base footprint
                    # self.pub.publish(base_angle)
                    send_body_wrench('base_footprint', reference_frame,
                                                    None, wrench_base, start_time,
                                                    duration )

                    rospy.sleep(duration)

                    clear_active_wrenches('base_footprint')

                # set ubar_i = u_i, xbar_i = x_i and repeat traj_opt # step 5 DDP book
                new_sample_info.traj_info.nominal_action = new_sample_info.traj_info.action
                new_sample_info.traj_info.nominal_state  = new_sample_info.traj_info.state

                eta = np.linalg.norm(new_sample_info.cost_info.Vx, ord=2)

                rospy.loginfo('Eta decreased. New Eta: %.4f', eta)
                # repeat trajectory optimization process if eta does not meet stopping criteria
                sample_info = new_sample_info

            else: # repeat back+forward pass if integration diverged from prev traj by so much
                rospy.loginfo('Repeating traj opt phase: iteration not accepted')
                self.linesearch_param = self.linesearch_param - self.linesearch_param * 0.01

            # rospy.loginfo('Finished Trajectory optimization process')

    def backward(self, sample_info, noisy=False):
        T  = self.T
        dU = self.dU
        dX = self.dX

        cost_info = sample_info.cost_info
        traj_info = sample_info.traj_info
        # mu        = self.mu

        non_pos_def = True
        while (non_pos_def):

            non_pos_def = False
            # rospy.logdebug('Restarting back pass')

            for t in range (T-1, -1, -1):
                """
                get derivatives in a different execution thread. Following Todorov's
                recommendation in synthesis and stabilization paper
                """

                # retrieve the erstwhile costs from future T
                # stage_jacs = self.get_traj_cost_info(noisy=False)

                # p = self.pool_derivatives.apply_async(
                #                         self.get_traj_cost_info,
                #                         args=(False)
                #                         )

                cost_info.Qx[t,:]     = cost_info.lx[t]
                cost_info.Qu[t,:]     = cost_info.lu[t]
                cost_info.Qxx[t,:,:]  = cost_info.lxx[t]
                cost_info.Qux[t,:,:]  = cost_info.lux[t]
                cost_info.Quu[t,:,:]  = cost_info.luu[t]
                cost_info.Quu_tilde[t,:,:]  = cost_info.luu[t]

                # form Q derivatives at time t first
                if t < T-1:
                    # wil be --> (3) + (3x3) x (3) ==> (3)
                    cost_info.Qx[t,:] = cost_info.lx[t] + traj_info.fx[t,:].T.dot(cost_info.Vx[t+1,:])
                    # wil be --> (4) + (4,3) x (3) ==> (4)
                    cost_info.Qu[t,:] = cost_info.lu[t] + traj_info.fu[t,:].T.dot(cost_info.Vx[t+1,:])
                    # wil be --> (3) + (3,3) x (3,3) x ((3,3)) ==> (3,3)
                    cost_info.Qxx[t,:,:]  = cost_info.lxx[t] + traj_info.fx[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fx[t,:])
                    # wil be --> (4, 3) + (4,3) x (3,3) x ((3,3)) ==> (4,3)
                    cost_info.Qux[t,:,:]  = cost_info.lux[t] + traj_info.fu[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fx[t,:])
                    # wil be --> (4, 4) + (4,3) x (3,3) x ((3,4)) ==> (4,4)
                    cost_info.Quu[t,:,:]  = cost_info.luu[t] + traj_info.fu[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fu[t,:])

                    # calculate the Qvals that penalize devs from the states
                    cost_info.Qu_tilde[t,:] = cost_info.Qu[t,:] + self.mu * np.ones(dU)
                    cost_info.Quu_tilde[t,:,:] = cost_info.luu[t] + traj_info.fu[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fu[t,:])
                    cost_info.Qux_tilde[t,:,:] = cost_info.lux[t] + traj_info.fu[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fx[t,:]) #+ \
                                    #traj_info.fux[t,:,:].dot(cost_info.Vx[t+1,:])

                # symmetrize the second order moments of Q
                cost_info.Quu[t,:,:] = 0.5 * (cost_info.Quu[t].T + cost_info.Quu[t])
                cost_info.Qxx[t,:,:] = 0.5 * (cost_info.Qxx[t].T + cost_info.Qxx[t])

                # symmetrize for improved Q state improvement values too
                cost_info.Quu_tilde[t,:,:] = 0.5 * (cost_info.Quu_tilde[t].T + \
                            cost_info.Quu_tilde[t])
                # Compute Cholesky decomposition of Q function action component.
                try:
                    U_tilde = sp.linalg.cholesky(cost_info.Quu_tilde[t,:,:])
                    L_tilde = U_tilde.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u]  or Q_tilde[u, u] is not spd
                    rospy.loginfo('LinAlgError: %s', e)
                    non_pos_def = True
                    # restart the backward pass if Quu is non positive definite
                    break

                # compute open and closed loop gains.
                small_k   = -sp.linalg.solve_triangular(
                                U_tilde, sp.linalg.solve_triangular(L_tilde,
                                cost_info.Qu_tilde[t, :], lower=True)
                            )
                big_K = -sp.linalg.solve_triangular(
                                U_tilde, sp.linalg.solve_triangular(L_tilde,
                                cost_info.Qux_tilde[t, :, :], lower=True)
                            )
                # store away the gains
                traj_info.gu[t, :] = small_k
                traj_info.Gu[t, :, :] = big_K

                # calculate improved value functions
                cost_info.Vxx[t,:,:] = cost_info.Qxx[t, :,:] \
                                        + big_K.T.dot(cost_info.Quu[t,:,:]).dot(big_K) \
                                        + big_K.T.dot(cost_info.Qux[t,:,:] \
                                        ) + cost_info.Qux[t,:,:].T.dot(traj_info.Gu[t,:,:])

                cost_info.Vx[t,:] = cost_info.Qx[t,:] \
                                    + big_K.T.dot(cost_info.Quu[t,:,:]).dot(small_k) \
                                    + big_K.T.dot(cost_info.Qu[t,:]) \
                                    + traj_info.gu[t,:].T.dot(cost_info.Qux[t,:,:])

                cost_info.V[t] = 0.5 * np.expand_dims(small_k, 1).T.dot(\
                                    cost_info.Quu[t,:,:]).dot(np.expand_dims(small_k, 1))  \
                                    +  np.expand_dims(small_k, 1).T.dot(cost_info.Qu[t,:])

                # symmetrize quadratic Value hessian
                cost_info.Vxx[t,:,:] = 0.5 * (cost_info.Vxx[t,:,:] + cost_info.Vxx[t,:,:].T)

                # print('Qu_tilde: \n', cost_info.Qu_tilde[t])
                # print('Quu_tilde: \n', cost_info.Quu_tilde[t,:])
                # print('Qux_tilde: \n', cost_info.Qux_tilde[t,:])
                # print('V: ', cost_info.V[t])
                # print('Vx: \n', cost_info.Vx[t])
                # print('Vxx: \n', cost_info.Vxx[t])
                # time.sleep(2)

            if non_pos_def: # restart the back-pass process
                old_mu = self.mu
                self.reg_sched(increase=True)
                rospy.logdebug("Hessian became non positive definite")
                rospy.logdebug('Increasing mu: {} -> {}'.format(old_mu, self.mu))
                break
            else:
                # if successful, decrese mu
                old_mu = self.mu
                self.reg_sched(increase=False)
                rospy.logdebug('Decreasing mu: {} -> {}'.format(old_mu, self.mu))

        # update sample_info
        sample_info.cost_info = cost_info
        sample_info.traj_info = traj_info

        return sample_info

    def forward(self, prev_sample_info, noisy):
        T  = self.T
        dU = self.dU
        dX = self.dX

        # forward pass params
        alpha = self.linesearch_param
        delta_t = T/(T-1) # euler step

        # get samples used in the backward pass
        traj_info = prev_sample_info.traj_info
        cost_info = prev_sample_info.cost_info

        # start fwd pass
        for t in range(1, T-1):
            # update the nominal action that was assumed
            traj_info.nominal_action[t,:] = traj_info.nominal_action[t,:] \
                                                + alpha * traj_info.gu[t, :] \
                                                + traj_info.Gu[t, :].dot(traj_info.nominal_state[t,:])

            traj_info.delta_action[t,:] = traj_info.delta_action[t,:] \
                                                + alpha * traj_info.gu[t, :] \
                                                + traj_info.Gu[t, :].dot(traj_info.delta_state[t,:])
            """
            b_dyn   = self.assemble_dynamics()

            # get nominal state and action rhs
            nom_rhs = self.get_state_rhs_eq(b_dyn, traj_info.nominal_state[t,:],
                                    traj_info.nominal_action[t,:])
            # # get nonlinear state and action rhs
            nlr_rhs = self.get_state_rhs_eq(b_dyn, traj_info.state[t,:],
                                    traj_info.action[t,:])
            # # get diff rhs
            # delt_rhs = self.get_state_rhs_eq(b_dyn, traj_info.delta_state[t,:],
            #                         traj_info.delta_action[t,:])

            # update state at t+1
            traj_info.nominal_state[t+1,:]  = traj_info.nominal_state[t,:] \
                                                + delta_t * nom_rhs
            # update state at t+1
            traj_info.state[t+1,:]  = traj_info.state[t,:] \
                                                + delta_t * nlr_rhs
            # calculate delta_state
            traj_info.delta_state[t,:] = traj_info.state[t,:] - traj_info.nominal_state[t,:]

            # update cost to go terms
            delu_exp = self.expand_array(traj_info.delta_action[t,:], 0)
            delx_exp = self.expand_array(traj_info.delta_state[t,:]-delta_x_star, 0)
            cost_action_term = 0.5 * self.action_penalty[0] * delu_exp.T.dot(delu_exp)
            cost_state_term  = 0.5 * self.state_penalty[0]  * delx_exp.T.dot(delx_exp)

            # nominal cost terms
            ubar_exp = self.expand_array(traj_info.nominal_state[t,:], 0)
            xbar_exp = self.expand_array(x_bar[t,:] - delta_x_star, 0)
            cost_nom_action_term = 0.5 * self.action_penalty[0] * ubar_exp.T.dot(ubar_exp)
            cost_nom_state_term  = 0.5 * self.state_penalty[0] * xbar_exp.T.dot(xbar_exp)

            l[t]       = cost_action_term + cost_state_term
            l_nom[t]   = cost_nom_action_term + cost_nom_state_term #+ cost_nom_l12_term

            # I think we need to do this
            l[t] += l_nom[t] # then it becomes the nonlinear cost functional

            # first order cost terms
            lu[t]      = self.action_penalty[0] * traj_info.delta_action[t,:]
            lx[t]      = self.state_penalty[0]  * traj_info.delta_state[t,:]

                        # 2nd order cost terms
            luu[t]     = np.diag(self.action_penalty)
            lux[t]     = np.zeros((self.dU, self.dX))

            # # calculate the jacobians
            Minv       = np.linalg.inv(b_dyn.M)
            fx[t]      = np.eye(self.dX) - delta_t * Minv.dot(b_dyn.C)
            fu[t]      = -(delta_t * self.wheel['radius']) * Minv.dot(b_dyn.B.T)

            traj_info.fx[t,:]            = fx[t,:]
            traj_info.fu[t,:]            = fu[t,:]
            traj_info.action[t,:]        = u[t,:]
            traj_info.state[t,:]         = x[t,:]
            traj_info.delta_state[t,:]   = delta_x[t,:]
            traj_info.delta_action[t,:]  = delta_u[t,:]
            traj_info.nominal_state[t,:] = x_bar[t,:]
            traj_info.nominal_action[t,:]= u_bar[t,:]

            cost_info.l[t]   = l[t]
            cost_info.lu[t]  = lu[t]
            cost_info.lx[t]  = lx[t]
            cost_info.lux[t] = lux[t]
            cost_info.luu[t] = luu[t]
            cost_info.lxx[t] = lxx[t]
            """

        #update the traj_info
        # new_sample_info = IterationData()

        prev_sample_info.traj_info = traj_info
        # prev_sample_info.cost_info = cost_info

        return prev_sample_info


if __name__ == '__main__':

    from scripts import __file__ as scripts_filepath
    scripts_filepath = os.path.abspath(scripts_filepath)
    scripts_dir = '/'.join(str.split(scripts_filepath, '/')[:-1]) + '/'
    hyperparams_file = scripts_dir + 'config.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.silent:
        log_level = rospy.INFO
    else:
        log_level = rospy.DEBUG
    try:

        trajopt = TrajectoryOptimization(args, rate=30, config=hyperparams.config)
        rospy.init_node('trajectory_optimization',
                        disable_signals=False, anonymous=True,
                        log_level=log_level)

        rospy.logdebug('Started trajectory optimization node')

        if not rospy.is_shutdown():
            # optimize_trajectories = threading.Thread(
            # target=lambda: trajopt.do_traj_opt()
            # )
            # optimize_trajectories.daemon = True
            # optimize_trajectories.start()

            # on first iteration, obtain trajectory samples from the robot
            sample_info = trajopt.get_traj_cost_info(noisy=False)

            # print('traj info: ', sample_info.traj_info.nominal_action,
            #     sample_info.cost_info.l)
            # time.sleep(40)

            stop_cond = hyperparams.config['agent']['TOL']
            trajopt.do_traj_opt(sample_info, stop_cond, noisy=False)


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
