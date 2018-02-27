#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

import os
from os.path import join, expanduser
import imp
import time
import copy
import rospy
import rospkg
import roslib
import logging
import argparse
import threading
import numpy as np
import scipy as sp
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
from scripts.subscribers import PointCloudsReceiver

from multiprocessing import Pool, Process

import matplotlib as mpl
mpl.use('QT4Agg')
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

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

        # config                = hyperparams.config
        self.T                = config['agent']['T']
        self.dU               = config['agent']['dU']
        self.dV               = config['agent']['dV']
        self.dX               = config['agent']['dX']
        self.euler_step       = config['agent']['euler_step']
        self.euler_iter       = config['agent']['euler_iter']
        self.goal_state       = config['agent']['goal_state']
        self.TOL              = config['agent']['TOL']

        # backpass regularizers
        self.mu               = config['agent']['mu']
        self.delta            = config['agent']['delta']
        self.mu_min           = config['agent']['mu_min']
        self.delta_nut        = config['agent']['delta_nut']

        self.config           = config

        rp = rospkg.RosPack()
        self.path = rp.get_path('youbot_navigation')

        # mpl params
        plt.ioff()
        self.fig = plt.figure()
        self._ax = self.fig.gca()
        self.linesearch_param = config['agent']['linesearch_param']#1 # this is rho in synthesis paper
        self.first_iteration = False

        self.pcl_rcvr = PointCloudsReceiver(rate)

    def get_state_rhs_eq(self, bdyn, x, u):
        M, C, B, S, f = bdyn.M, bdyn.C, bdyn.B, bdyn.S, bdyn.f
        Minv          = np.linalg.inv(M)
        rhs  = - Minv.dot(C).dot(x) - Minv.dot(B.T).dot(S.dot(f)) \
                         + Minv.dot(B.T).dot(u)/self.wheel['radius']

        return rhs

    def get_traj_cost_info(self, noisy=False):

        sample        = self.get_samples(noisy)
        cur           = IterationData()

        cur.traj_info = sample['traj_info']
        cur.cost_info = sample['cost_info']

        if self.args.plot_state:
            plt.close('all')
            self._ax.plot(cur.cost_info.l, label="aprrox quadratized cost")
            # self._ax.plot(traj_info.nom_state[t:,:], 'b', label='qpos', fontweight='bold')
            # # self._ax.plot(tt, nom_state[:,1], 'g', label='qpos', fontweight='bold')
            self._ax.legend(loc='best')
            self._ax.set_xlabel('time (discretized)', fontweight='bold')
            self._ax.set_ylabel('cost', fontweight='bold')
            self._ax.grid()
            plt.gcf().set_size_inches(10,4)
            self._ax.cla()

            plt.show()

            if self.args.save_figs:
                figs_dir = os.path.join(self.path, 'figures')
                os.makedirs(figs_dir) if not os.path.exists(figs_dir) else None
                self.fig.savefig(figs_dir + '/cost.png',
                        bbox_inches='tight',facecolor='None')

        return cur

    def reg_sched(self, increase=False):
        # increase mu
        if increase:
            self.delta = max(self.delta_nut, self.delta * self.delta_nut)
            mu = self.mu * 1.1 #max(self.mu_min, self.mu * self.delta) #
        else: # decrease mu
            mu = self.mu
            # self.delta = min(1/self.delta_nut, self.delta/self.delta_nut)
            # if mu * self.delta > self.mu_min:
            #     mu = mu * self.delta
            # else:
            #     mu = 0
            mu *= 0.09
        self.mu = mu

    def do_traj_opt(self, sample_info, noisy=False, save=False):
        T = self.T

        save_dir = join(self.path, 'data')

        if save:
            os.makedirs(save_dir) if not os.path.exists(save_dir) else None
            savefile = save_dir + '/trajectory.txt'
            costfile = save_dir + '/quadratized_cost.txt'
            os.remove(savefile) if os.path.isfile(savefile) else None
            os.remove(costfile) if os.path.isfile(costfile) else None

        stop_cond       = self.config['trajectory']['stopping_condition']
        eta             = self.config['trajectory']['stopping_eta']
        duration_length = self.config['trajectory']['duration_length']

        traj_samples = [] #use this to store traj info
        c_zero = self.config['trajectory']['c_zero']

        tic    = time.time()

        wrench_base = Wrench()
        base_angle = Twist()

        goal_sign = np.sign(self.goal_state)

        rho = self.linesearch_param

        run = True

        while run:
            print('first sample: ', sample_info.cost_info.l[80])
            prev_sample_info = self.backward(sample_info, noisy)

            new_sample_info  = self.forward(prev_sample_info, noisy)
            traj_samples.append(new_sample_info)
            #################################################################
            # check if integration of state diverged from previous iteration
            # see synthesis paper section D
            #################################################################
            if len(traj_samples) > 1:
                prev_traj = traj_samples[-2]
                # protagonist terms
                gu_expand = np.expand_dims(prev_traj.traj_info.gu, 2)
                Qu_expand = np.expand_dims(prev_traj.cost_info.Qu, 2)
                Quu       = prev_traj.cost_info.Quu_tilde
                # adversarial terms
                gv_expand = np.expand_dims(prev_traj.traj_info.gv, 2)
                Qv_expand = np.expand_dims(prev_traj.cost_info.Qv, 2)
                Qvv       = prev_traj.cost_info.Qvv_tilde

                J_t1, J_t2 = 0, 0
                for i in range(T-1):
                    J_t1 = J_t1 + gu_expand[i,:,:].T.dot(Qu_expand[i,:,:]) + gv_expand[i,:,:].T.dot(Qv_expand[i,:,:])
                    J_t2 = J_t2 + gu_expand[i,:,:].T.dot(Quu[i,:,:]).dot(gu_expand[i,:,:]) + gv_expand[i,:,:].T.dot(Qvv[i,:,:]).dot(gv_expand[i,:,:])

                J_prev_rho = J_t1*rho + J_t2*(rho**2)/2
                J_prev  = (np.sum(prev_traj.cost_info.l[:-1], axis=0) \
                          + prev_traj.cost_info.l[-1]).squeeze() # see DDP pg 113
                
                #############################################################################
                # calculate J for estimated cost and trajectory samples
                gu_expand = np.expand_dims(new_sample_info.traj_info.gu, 2)
                Qu_expand = np.expand_dims(new_sample_info.cost_info.Qu, 2)
                Quu_tilde = new_sample_info.cost_info.Quu_tilde
                # adversarial terms
                gv_expand = np.expand_dims(new_sample_info.traj_info.gv, 2)
                Qv_expand = np.expand_dims(new_sample_info.cost_info.Qv, 2)
                Qvv       = new_sample_info.cost_info.Qvv_tilde

                J_t1, J_t2 = 0, 0
                for i in range(T-1):
                    J_t1 = J_t1 + gu_expand[i,:,:].T.dot(Qu_expand[i,:,:]) + gv_expand[i,:,:].T.dot(Qv_expand[i,:,:])
                    J_t2 = J_t2 + gu_expand[i,:,:].T.dot(Quu[i,:,:]).dot(gu_expand[i,:,:]) + gv_expand[i,:,:].T.dot(Qvv[i,:,:]).dot(gv_expand[i,:,:])

                J_curr_rho = J_t1*rho + J_t2*(rho**2)/2
                J_curr   = (np.sum(new_sample_info.cost_info.l[:-1], axis=0) \
                                                + new_sample_info.cost_info.l[-1]).squeeze()
                
                DeltaJ_rho = J_curr_rho.squeeze() + 1e-6#(J_prev_rho - J_curr_rho).squeeze() + 1e-6 # to remove -0
                DeltaJ   = (J_curr - J_prev ).squeeze()

                J_diff   = DeltaJ#/DeltaJ_rho

                # rospy.loginfo('DeltaJ_rho: {} J_diff: {}, Jprev; {}, Jcur: {}'
                #     .format(DeltaJ_rho, J_diff, J_prev, J_curr))
            else: #set an arbitrary value for cost change
                J_diff = -0.5

            # clear buffers when list is getting too big
            traj_samples[:-2] = [] if len(traj_samples) > 10 else traj_samples

            c = 0.5  # see DDP pg 113
            if J_diff > c and J_diff > c_zero: # accept the trajectory

                # ros com params
                start_time = Duration(secs = 0, nsecs = 0) # start asap
                duration = Duration(secs = duration_length, nsecs = 0) # apply effort continuously without end duration = -1
                reference_frame = None #'empty/world/map' #"youbot::base_footprint" #

                # rospy.loginfo("Found suitable trajectory. J_diff: %4f" %(J_diff))

                for t in range(T):
                    # send the computed torques to ROS
                    torques = new_sample_info.traj_info.delta_action[t,:]
                    # torques = new_sample_info.traj_info.action[t,:]

                    # calculate the genralized force and torques
                    bdyn = self.assemble_dynamics()
                    theta = torques[-1]  #bdyn.q[-1] #

                    sign_phi = np.sign(self.Phi_dot)

                    # see me notes
                    # F = (1/r) * B^T \tau - B^T S f
                    forces = (1/self.wheel['radius']) * bdyn.B.T.dot(torques) \
                                - bdyn.B.T.dot(bdyn.S.dot(bdyn.f))

                    # take care of directions in which we want our robot to move
                    # forces *= goal_sign

                    # get laser readings
                    self.pcl_rcvr.laser_listener()
                    # print('points: {}, pts len: {}'.format(self.pcl_rcvr.points, len(self.pcl_rcvr.points)))
                    #  reset points
                    self.pcl_rcvr.points = []
                    scale_factor = 5

                    wrench_base.force.x =  forces[0] * scale_factor
                    wrench_base.force.y =  forces[1] * scale_factor
                    wrench_base.torque.z = forces[2] * scale_factor

                    rospy.loginfo('Fx: %4f, Fy: %4f, Ftheta: %4f' % (wrench_base.force.x,
                            wrench_base.force.y, wrench_base.torque.z))

                    state_change = bdyn.q - self.goal_state
                    # rospy.loginfo("\ntorques: {}".format(torques))
                    rospy.loginfo("\nx:\t {}, \ndelx:\t {}, \nu:\t {}, \ndelu:\t {},\nq:\t {}"
                        ", \nq*:\t {}".format(
                        new_sample_info.traj_info.state[t,:],
                        new_sample_info.traj_info.delta_state[t,:], 
                        new_sample_info.traj_info.action[t,:],
                        new_sample_info.traj_info.delta_action[t,:], 
                        bdyn.q,
                        self.goal_state))
                    # send the torques to the base footprint
                    send_body_wrench('base_footprint', reference_frame,
                                                    None, wrench_base, start_time,
                                                    duration )

                    rospy.sleep(duration)

                    clear_active_wrenches('base_footprint')

                    gs = np.linalg.norm(self.goal_state)
                    cs = np.linalg.norm(bdyn.q)
                    if np.allclose(gs, cs, rtol=0, atol=stop_cond):
                    # if np.all(state_change < stop_cond):
                        rospy.loginfo("Reached goal state. Terminating.")
                        # eta = stop_cond
                        run = False
                        break

                    # rospy.loginfo('Eta: {}'.format(abs(gs-cs)))

                    if save:
                        with open(savefile, 'ab') as f:
                            np.savetxt(f, np.expand_dims(bdyn.q, 0))

                if save:
                    with open(costfile, 'ab') as fc:
                        np.savetxt(fc, new_sample_info.cost_info.l)
                # set ubar_i = u_i, xbar_i = x_i and repeat traj_opt # step 5 DDP book
                new_sample_info.traj_info.nom_action = new_sample_info.traj_info.action
                new_sample_info.traj_info.nom_state  = new_sample_info.traj_info.state

                if not run :
                    print("Finished Traj Opt")
                    break

                sample_info = new_sample_info

            else: # repeat back+forward pass if integration diverged from prev traj by so much
                # rospy.loginfo('Repeating traj opt phase: iteration not accepted')
                self.linesearch_param = self.linesearch_param - self.linesearch_param * 0.01
                sample_info = self.get_traj_cost_info(noisy)

            # rospy.loginfo('Finished Trajectory optimization process')

        toc = time.time()

        rospy.loginfo("Reaching the goal state took {} secs".format(toc-tic))

        with open(save_dir + '/time.txt', 'a') as f:
            f.write("time taken: %s" % str(toc-tic))

    def backward(self, sample_info, noisy=False):
        T  = self.T
        dU = self.dU
        dV = self.dV
        dX = self.dX

        cost_info = sample_info.cost_info
        traj_info = sample_info.traj_info
        # mu        = self.mu

        non_pos_def = True
        while (non_pos_def):

            non_pos_def = False

            for t in range (T-1, -1, -1):

                cost_info.Qnut[t]   = cost_info.l[t]
                cost_info.Qx[t,:]     = cost_info.lx[t]
                cost_info.Qu[t,:]     = cost_info.lu[t]
                cost_info.Quu[t,:,:]  = cost_info.luu[t]
                cost_info.Qxx[t,:,:]  = cost_info.lxx[t]
                cost_info.Qux[t,:,:]  = cost_info.lux[t]
                cost_info.Qv[t,:]     = cost_info.lv[t]
                cost_info.Qvx[t,:,:]  = cost_info.lvx[t]
                cost_info.Qvv[t,:,:]  = cost_info.lvv[t]
                cost_info.Quu_tilde[t,:,:]  = cost_info.luu[t]
                cost_info.Qvv_tilde[t,:,:]  = cost_info.lvv[t]

                # form Q derivatives at time t first
                if t < T-1:
                    # wil be --> (3) + (3x3) x (3) ==> (3)
                    cost_info.Qx[t,:] = cost_info.lx[t] + traj_info.fx[t,:].T.dot(cost_info.Vx[t+1,:])
                    # wil be --> (4) + (4,3) x (3) ==> (4)
                    cost_info.Qu[t,:] = cost_info.lu[t] + traj_info.fu[t,:].T.dot(cost_info.Vx[t+1,:])
                    # wil be --> (4) + (4,3) x (3) ==> (4)
                    cost_info.Qv[t,:] = cost_info.lv[t] + traj_info.fv[t,:].T.dot(cost_info.Vx[t+1,:])
                    # wil be --> (3) + (3,3) x (3,3) x ((3,3)) ==> (3,3)
                    cost_info.Qxx[t,:,:]  = cost_info.lxx[t] + traj_info.fx[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fx[t,:])
                    # wil be --> (4, 3) + (4,3) x (3,3) x ((3,3)) ==> (4,3)
                    cost_info.Qux[t,:,:]  = cost_info.lux[t] + traj_info.fu[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fx[t,:])
                    # wil be --> (4, 3) + (4,3) x (3,3) x ((3,3)) ==> (4,3)
                    cost_info.Qvx[t,:,:]  = cost_info.lvx[t] + traj_info.fv[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fx[t,:])
                    # wil be --> (4, 4) + (4,3) x (3,3) x ((3,4)) ==> (4,4)
                    cost_info.Quu[t,:,:]  = cost_info.luu[t] + traj_info.fu[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fu[t,:])
                    # wil be --> (4, 4) + (4,3) x (3,3) x ((3,4)) ==> (4,4)
                    cost_info.Qvv[t,:,:]  = cost_info.lvv[t] + traj_info.fv[t,:].T.dot(cost_info.Vxx[t+1,:,:]).dot(traj_info.fv[t,:])

                    # calculate the Qvals that penalize devs from the states
                    cost_info.Qu_tilde[t,:] = cost_info.lu[t,:] +  traj_info.fu[t,:].T.dot(cost_info.Vx[t+1,:]+self.mu * np.ones(dX))
                    cost_info.Qv_tilde[t,:] = cost_info.lv[t,:] +  traj_info.fv[t,:].T.dot(cost_info.Vx[t+1,:]+self.mu * np.ones(dX))
                    cost_info.Quu_tilde[t,:,:] = cost_info.luu[t] + traj_info.fu[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fu[t,:])
                    cost_info.Qvv_tilde[t,:,:] = cost_info.lvv[t] + traj_info.fv[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fv[t,:])
                    cost_info.Qux_tilde[t,:,:] = cost_info.lux[t] + traj_info.fu[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fx[t,:]) #+ \
                                    #traj_info.fux[t,:,:].dot(cost_info.Vx[t+1,:])
                    cost_info.Quv_tilde[t,:,:] = cost_info.luv[t] + traj_info.fu[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fv[t,:]) #+ \
                                    #traj_info.fux[t,:,:].dot(cost_info.Vx[t+1,:])
                    cost_info.Qvx_tilde[t,:,:] = cost_info.lvx[t] + traj_info.fv[t,:].T.dot(\
                                    cost_info.Vxx[t+1,:,:] + self.mu * np.eye(dX)).dot(traj_info.fx[t,:]) #+ \
                                    #traj_info.fux[t,:,:].dot(cost_info.Vx[t+1,:])

                # symmetrize the second order moments of Q
                cost_info.Quu[t,:,:] = 0.5 * (cost_info.Quu[t].T + cost_info.Quu[t])
                cost_info.Qvv[t,:,:] = 0.5 * (cost_info.Qvv[t].T + cost_info.Qvv[t])
                cost_info.Qxx[t,:,:] = 0.5 * (cost_info.Qxx[t].T + cost_info.Qxx[t])

                # symmetrize for improved Q state improvement values too
                cost_info.Quu_tilde[t,:,:] = 0.5 * (cost_info.Quu_tilde[t].T + cost_info.Quu_tilde[t])
                cost_info.Qvv_tilde[t,:,:] = 0.5 * (cost_info.Qvv_tilde[t].T + cost_info.Qvv_tilde[t])
                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(cost_info.Quu_tilde[t,:,:])
                    L = U.T
                    V = sp.linalg.cholesky(cost_info.Qvv_tilde[t,:,:])
                    Lv = V.T
                except LinAlgError as e:
                    rospy.loginfo('LinAlgError: %s', e)
                    non_pos_def = True
                    # restart the backward pass if Quu is non positive definite
                    break

                # compute open and closed loop gains.
                invQuu_tilde = sp.linalg.solve_triangular(
                            U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True) )
                invQvv_tilde = sp.linalg.solve_triangular(
                            V, sp.linalg.solve_triangular(Lv, np.eye(dV), lower=True) )
                Ku_tilde    = np.linalg.pinv(
                                    (np.ones_like(cost_info.Quu_tilde[t]) 
                                    - invQuu_tilde.dot(cost_info.Quv_tilde[t]).dot(\
                                      invQvv_tilde).dot(cost_info.Quv_tilde[t].T)).dot(invQuu_tilde[t])
                                    )
                Kv_tilde    = np.linalg.pinv(
                                    (np.ones_like(cost_info.Qvv_tilde[t]) 
                                    - invQvv_tilde.dot(cost_info.Quv_tilde[t].T).dot(\
                                      invQuu_tilde).dot(cost_info.Quv_tilde[t])).dot(invQvv_tilde[t])
                                    )               
                gu          = Ku_tilde.dot(cost_info.Quv_tilde[t].dot(invQvv_tilde).dot(cost_info.Qv_tilde[t]) \
                                 - cost_info.Qu_tilde[t]
                                 )
                gv          = Kv_tilde.dot(cost_info.Quv[t].T.dot(invQuu_tilde).dot(cost_info.Qu_tilde[t]) \
                                 - cost_info.Qv_tilde[t]
                                 )   
                Gu          = Ku_tilde.dot(cost_info.Quv[t].dot(invQvv_tilde).dot(cost_info.Qvx_tilde[t]) \
                                 - cost_info.Qux_tilde[t]
                                 )
                Gv          = Kv.dot(cost_info.Quv[t].T.dot(invQuu_tilde).dot(cost_info.Qux_tilde[t]) \
                                 - cost_info.Qvx_tilde[t]
                                 )                 

                # store away the gains
                traj_info.gu[t, :] = gu
                traj_info.gv[t, :] = gv
                traj_info.Gu[t, :, :] = Gu
                traj_info.Gv[t, :, :] = Gv

                # calculate improved value functions
                cost_info.V[t] = 0.5 *(gu.dot(cost_info.Quu_tilde[t]).dot(gu)  \
                                    +  gv.dot(cost_info.Qvv_tilde[t]).dot(gv)) \
                                    +  gu.dot(cost_info.Qu_tilde[t]) \
                                    +  gv.dot(cost_info.Qv_tilde[t]) \
                                    +  gu.dot(cost_info.Quv[t]).dot(gv) 

                cost_info.Vx[t,:] = cost_info.Qx[t,:] \
                                    + Gu.T.dot(cost_info.Qu_tilde[t])\
                                    + Gv.T.dot(cost_info.Qv_tilde[t])\
                                    + Gu.T.dot(cost_info.Quu_tilde[t]).dot(gu) \
                                    + Gv.T.dot(cost_info.Qvv_tilde[t]).dot(gv) \
                                    + gu.dot(cost_info.Qux_tilde[t]) \
                                    + gv.dot(cost_info.Qvx_tilde[t]) \
                                    + Gv.T.dot(cost_info.Quv_tilde[t].T).dot(gu) \
                                    + Gu.T.dot(cost_info.Quv_tilde[t].T).dot(gv)

                cost_info.Vxx[t,:,:] = 0.5*(cost_info.Qxx[t] \
                                    + Gu.T.dot(cost_info.Quu_tilde[t,:,:]).dot(Gu) \
                                    + Gv.T.dot(cost_info.Qvv_tilde[t,:,:]).dot(Gv)) \
                                    + Gu.T.dot(cost_info.Qux_tilde[t,:,:]) \
                                    + Gv.T.dot(cost_info.Qvx_tilde[t,:,:]) \
                                    + Gu.T.dot(cost_info.Quv_tilde[t,:,:]).dot(Gv) 

                # symmetrize quadratic Value hessian
                cost_info.Vxx[t,:,:] = 0.5 * (cost_info.Vxx[t,:,:] + cost_info.Vxx[t,:,:].T)

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
        dV = self.dV
        dX = self.dX

        # forward pass params
        rho = self.linesearch_param
        delta_t = T/(T-1) # euler step

        # get samples used in the backward pass
        traj_info = prev_sample_info.traj_info
        cost_info = prev_sample_info.cost_info

        bdyn = self.assemble_dynamics()
        # start fwd pass
        traj_info.state[0] = traj_info.nom_state[0]

        for t in range(1, T-1):
            traj_info.action[t,:]       = traj_info.action[t,:] \
                                                + rho * traj_info.gu[t, :] \
                                                + traj_info.Gu[t, :].dot(traj_info.state[t,:])

            # update state x(t+1) = f(x(t), u(t))
            traj_info.state[t+1,:]      = traj_info.state[t] \
                                            + delta_t * self.get_state_rhs_eq(bdyn, traj_info.state[t], traj_info.action[t])

            # set ubar_i = u_i, x_bar_i = x_i # pg 113 DDp Book
            traj_info.nom_action[t]   = traj_info.action[t]
            traj_info.nom_state[t]    = traj_info.state[t]


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
            stop_cond = hyperparams.config['agent']['TOL']
            trajopt.do_traj_opt(sample_info, stop_cond, save=True)


    except KeyboardInterrupt:
        LOGGER.critical("shutting down ros")
