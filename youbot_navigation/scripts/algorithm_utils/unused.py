    def get_traj_cost_info(self, noisy=False):

        T, dU, dX = self.T, self.dU, self.dX

        N = self.config['agent']['sample_length']

        # allocate 
        fx      = np.zeros((N, T, dX, dX))
        fu      = np.zeros((N, T, dX, dU))
        fuu     = np.zeros((T, dU, dU))
        fux     = np.zeros((T, dU, dX))

        u       = np.zeros((N, T, dU))
        x       = np.zeros((N, T, dX))

        # change in state-action pair
        delta_u = np.zeros_like(u)
        delta_x = np.zeros_like(x)

        # nominal state-action pair
        u_bar   = np.zeros_like(u)
        x_bar   = np.zeros_like(x)

        # costs
        l        = np.zeros((N, T))
        l_nom    = np.zeros((N, T))
        lu       = np.zeros((N, T, dU))
        lx       = np.zeros((N, T, dX))
        lxx      = np.zeros((N, T, dX, dX))
        luu      = np.zeros((N, T, dU, dU))
        lux      = np.zeros((N, T, dU, dX))

        # get real samples from the robot
        for n in range(N):

            sample = self.get_samples(noisy)

            traj_info = sample['traj_info']

            # get the derivatives of f
            fx[n]         = traj_info.fx
            fu[n]         = traj_info.fu

            u[n]          = traj_info.action
            x[n]          = traj_info.state

            delta_u[n]    = traj_info.delta_action
            delta_x[n]    = traj_info.delta_state
            x_bar[n]      = traj_info.nominal_state
            u_bar[n]      = traj_info.nominal_action

            cost_info = sample['cost_info']
            # now calculate the cost-to-go around each sample
            
            l[n]    = cost_info.l
            l_nom[n]= cost_info.l_nom
            lx[n]   = cost_info.lx
            lu[n]   = cost_info.lu                
            lxx[n]  = cost_info.lxx
            luu[n]  = cost_info.luu
            lux[n]  = cost_info.lux

        cost_info = CostInfo(self.config)
        traj_info = TrajectoryInfo(self.config)

        # fill in the cost estimate
        cost_info.l     = np.mean(l, 0)
        cost_info.l_nom = np.mean(l_nom, 0)
        cost_info.lu    = np.mean(lu, 0)
        cost_info.lx    = np.mean(lx, 0)
        cost_info.lxx   = np.mean(lxx, 0)
        cost_info.luu   = np.mean(luu, 0)
        cost_info.lux   = np.mean(lux, 0)

        # store away stage trajectories
        traj_info.fx            = np.mean(fx,    axis=0)
        traj_info.fu            = np.mean(fu,    axis=0)
        traj_info.action        = np.mean(u,     axis=0)
        traj_info.state         = np.mean(x,     axis=0)
        traj_info.delta_state   = np.mean(delta_x, axis=0)
        traj_info.delta_action  = np.mean(delta_u, axis=0)
        traj_info.nominal_state = np.mean(x_bar,   axis=0)

        

        # joint names of the four wheels
        wheel_joint_bl = 'wheel_joint_bl'
        wheel_joint_br = 'wheel_joint_br'
        wheel_joint_fl = 'wheel_joint_fl'
        wheel_joint_fr = 'wheel_joint_fr'
             # send in this order: 'wheel_joint_bl', 'wheel_joint_br', 'wheel_joint_fl', 'wheel_joint_fr'
    def get_new_state(self, theta, t):
        body_dynamics = self.assemble_dynamics()

        # time varying inverse dynamics parameters
        M     = body_dynamics.M
        C     = body_dynamics.C
        B     = body_dynamics.B
        S     = body_dynamics.S
        f     = body_dynamics.f
        qvel  = body_dynamics.qvel
        qaccel= body_dynamics.qaccel

        # update time-varying parameters of mass matrix
        d1, d2  =  1e-2, 1e-2

        mw, mb, r  = self.wheel['mass'], self.base['mass'], self.wheel['radius']
        I, I_b     = self.wheel['mass_inertia'][1,1], self.base['mass_inertia'][-1,-1]
         
        base_footprint_dim = 0.001  # retrieved from the box geom in gazebo
        l = np.sqrt(2* base_footprint_dim)
        l_sqr = 2* base_footprint_dim

        b, a = 0.19, 0.145 # meters as measured from the real robot
        alpha = np.arctan2(b, a)

        m13 = mb * ( d1 * np.sin(theta) + d2 * np.cos(theta) )
        m23 = mb * (-d1 * np.cos(theta) + d2 * np.sin(theta) )
        m33 = mb * (d1 ** 2 + d2 ** 2) + I_b + \
                    8 * (mw + I/(r**2)) * l_sqr * pow(np.sin(np.pi/4.0 - alpha), 2)

        # update mass_matrix
        M[0,2], M[2,0], M[1,2], M[2,1] = m13, m13, m23, m23

        # update B matrix
        B[:,:2].fill(np.cos(theta) + np.sin(theta))
        B[:,-1].fill(-np.sqrt(2)*l*np.sin(np.pi/4 - alpha))
        B[0,0] = np.sin(theta) - np.cos(theta)
        B[1,0] *= -1
        B[2,0] = np.cos(theta) - np.sin(theta)
        B[0,1]          = -1.0 * B[0,1]
        B[1,1], B[3,1]  = B[2,0], B[0,0]

        # calculate phi dor from eq 6
        Phi_coeff = -(np.sqrt(2)/r) 
        # mid matrix
        Phi_left_mat = np.ones((4, 3))
        Phi_left_mat[:,:2].fill(np.sqrt(2)/2)
        Phi_left_mat[:,2].fill(l*np.sin(np.pi/4 - alpha))
        # column 0
        Phi_left_mat[2, 0] *= -1
        Phi_left_mat[3, 0] *= -1
        # column 1
        Phi_left_mat[1, 1] *= -1
        Phi_left_mat[2, 1] *= -1

        Phi_right_mat = np.zeros((3,3))
        Phi_right_mat[0,0] = np.cos(theta)
        Phi_right_mat[1,1] = np.cos(theta)

        Phi_right_mat[0,1] = np.sin(theta)
        Phi_right_mat[1,0] = -np.sin(theta)

        xdot = self.odom.twist.twist.linear.x
        ydot = self.odom.twist.twist.linear.y
        theta_dot = self.odom.twist.twist.angular.z

        Phi_right_vector   = np.asarray([xdot, ydot, theta_dot]) 
        # assemble Phi vector  --> will be 4 x 1
        Phi_dot = Phi_coeff * Phi_left_mat.dot(Phi_right_mat).dot(Phi_right_vector)
        S = np.diag(np.sign(Phi_dot).squeeze())

        # calculate inverse dynamics equation
        BBT             = B.dot(B.T)
        Inv_BBT         = np.linalg.inv(BBT)
        multiplier      = Inv_BBT.dot(self.wheel_rad * B)
        inv_dyn_eq      = M.dot(qaccel) + C.dot(qvel) + \
                                B.T.dot(S).dot(f)

        mass_inv            = -np.linalg.inv(M)
        for n in range(1, self.euler_iter):
            self.traj_info.nominal_state_[n,:] = self.traj_info.nominal_state_[n-1,:] + \
                            self.euler_step * (mass_inv.dot(C).dot(self.traj_info.nominal_state[t,:])) - \
                            mass_inv.dot(B.T).dot(S).dot(f)+ \
                            (mass_inv.dot(B.T).dot(self.traj_info.nominal_action[t,:]))/self.wheel_rad
        new_state = self.traj_info.nominal_state_[n,:] # decode nominal state at last euler step


        return new_state


    def eval_cost(sample):
        u       =  sample.control_seq
        x       =  sample.state_seq

        delta_x =  sample.control_delta
        delta_u =  sample.state_delta

        x_bar   =  sample.state_nom
        u_bar   =  sample.control_nom

        delta_x_star = self.goal_state

        # calculate cost-to-go and derivatives of stage_cost
        cost_action_term = 0.5 * np.sum(self.action_penalty[0] * \
                                    (np.linalg.norm(delta_u, axis=1) ** 2),
                                  axis = 0)
        cost_state_term  = 0.5 * np.sum(self.state_penalty[0] * \
                                    (np.linalg.norm(delta_x, axis=1) ** 2),
                                  axis = 0)
        cost_l12_term    = np.sqrt(self.l21_const + (delta_x[-1,:] - delta_x_star)**2)
        # cost_l12_term    = np.sqrt(self.l21_const + (delta_x - delta_x_star)**2)


        # nominal cost terms
        cost_nom_action_term = 0.5 * np.sum(self.action_penalty[0] * \
                                    (np.linalg.norm(u_bar, axis=1) ** 2),
                                    axis = 0)
        cost_nom_state_term  = 0.5 * np.sum(self.state_penalty[0] * \
                                    (np.linalg.norm(x_bar, axis=1) ** 2),
                                    axis = 0)        
        cost_nom_l12_term   = np.sqrt(self.l21_const + (x_bar[-1,:] - delta_x_star)**2) 
        # cost_nom_l12_term    = np.sqrt(self.l21_const + (x_bar - delta_x_star)**2)        

        # define lx/lxx cost costants        
        final_state_diff = (x[-1,:] - self.goal_state)
        sqrt_term        = np.sqrt(self.l21_const + (final_state_diff**2))

        #system cost
        l       = cost_action_term + cost_state_term + cost_l12_term        
        # nominal cost about linear traj
        l_nom   = cost_nom_action_term + cost_nom_state_term + cost_nom_l12_term

        # first order cost terms
        lu = np.sum(self.action_penalty[0] * delta_u, axis=0) 
        lx = np.sum(self.state_penalty[0] * delta_x, axis=0)  \
                + (final_state_diff/np.sqrt(self.l21_const + final_state_diff**2)) 

        # 2nd order cost terms
        luu     = np.diag(self.action_penalty)        
        lux     = np.zeros((self.dU, self.dX))

        lxx_t1  = np.diag(self.state_penalty)
        lxx_t2_top = sqrt_term - (final_state_diff**2) /sqrt_term
        lxx_t2_bot = self.l21_const + (final_state_diff**2)

        lxx        = lxx_t1 + lxx_t2_top/lxx_t2_bot 

        # squeeze dims of first order derivatives
        lx = lx.squeeze() if lx.ndim > 1 else lx            
        lu = lu.squeeze() if lu.ndim > 1 else lu 

        return l, lx, lu, lxx, luu, lux 


# method for calculatingt ol controls
        """
        # open loop u matrices
        G  = np.zeros((dX*T, dX))
        H  = np.eye(dX*T)

        # Calculate open loop controls
        # this is taken from Tyler Summer's DP class code
        for i in range(1, T):
            G[range((i-1)*dX,dX*i),:] = fx[(i-1),:,:]**i
            for j in range(1, T):
                if i > j:
                    H[slice((i-1)*dX, dX*i), slice((j-1)*dX, dX*j)] = fx[(i-1),:,:]**(i-j)

        BB = np.kron(np.eye(T), np.mean(fu, 0))
        HH = H.dot(BB)

        ustar = np.linalg.lstsq(-(np.eye(T*dU) + HH.T.dot(HH)), HH.T.dot(G).dot(x[0,:]))   
        u_bar = ustar[0]#.squeeze()
        u_bar = u_bar.reshape(T, dU)
        """

        # rospy.logdebug("Integrating inverse dynamics equation")
        # # euler forward integration
        # x_bar = np.zeros((T, dX))
        # mass_inv = -np.linalg.inv(mass_matrix)
        # for n in range(1, self.euler_iter):
        #     x_bar[n] = x_bar[n] + self.euler_step * (mass_inv.dot(coriolis_matrix).dot(x_bar)) - \
        #                     mass_inv.dot(B_matrix.T).dot(S_matrix).dot(friction_vector)+ \
        #                     (mass_inv.dot(B_matrix.T).dot(u_bar[n,:]))/self.wheel_rad # check this
        

       # the stage costs are deterministic. do not use noisy
        if noisy:
            noise_covar[t] = (self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:])).T.dot(\
                                self.traj_distr.delta_state[t,:] - np.mean(self.traj_distr.delta_state[t,:]))


        # cost_action_term = self.action_penalty[0] * np.expand_dims(delta_u, axis=0).dot(\
        #                     np.expand_dims(delta_u, axis=1))
        # cost_state_term  = 0.5 * self.state_penalty[0] * np.expand_dims(delta_x, axis=0\
        #                     ).T.dot(np.expand_dims(delta_x, axis=1)))
        # cost_l12_term    = np.sqrt(self.l21_const + state_diff.T.dot(state_diff))

        # nominal action
        # cost_nom_action_term = self.action_penalty[0] * np.expand_dims(self.traj_distr.action_nominal[t,:], axis=0).dot(\
        #                                                 np.expand_dims(self.traj_distr.action_nominal[t,:], axis=1))
        # cost_nom_state_term  = 0.5 * self.state_penalty[0] * state_diff_nom.T.dot(state_diff_nom)
        # cost_nom_l12_term    = np.sqrt(self.l21_const + state_diff_nom.T.dot(state_diff_nom))

        # # first order derivatives
        # lu      = self.action_penalty * self.traj_distr.delta_action[t,:]
        # lx      = self.state_penalty[0] * state_diff + state_diff/np.sqrt(self.alpha + state_diff.T.dot(state_diff))                                   
        # luu     = np.diag(self.action_penalty)

        # lxx_t1  = np.diag(self.state_penalty)
        # lxx_t2  = np.eye(self.dX)
        # lxx_t3  = (state_diff.T.dot(state_diff)) / ((self.alpha + state_diff.T.dot(state_diff))**3)
        # lxx     = lxx_t1 +  lxx_t2 * np.eye(self.dX) - lxx_t3 * np.eye(self.dX)
        # lux     = np.zeros((self.dU, self.dX))

                    # create four different asynchronous threads for each wheel
            """
            wheel_joint_bl_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_bl_thread', 
                                        args=(wheel_joint_bl, torques[0], start_time, duration))
            wheel_joint_br_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_br_thread', 
                                        args=(wheel_joint_br, torques[1], start_time, duration))
            wheel_joint_fl_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_fl_thread', 
                                        args=(wheel_joint_fl, torques[2], start_time, duration))
            wheel_joint_fr_thread = threading.Thread(group=None, target=send_joint_torques, 
                                        name='wheel_joint_fr_thread', 
                                        args=(wheel_joint_fr, torques[3], start_time, duration))
            
            """
            """
            wrench_bl, wrench_br, wrench_fl, wrench_fr = Wrench(), Wrench(), Wrench(), Wrench()

            wrench_bl.force.x = torques[0]
            wrench_bl.force.y = torques[0]

            wrench_br.force.x = torques[1]
            wrench_br.force.y = torques[1]

            wrench_fl.force.x = torques[2]
            wrench_fl.force.y = torques[2]

            wrench_fr.force.x = torques[3]
            wrench_fr.force.y = torques[3]

            resp_bl = send_body_wrench('wheel_link_bl', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_br = send_body_wrench('wheel_link_br', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_fl = send_body_wrench('wheel_link_fl', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            resp_fr = send_body_wrench('wheel_link_fr', reference_frame, 
                                            None, wrench_bl, start_time, 
                                            duration )
            rospy.sleep(duration)

            # clear active wrenches
            clear_bl = clear_active_wrenches('wheel_link_bl')
            clear_br = clear_active_wrenches('wheel_link_bl')
            clear_fl = clear_active_wrenches('wheel_link_bl')
            clear_fr = clear_active_wrenches('wheel_link_bl')
            """
            # if args.silent:
            #     print('\n')