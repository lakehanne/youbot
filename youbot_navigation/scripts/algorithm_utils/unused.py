
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
        