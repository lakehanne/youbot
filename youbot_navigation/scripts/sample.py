

class Sample(object):
	"""docstring for Sample"""
	def __init__(self, hyperparams):
		super(Sample, self).__init__()
		self.agent = hyperparams['agent']

        self.T = self.agent['T']
        self.dU = self.agent['dU']
        self.dV = self.agent['dV']
        self.dX = self.agent['dX']
        self.dO = self.agent['dO']

        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)

	    self._X = np.empty((self.T, self.dX))
	    self._X.fill(np.nan)

    def set(self, t=None):
        """ Set trajectory data for a particular sensor. """
        if t is None:
            self._X.fill(np.nan)  # Invalidate existing X.
            self._obs.fill(np.nan)  # Invalidate existing obs.
        else:
            self._X[t, :].fill(np.nan)
            self._obs[t, :].fill(np.nan)

	def get_X(self, t=None):
        X = self._X if t is None else self._X[t, :]

        return X

    def get_obs(self, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._obs if t is None else self._obs[t, :]
        
        return obs    

		
def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to be used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        # new_sample_adv = copy.deepcopy(new_sample)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        V = np.zeros([self.T, self.dV])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t) #see sample.py
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act_u(X_t, obs_t, t, noise[t, :])
            mj_V = policy.act_v(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            V[t, :] = mj_V
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)  # run the passive dynamics
                    # mj_X, _ = self._world[condition].step(mj_X, mj_V)  # run the passive dynamics
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        new_sample.set(ACTION_V, V)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
