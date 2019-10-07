import numpy as np

class Sampler:
	def __init__(self, phi_num):
		self.phi_num = phi_num
		self.phi_A = np.zeros((0,self.phi_num))
		self.phi_B = np.zeros((0,self.phi_num))
		self.a = [] # responses (-1 for A, 1 for B, 0 for IDK)
	
	def feed(self, phi_A, phi_B, a):
		phi_A = np.array(phi_A).reshape(-1, self.phi_num)
		phi_B = np.array(phi_B).reshape(-1, self.phi_num)
		
		self.phi_A = np.vstack((self.phi_A, phi_A))
		self.phi_B = np.vstack((self.phi_B, phi_B))
		for ax in a:
			self.a.append(ax)
		
	def logp(self, i, w, delta=0):
		phi_A = self.phi_A[i]
		phi_B = self.phi_B[i]
		a = self.a[i]
		
		if a != 0:
			phi_select = phi_A if a < 0 else phi_B
			phi_nonsel = phi_B if a < 0 else phi_A
			psi = phi_nonsel - phi_select
			return np.log(1 / (1 + np.exp(delta + psi.dot(w))))
		else:
			psi = phi_A - phi_B
			return np.log((np.exp(2*delta)-1) / (1 + np.exp(delta + psi.dot(w)) + np.exp(delta - psi.dot(w)) + np.exp(2*delta)))
		
	def logprob(self, w, delta=0):
		if np.sum(w**2) > 1 or delta < 0:
			return -np.inf
		return np.sum([self.logp(i, w, delta) for i in range(len(self.a))])
		
	def sample(self, sample_count, query_type, burn=1000, thin=50, step_size=0.1):
		if query_type == 'weak':
			x = np.array([0]*self.phi_num + [1]).reshape(1,-1)
			old_logprob = self.logprob(x[0,:self.phi_num], x[0,-1])
			for _ in range(burn + thin*sample_count):
				new_x = x[-1] + np.random.randn(self.phi_num + 1) * step_size
				new_logprob = self.logprob(new_x[:self.phi_num], new_x[-1])
				if np.log(np.random.rand()) < new_logprob - old_logprob:
					x = np.vstack((x,new_x))
					old_logprob = new_logprob
				else:
					x = np.vstack((x,x[-1]))
			x = x[burn+thin-1::thin]
			return x[:,:self.phi_num], x[:,-1]
		elif query_type == 'strict':
			x = np.array([0]*self.phi_num).reshape(1,-1)
			old_logprob = self.logprob(x[0], 0)
			for _ in range(burn + thin*sample_count):
				new_x = x[-1] + np.random.randn(self.phi_num) * step_size
				new_logprob = self.logprob(new_x, 0)
				if np.log(np.random.rand()) < new_logprob - old_logprob:
					x = np.vstack((x,new_x))
					old_logprob = new_logprob
				else:
					x = np.vstack((x,x[-1]))
			x = x[burn+thin-1::thin]
			return x, np.zeros((sample_count,))
		else:
			print('There is no query type called ' + query_type)
			exit(0)

	def sample_given_delta(self, sample_count, query_type, delta, burn=1000, thin=50, step_size=0.1):
		assert query_type in ['strict','weak'], 'There is no query type called ' + query_type
		if query_type == 'strict':
			delta = 0
		assert delta >= 0

		x = np.array([0]*self.phi_num).reshape(1,-1)
		old_logprob = self.logprob(x[0], delta)
		for _ in range(burn + thin*sample_count):
			new_x = x[-1] + np.random.randn(self.phi_num) * step_size
			new_logprob = self.logprob(new_x, delta)
			if np.log(np.random.rand()) < new_logprob - old_logprob:
				x = np.vstack((x,new_x))
				old_logprob = new_logprob
			else:
				x = np.vstack((x,x[-1]))
		x = x[burn+thin-1::thin]
		return x, delta * np.ones((x.shape[0],))

