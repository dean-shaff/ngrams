from textprocessor import TextProcessor
import nltk
import codecs
import time 
import numpy as np 
import emcee
import h5py
import types
import multiprocessing
import cPickle as pickle
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

debug = 0

class BayesNetwork(TextProcessor):


	def __init__(self, *args,**kwargs):
		"""
		args:
			to be passed to TextProcessor
		kwargs:
			to be passed to TextProcessor
		"""
		TextProcessor.__init__(self, *args, **kwargs)

	def obs_prob(self, indices):
		"""
		calculate the observed probability
		P(A, B, C, D,...) = P(A)*P(B|A)*P(C|B,A)*P(D|C,B,A)*...
		args:
			- indices: The indices of the characters/words in self.unique_tokens 
				for which to calculate the observed probability. Only works with 
			- biglist: the list in which we're searching for indices.
		"""
		t0 = time.time()
		biglist = self.master_tokens_int
		n = len(indices)
		len_big = self.n_tokens
		if isinstance(indices[0],int):
			target_ints = indices
		elif isinstance(indices[0],types.StringTypes):
			target_ints = [self.unique_tokens.index(char) for char in indices]
		reduced = np.where(biglist == target_ints[0])[0]
		# if debug: print(reduced.shape[0])
		prob = float(reduced.shape[0]) / float(len_big)
		# if debug: print("Initial probabiltity: {}".format(prob))
		if prob == 0: 
			if debug: print("Time in calculation (zero prob, early exit): {:.3f}".format(time.time() - t0))
			return 0.0
		for i in xrange(1,n):
			target = target_ints[i]
			given = target_ints[:i]
			g_s = len(given)
			# if debug: print(target, given)
			total_occ = 0 # all occurences of given 
			target_occ = 0 # number of times given shows up and target appears
			# new_reduced = []
			
			for offset in xrange(g_s):
				# new_len = int(len_big / g_s) * g_s
				dim0 = int(len_big/g_s)-1
				reshape = biglist[offset: (dim0*g_s)+offset].reshape((dim0, g_s))
				given_loc = np.all(np.equal(reshape,given),axis=1)
				total_occ += np.sum(given_loc)
				given_loc = np.where(given_loc)[0]*g_s
				target_occ += np.sum(biglist[given_loc+offset+i] == target)

			if (target_occ == 0 or total_occ == 0):
				# if debug: print("Time in calculation (zero prob, late exit): {:.3f}".format(time.time() - t0))
				return 0.0
			prob *= (float(target_occ)/float(total_occ))
		if debug: print("Time in calculation: {:.3f}, prob: {:.7f}, pos: {} ".format(time.time() - t0, prob, indices))
		return prob

	def obs_prob_old(self, indices):
		"""
		calculate the observed probability
		P(A, B, C, D,...) = P(A)*P(B|A)*P(C|B,A)*P(D|C,B,A)*...
		args:
			- indices: The indices of the characters/words in self.unique_tokens 
				for which to calculate the observed probability. Only works with 
			- biglist: the list in which we're searching for indices.
		"""
		t0 = time.time()
		biglist = self.master_tokens_int
		n = len(indices)
		len_big = self.n_tokens
		if isinstance(indices[0],int):
			target_ints = indices
		elif isinstance(indices[0],types.StringTypes):
			target_ints = [self.unique_tokens.index(char) for char in indices]
		reduced = np.where(biglist == target_ints[0])[0]
		if debug: print(reduced.shape[0])
		prob = float(reduced.shape[0]) / float(len_big)
		if debug: print("Initial probabiltity: {}".format(prob))
		if prob == 0: 
			if debug: print("Time in calculation (zero prob, early exit): {:.3f}".format(time.time() - t0))
			return 0.0
		for i in xrange(1,n):
			target = target_ints[i]
			given = target_ints[:i]
			# if debug: print(target, given)
			total_occ = 0 # all occurences of given 
			target_occ = 0 # number of times given shows up and target appears
			new_reduced = []
			for j in reduced:
				if np.all(biglist[j:j+i] == given):
					total_occ += 1 
					new_reduced.append(j)
					if j+i == len_big:
						continue
					if biglist[j+i] == target:
						target_occ += 1
			reduced = new_reduced
			if (target_occ == 0 or total_occ == 0):
				if debug: print("Time in calculation (zero prob, late exit): {:.3f}".format(time.time() - t0))
				return 0.0
			prob *= (float(target_occ)/float(total_occ))
		if debug: print("Time in calculation: {:.3f}".format(time.time() - t0))
		return prob

	def generate_single_prob_distr(self, seed):
		"""
		generate a single probability distribution given a seed.
		args:
			- seed: the source text to use for generating phrase.
		"""
		if isinstance(seed, str):
			seed = list(seed)
		probs = []
		for char in self.unique_tokens:
			t0 = time.time()
			seed_temp = seed[:] 
			seed_temp.append(char)
			# print(seed_temp)
			prob_temp = self.obs_prob(seed_temp)
			# print(seed_temp, prob_temp)
			probs.append(prob_temp)

		return probs

	def generate_phrase(self, seed, n, maxlen=5):
		"""
		Generate a phrase given some seed. Will iterate n times. 
		(Final length will be len(seed) + n)
		args:
			- seed: The original seed. 
			- n: The number of times to add to seed.
		"""
		if isinstance(seed, str):
			seed = list(seed)
		for i in xrange(n):
			if len(seed) < maxlen:
				p = np.array(self.generate_single_prob_distr(seed))
			elif len(seed) >= maxlen:
				p = np.array(self.generate_single_prob_distr(seed[-maxlen:]))
			p = p / np.sum(p)
			print(p)
			n_i = np.random.choice(self.n_unique_tokens,1, p=p)[0] # next index
			seed.append(self.unique_tokens[n_i])
			print("".join(seed))
		seed_str = "".join(seed)
		return seed, seed_str

	def map_domain(self, pos_int):
		"""
		Map some integer values in the range 0,self.n_unique_tokens to the range (0,1)
		args:
			- pos_int: A tuple or list of either chars or ints. Or it could just be a plain string! 
		"""
		m = interp1d([0,self.n_unique_tokens],[0.,1.])
		if isinstance(pos_int, types.StringTypes):
			seed = list(seed)
			seed = [self.unique_tokens.index(char) for char in pos_int]
		elif isinstance(pos_int[0], types.StringTypes):
			seed = [self.unique_tokens.index(char) for char in pos_int]
		elif isinstance(pos_int[0], int):
			seed = pos_int		

		mapped = m(seed)

		return mapped	

	def __call__(self,pos):
		"""
		This function is for emcee. It takes a position vector, whose elements are between 0 and 1,
		maps it to indices in the bn.unique_tokens integer range, and calculates a probablity.
		It returns the log of the probability from obs_prob.
		args:
			-pos: a numpy array whose element are between 0 and 1
		"""
		cond_above = pos >= self.n_unique_tokens 
		cond_below = pos < 0
		if np.any(cond_above) or np.any(cond_below):
			return -np.inf 
		else:
			# pos = (self.n_unique_tokens*pos).astype(int)
			pos = pos.astype(int)
			prob = self.obs_prob(pos)
			if prob == 0.0:
				return -np.inf
			else:
				return prob

def run_emcee(nwalkers, ndim, bn):
	"""
	run markov chain monte carlo using dfm emcee. 
	args:
		- nwalkers: The number of walkers 
		- ndim: The number of dimensions. In the case of Bayes Network, this corresponds to the 
			number of character/words we're trying to predict. 
		- log_prob: The callback function to the log of the probability distribution
	"""
	p0 = np.random.randint(0,bn.n_unique_tokens,(nwalkers,ndim)) 
	sampler = emcee.EnsembleSampler(nwalkers,ndim,bn,threads=1)
	t0 = time.time()
	iterations = 100
	for i, result in enumerate(sampler.sample(p0, iterations=iterations)):
		end_pos = result[0]
		if i % 10 == 0:
			t1 = time.time()
			print("{} iterations left. This iteration took {:.2f} seconds".format(iterations - i, t1 - t0))
			t0 = t1
	t0 = time.time()
	print("Starting the real run!")	
	sampler.reset()
	iterations = 400
	for i, result in enumerate(sampler.sample(end_pos, iterations=iterations)):
		end_pos = result[0]
		if i % 10 == 0:
			t1 = time.time()
			print("{} iterations left. This iteration took {:.2f} seconds".format(iterations - i, t1 - t0))
			t0 = t1

	return sampler 

if __name__ == '__main__':
	bn = BayesNetwork(["texts/AustenPride.txt",
                    "texts/DickensTaleofTwo.txt",
                    "texts/china.txt","texts/fairy.txt",
                    "texts/harperlee.txt","texts/conrad.txt",
                    "texts/christie.txt","texts/kafka.txt","texts/kant.txt"],'char',
                    remove_tokens=[u'\xea', u'\xae', u'\xe9', u'&', u'+',
                     u'*', u'\xaf', u'1', u'0', u'3', u'2', u'5', u'4', u'7', u'6',
                      u'9', u'8', u'\xe0',])
    
	# bn = BayesNetwork("texts/AustenPride.txt",'char')
	# print(bn.obs_prob(list('the')))
	# print(bn.obs_prob_old(list('the')))
	# dat = pickle.dumps(bn)
	t0 = time.time()
	sampler = run_emcee(200, 5, bn)
	print("Time running {:.2f}".format(time.time() - t0))
	f = h5py.File("distr.hdf5",'w')
	f.create_dataset('flatchain',data=sampler.flatchain)
	f.create_dataset('probs',data=sampler.flatlnprobability)
	f.close()
	# # for i in xrange(3):
	# # 	plt.figure()
	# 	distr = sampler.flatchain[:,i]
	# 	probs = sampler.flatlnprobability
	# 	cond1 = distr >= 0
	# 	cond2 = distr < 1
	# 	cond_tot = np.logical_and(cond1, cond2)
	# 	print(cond_tot.shape)
	# 	distr_cut = distr[np.all(cond_tot, axis=1)]
	# 	plt.hist(distr_cut,40, color='k')
	# 	plt.title("Dimension {}".format(i))

	# plt.show()

	# print(bn.lnprob(np.array([1.0,0.0,0.1,0.1])))

	# t0 = time.time()
	# print(bn.n_unique_tokens)
	# vals = np.zeros((bn.n_unique_tokens,bn.n_unique_tokens))
	# for l in np.linspace(0.0,1.0,bn.n_unique_tokens):
	# 	for h in np.linspace(0.0,1.0,bn.n_unique_tokens):
	# 		print(bn.lnprob(np.array([l,h])))
	# 		vals[l,h] = bn.obs_prob([l,h])
	# 		# print(bn.obs_prob(['t','h','e','n',' ','h']))
	# 	print("Row {} done!".format(l))
	# f = h5py.File('dat.hdf5','w')
	# f.create_dataset('probs',data=vals)
	# f.close()
	# print(time.time() - t0)
