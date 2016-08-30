from textprocessor import TextProcessor
import nltk
import codecs
import time 
import numpy as np 

debug = 1

class BayesNetwork(TextProcessor):

	def __init__(self, *args,**kwargs):
		"""
		args:
			to be passed to TextProcessor
		kwargs:
			to be passed to TextProcessor
		"""
		TextProcessor.__init__(self, *args, **kwargs)

	def obs_prob(self, indices,biglist):
		"""
		calculate the observed probability
		P(A, B, C, D,...) = P(A)*P(B|A)*P(C|B,A)*P(D|C,B,A)*...
		args:
			- indices: The indices of the characters/words in self.unique_tokens 
				for which to calculate the observed probability. 
			- biglist: the list in which we're searching for indices.
		"""
		n = len(indices)
		len_big = len(biglist)
		if isinstance(indices[0],int):
			target_chars = [self.unique_tokens[i] for i in indices]
		elif isinstance(indices[0], str):
			target_chars = indices
		# if debug: print(target_chars)
		t0 = time.time()
		reduced_biglist_indices = [i for i in xrange(len_big) if biglist[i] == target_chars[0]]
		prob = float(len(reduced_biglist_indices)) / float(len_big)
		if prob == 0: return 0.0
		for i in xrange(1,n):
			target = target_chars[i]
			given = target_chars[:i]
			if debug: print(target, given)
			total_occ = 0 # all occurences of given 
			target_occ = 0 # number of times given shows up and target appears
			new_reduced = []
			for j in reduced_biglist_indices:
				if biglist[j:j+i] == given:
					total_occ += 1 
					new_reduced.append(j)
					if biglist[j+i] == target:
						target_occ += 1
			# reduced_biglist_indices = new_reduced
			# print(total_occ, target_occ)
			if (target_occ == 0 or total_occ == 0):
				return 0.0
			prob *= (float(target_occ)/float(total_occ))
		if debug: print("Time in calculation: {:.3f}".format(time.time() - t0))
		return prob



if __name__ == '__main__':
	bn = BayesNetwork('texts/AustenPride.txt','char')
	print(bn.obs_prob(['t','h','e','n',' ','h','e'],bn.master_tokens))