#ngrams.py
from textprocessor import TextProcessor
import nltk
import codecs
import time 
import numpy as np 

class NGram(TextProcessor):
	"""
	A class for creating ngrams 
	"""
	def __init__(self, *args,**kwargs):
		"""
		args:
			to be passed to TextProcessor
		kwargs:
			to be passed to TextProcessor
		"""
		TextProcessor.__init__(self, *args, **kwargs)


    def find_sublist(self,sublist,biglist,**kwargs):
        """
        Find occurances of sublist in biglist 
        Returns:
            dict: 
                - 'unique_matches': The unique characters or words that follow sublist
                - 'freq': The frequency of each of these unique matches.
        """
        n = len(sublist)
        max_length = kwargs.get('max_length',n)
        if (max_length > n):
            max_length = n
        indices = []
        matches = []
        # print(sublist[n-max_length:])
        for i in xrange(len(biglist) - n + max_length):
            if biglist[i+n-max_length:i+n] == sublist[n-max_length:]:
                indices.append(i)
                matches.append(biglist[i+n])
            else:
                continue
        unique_matches = list(set(matches))
        n_matches = len(matches)
        freq = [float(matches.count(unique))/n_matches for unique in unique_matches]
        if list(freq) == []:
            freq = np.array([0.0])
        return {
            'n_matches': n_matches,
            'unique_matches':unique_matches,
            'freq':freq
        }

    def seed_sample_ngram(self,seed,n,mxlen):
        """
        Seed and sample from the ngram.
        args:
            seed: The word seed for the ngram 
            n: The word length of the resulting phrase.
            mxlen: The number of characters to use for predicting probablities.
        """
        while seed not in self.unique_tokens:
            print("Seed is not in the text. Try a different word or char.")
            seed = str(raw_input("New seed: "))
        seed = [seed]
        orig_seed = seed
        backup = -1 
        mxlen_orig = mxlen
        while len(seed) < n:
            sublist_dict = self.find_sublist(seed,self.master_tokens,max_length=mxlen)
            unique_matches = sublist_dict['unique_matches']
            distr = sublist_dict['freq']
            while distr[0] == 1.0 or distr[0] == 0.0:
                rando = np.random.randint(1,3)
                seed = seed[:-rando]
                if seed == []:
                    seed = [orig_seed]
                mxlen -= 1 
                sublist_dict = self.find_sublist(seed,self.master_tokens,max_length=mxlen)
                unique_matches = sublist_dict['unique_matches']
                distr = sublist_dict['freq']
                # if debug: print(mxlen)
                # if debug: print("".join(seed))
                # if debug: print(np.amax(distr))
                if distr[0] == 0.0:
                    continue
                next_index = np.random.choice(len(unique_matches),1,p=distr)[0]
                seed.append(unique_matches[next_index])
                if mxlen <= 2:
                    mxlen = mxlen_orig
                    break
            mxlen = mxlen_orig
            next_index = np.random.choice(len(unique_matches),1,p=distr)[0]
            seed.append(unique_matches[next_index])

        if self.mode == 'word':
            return " ".join(seed)
        elif self.mode == 'char':
            return "".join(seed)






if __name__ == "__main__":
    ngrammer = NGram(["texts/AustenPride.txt",
                    "texts/DickensTaleofTwo.txt",
                    "texts/china.txt","texts/fairy.txt",
                    "texts/harperlee.txt"], "char")#,remove_tokens=[u"``",u"''",u"--"])
    print(ngrammer.seed_sample_ngram("t",50,20))