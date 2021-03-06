#ngram.py
import nltk
import codecs
import time 
import numpy as np 

debug = 1

class NGram(object):
    """
    A class for creating Ngrams.  
    """ 
    def __init__(self, filenames, mode,**kwargs):
        """
        Initialize the NGram object. 
        args:
            - filenames: The files containining source text.
            - mode: Word or character based? 
        kwargs:
            - remove_tokens: Tokens to be removed.
        """
        self.remove_tokens = kwargs.get('remove_tokens',[])
        self.master_str = ""
        self.master_tokens = []
        if mode == 'word':
            self.add_files_word(filenames)
        elif mode == 'char':
            self.add_files_char(filenames)
        self.mode = mode

    def add_files_word(self,filenames):
        """
        Add the contents of a file to the master source string for word based approach. 
        args:
            - filenames: The name of the files to be added.
        """
        master_str = ""
        t0 = time.time()
        if isinstance(filenames, list):
            for filename in filenames:
                with codecs.open(filename, 'r',encoding='utf-8') as src_file:
                    master_str += src_file.read()
        elif isinstance(filenames, str):
            with codecs.open(filenames, 'r',encoding='utf-8') as src_file:
                    master_str += src_file.read()
        print("Took {:.4f} seconds to generate master string".format(time.time() - t0))
        self.master_str += master_str.lower()
        t0 = time.time()
        self.master_tokens.extend(nltk.word_tokenize(self.master_str))
        for rm in self.remove_tokens:
            self.master_tokens = [word for word in self.master_tokens if word != rm]
        # self.master_tokens = [word for word in self.master_tokens]
        print("Took {:.2f} seconds to tokenize master string.".format(time.time() - t0))
        self.unique_tokens = list(set(self.master_tokens))

    def add_files_char(self, filenames):
        """
        Add files to the master source string for character based approach.
        args:
            - filenames: The name of the files to be added.
        """
        master_str = ""
        t0 = time.time()
        if isinstance(filenames, list):
            for filename in filenames:
                with codecs.open(filename, 'r',encoding='utf-8') as src_file:
                    master_str += src_file.read()
        elif isinstance(filenames, str):
            with codecs.open(filenames, 'r',encoding='utf-8') as src_file:
                    master_str += src_file.read()
        print("Took {:.4f} seconds to generate master string".format(time.time() - t0))
        self.master_str += master_str.lower()
        self.master_tokens.extend(list(master_str))
        for rm in self.remove_tokens:
            self.master_tokens.remove(rm)
        self.unique_tokens = list(set(self.master_tokens))


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

    # print(len(ngrammer.master_tokens))
    # print(len(ngrammer.unique_tokens))