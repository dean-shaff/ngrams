#textprocessor.py
import nltk
import codecs
import time 
import numpy as np 

debug = 1

class TextProcessor(object):
    """
    A class for creating processing text data. 
    """ 
    def __init__(self, filenames, mode,**kwargs):
        """
        Initialize the TextProcessor object. 
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


if __name__ == "__main__":
    tp = TextProcessor(["texts/AustenPride.txt",
                    "texts/DickensTaleofTwo.txt",
                    "texts/china.txt","texts/fairy.txt",
                    "texts/harperlee.txt"], "char")#,remove_tokens=[u"``",u"''",u"--"])

