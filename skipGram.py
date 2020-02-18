from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from time import time
import pickle
from numpy.linalg import norm # for computing similarity
from multiprocessing import Pool # for multiprocessing
import string

import warnings
warnings.filterwarnings("error")


__authors__ = ['Mewe-Hezoudah KAHANAM','Mouad BOUCHATTAOUI']
__emails__  = ['mewe-hezoudah.kahanam@student.ecp.fr','mouad.bouchattaoui@student.ecp.fr']

translator = str.maketrans('', '', string.punctuation)

# decorator for printing code execution time
def executionTime(input_func):
	def wrapper(*args, **kwargs):
		start_t = time()
		result = input_func(*args, **kwargs)
		end_t = time()
		print("[%s] execution time: %.4f sec"%(input_func.__name__, end_t-start_t))
		return result
	return wrapper


def tokenize(lines):
	docs = []
	for l in lines:
		doc = l.lower().translate(translator)
		docs.append(tuple(doc.split()))
	return docs


@executionTime
def text2sentences(path, n_jobs=3):
	# feel free to make a better tokenization/pre-processing
	with open(path) as f:
		lines = f.readlines()
	
	# this approche increases speed
	chunks = [lines[i::n_jobs] for i in range(n_jobs)]
	pool = Pool(processes=n_jobs)

	sentences = pool.map(tokenize, chunks)
	pool.close() # close the processes
	sentences = sum(sentences, []) # merge the results of all processes
	
	return sentences


def count_word(sentences):
	all_sentences = sum(sentences, ())
	
	return all_sentences

@executionTime
def get_vocab(sentences, minCount=5):
	counts = {}

	for sentence in sentences:
		for word in sentence:
			if word in counts.keys():
				counts[word] += sentence.count(word)
			else:
				counts[word] = sentence.count(word)
	vocab = {}
	n_unk = 0
	for word in counts:
		if counts[word] > minCount:
			vocab[word] = counts[word]
		else:
			n_unk += counts[word]
	
	vocab['<unk>'] = n_unk
	return vocab

@executionTime
def get_scores(counts):
	alpha = 0.75
	scores = {word: counts[word]**alpha for word in counts.keys()}
	n = sum(scores.values())
	scores = {word: scores[word]/n for word in scores.keys()}
	return scores

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
		self.winSize = winSize
		self.negativeRate = negativeRate
		self.nEmbed = nEmbed
		# setting the training data
		counts = get_vocab(sentences, minCount=minCount)
		# training words dictionnary
		self.vocab = counts.keys()
		# word to ID mapping
		self.w2id = {word: counter for (counter, word) in enumerate(self.vocab)}
		# distribution for negative sampling
		self.scores = get_scores(counts)

		# model words that are less than minCount as <unk>
		vocab_filter = lambda word: word if word in self.vocab else '<unk>'
		# set of training sentences
		self.trainset = [tuple(map(vocab_filter, sentence)) for sentence in sentences]

	def initialize(self, low=-0.8, high=0.8, learn_rate=0.01):
		self.learn_rate = learn_rate
		self.trainWords = 0
		self.accLoss = 0
		self.loss = []

		# initialize variables for training process
		v_size = len(self.vocab)
		self.target = np.random.uniform(low, high, (v_size, self.nEmbed))
		self.context = np.random.uniform(low, high, (v_size, self.nEmbed))
		self.target = normalize(self.target)
		self.context = normalize(self.context)

	@executionTime
	def sample(self, omit):
		wordIds = []
		scores = []
		for word in self.vocab:
			wIdx = self.w2id[word]
			if wIdx not in omit:
				wordIds.append(wIdx)
				scores.append(self.scores[word])
		scores = np.array(scores)/np.sum(scores)

		negative_words = np.random.choice(wordIds, size=self.negativeRate, replace=False, p=scores)

	def train(self, learn_rate=0.01, low=-0.8, high=0.8, initialize=True):
		# if we want to initialize the model parameters
		if initialize:
			self.initialize(low=low, high=high, learn_rate=learn_rate)

		# train the corpus
		t=time()
		for counter, sentence in enumerate(self.trainset):

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))

# t = time()
# e = time()
# print("time : %f"%(e-t))