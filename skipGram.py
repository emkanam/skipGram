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


def lines_to_tuple(lines):
	sentences = []
	for l in lines:
		sentence = l.lower().translate(translator)
		sentences.append(tuple(sentence.split()))
	return sentences


@executionTime
def text2sentences(path, n_jobs=3):
	# feel free to make a better tokenization/pre-processing
	with open(path) as f:
		lines = f.readlines()
	
	# this approche increases speed by 10s
	chunks = [lines[i::n_jobs] for i in range(n_jobs)]
	pool = Pool(processes=n_jobs)

	result = pool.map(lines_to_tuple, chunks)
	pool.close() # close the processes
	sentences = sum(result, []) # merge the results of all processes
	
	return sentences


def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
		self.winSize = winSize
		self.negativeRate = negativeRate
		self.nEmbed = nEmbed
		# setting values
		self.vocab = ['<unk>'] # list of valid words
		list_words = sum(sentences, ())
		for word in set(list_words):
			if list_words.count(word) > minCount:
				self.vocab.append(word)
		
		# model words that are less than minCount as <unk>
		vocab_filter = lambda word: word if word in self.vocab else '<unk>'
		self.trainset = {tuple(map(vocab_filter, sentence)) for sentence in sentences} # set of sentences
		
		# word to ID mapping
		self.w2id = {word: counter for (counter, word) in enumerate(self.vocab)}

		# create score for words (speeds up the sampling function)
		alpha = 0.75
		counts = np.zeros(len(self.vocab))
		for word in self.vocab:
			word_idx = self.w2id[word]
			count_map = map(lambda sentence: sentence.count(word), self.trainset)
			counts[word_idx] = sum(count_map)
		self.scores = np.power(counts, alpha) / np.sum(np.power(counts, alpha))

		# one hot encoding
		self.one_hot = np.eye(len(self.vocab))

	def initialize(self, low=-0.8, high=-0.8, learn_rate=0.01):
		"""Initializing the model parameters using uniform random law.
		The values are also normalized using a min-max normalization.
		
		Keyword Arguments:
			low {float} -- lower bound for parameters value (default: {-0.8})
			high {float} -- upper bound for parameters value (default: {0.8})
			learn_rate {float} -- learning rate for the optimization (default: {0.01})
		"""

		# initialize trainig metrics
		self.learn_rate = learn_rate
		self.trainWords = 0
		self.accLoss = 0
		self.loss = []

		# initialize target and context matrices
		vocab_size = len(self.vocab)
		self.target = np.random.uniform(low, high, (vocab_size, self.nEmbed))
		self.context = np.random.uniform(low, high, (vocab_size, self.nEmbed))
		self.target = normalize(self.target)
		self.context = normalize(self.context)
	
	def sample(self, omit):
		"""samples negative words, ommitting those in set omit"""
		omit = list(omit)
		vocab_size = len(self.vocab)

		scores = np.copy(self.scores)
		scores += np.sum(scores[omit]) / (vocab_size - len(omit))
		scores[omit] = 0

		negative_words = np.random.choice(vocab_size, size=self.negativeRate, replace=False, p=scores)
		return list(negative_words)

	def train(self, epoch=2, batch_size=32, learn_rate=0.01, low=-0.8, high=0.8, initialize=True):
		"""Training our model
		
		Keyword Arguments:
			epoch {int} -- number of epoch for the training (default: {10})
			batch_size {int} -- batches for the training (default: {10})
			learn_rate {float} -- learning rate for the optimization (default: {0.01})
			low {float} -- lower bound for parameters value (default: {-0.8})
			high {float} -- upper bound for parameters value (default: {0.8})
			initialize {bool} -- if we want to initialize the parameters (default: {True})
		"""

		# if we want to initialize the model parameters
		if initialize:
			self.initialize(low=low, high=high, learn_rate=learn_rate)

		# train on epoch
		for ne in range(epoch):
			for counter, sentence in enumerate(self.trainset):
				sentence = list(map(lambda word: word if word in self.vocab else '<unk>', sentence))
				t=time()
				for wpos, word in enumerate(sentence):
					wIdx = self.w2id[word]
					winsize = self.winSize // 2
					start = max(0, wpos - winsize)
					end = min(wpos + winsize + 1, len(sentence))
					ctxtIds = list({ self.w2id[w] for w in sentence[start:end] if self.w2id[w] != wIdx })
					if len(ctxtIds) == 0: continue # avoid sentences with all <unk>
					negativeIds = [self.sample({wIdx, ctxtId}) for ctxtId in ctxtIds]
					# train all the context words at once
					self.trainWord(wIdx, ctxtIds, negativeIds)
					self.trainWords += len(ctxtIds)
				e = time()
				print("time : %f"%(e-t))

				if counter % 1000 == 0:
					print(' > training %d of %d' % (counter, len(self.trainset)))
					self.loss.append(self.accLoss / self.trainWords)
					self.trainWords = 0
					self.accLoss = 0.
			
			self.learn_rate = 1/( ( 1+self.learn_rate*(1+ne) ) ) 

	def trainWord(self, wordId, contextIds, negativeIds):
		alpha = 0.1
		t = self.target[wordId]

		c = np.dot(self.one_hot[contextIds], self.context)
		n = np.zeros((len(c), self.negativeRate, self.nEmbed))
					
		for i in range(len(c)):
			n[i,:,:] = self.context[negativeIds[i]]
		
		
		k = self.nEmbed
		l = len(c)
		m = self.negativeRate
		# compute the grad
		v_ct = -( 1-expit(alpha*np.dot(c,t)) )
		v_nt = 1 - expit( -alpha*np.dot( t, n.reshape((l,k,-1)) ) )
		grad_c = v_ct.reshape(-1,1)*t.reshape(1,-1)
		grad_t = np.dot(v_ct, c) + np.sum(np.einsum('ij,ijk->jk', v_nt,n), 0)
		
		# update weights
		self.context[contextIds] = c - self.learn_rate*alpha*grad_c
		self.target[wordId] = t - self.learn_rate*alpha*grad_t

		# compute the loss
		# loss = np.log( 1 + np.exp(-np.dot(c,t)) ) + np.sum( np.log( 1 + np.exp(np.dot(t,n.T)) ) )
		self.accLoss += 0

	def save(self,path):
		with open(path, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def similarity(self,word1,word2):
		"""Computes similarity between two words. Unknown words are mapped to <unk> embedding vector.
		the similarity is computed by : \alpha_{w_1, w_2} = \frac{1 + \cos(w_1,w_2)}{2} 
		
		Arguments:
			word1 {string} -- first word
			word2 {string} -- second word
		
		Returns:
			float -- a score \in [0,1] indicating the similarity (the higher the more similar)
		"""

		# create a function to get word id return id of <unk> if word not in vocab
		word_id = lambda word: self.w2id[word] if word in self.vocab else self.w2id['<unk>']

		# get the embeddings
		w1 = self.target[word_id(word1)]
		w2 = self.target[word_id(word2)]

		score = ( 1 + np.dot(w1, w2) / (norm(w1) * norm(w2)) ) / 2.0
		score = round(score, 4)

		# print("%s -- %s == %f"%(self.vocab[word1_id], self.vocab[word2_id], score))

		return score

	@staticmethod
	def load(path):
		with open(path, 'rb') as model_file:
			model = pickle.load(model_file)
		return model

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