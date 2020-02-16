from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import spacy
from time import time
import pickle
from numpy.linalg import norm # for computing similarity

import warnings
warnings.filterwarnings("error")


__authors__ = ['Mewe-Hezoudah KAHANAM','Mouad BOUCHATTAOUI']
__emails__  = ['mewe-hezoudah.kahanam@student.ecp.fr','mouad.bouchattaoui@student.ecp.fr']

# decorator for printing code execution time
def executionTime(input_func):
	def wrapper(*args, **kwargs):
		start_t = time()
		result = input_func(*args, **kwargs)
		end_t = time()
		print("[%s] execution time: %.4f sec"%(input_func.__name__, end_t-start_t))
		return result
	return wrapper

@executionTime
def text2sentences(path):
	nlp = spacy.load("en_core_web_sm")
	# feel free to make a better tokenization/pre-processing
	sentences = []
	with open(path) as f:
		lines = f.readlines()
	
	# this approche increases speed by 10s
	for l in lines:
		doc = nlp(l.lower())
		no_punc_sentence = [token.orth_ for token in doc if not token.is_punct | token.is_space]
		sentences.append(tuple(no_punc_sentence))
	
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

	def train(self, epoch=2, learn_rate=0.01, low=-0.8, high=0.8, initialize=True):
		"""Training our model
		
		Keyword Arguments:
			epoch {int} -- number of epoch for the training (default: {10})
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
			t=time()
			for counter, sentence in enumerate(self.trainset):
				sentence = list(map(lambda word: word if word in self.vocab else '<unk>', sentence))

				for wpos, word in enumerate(sentence):
					wIdx = self.w2id[word]
					winsize = np.random.randint(self.winSize) + 1
					start = max(0, wpos - winsize)
					end = min(wpos + winsize + 1, len(sentence))
					
					for context_word in sentence[start:end]:
						ctxtId = self.w2id[context_word]
						if ctxtId == wIdx: continue
						negativeIds = self.sample({wIdx, ctxtId})
						self.trainWord(wIdx, ctxtId, negativeIds)
						self.trainWords += 1
					self.target += 10**-6

				if counter % 1000 == 0:
					print(' > training %d of %d' % (counter, len(self.trainset)))
					self.loss.append(self.accLoss / self.trainWords)
					self.trainWords = 0
					self.accLoss = 0.
			
			self.learn_rate = 1/( ( 1+self.learn_rate*(1+ne) ) ) 
			e = time()
			print("time : %f"%(e-t))

	def trainWord(self, wordId, contextId, negativeIds):
		alpha = 0.1
		c = self.context[contextId]
		t = self.target[wordId]
		n = self.context[negativeIds]
		
		# compute the grad
		grad_c = -( 1-expit(alpha*np.dot(c,t)) )*t
		grad_t = -( 1-expit(alpha*np.dot(c,t)) )*c + np.sum( ( 1 - expit(-alpha*np.dot(t,n.T)) )*n.T, 1)
		
		# update weights
		self.context[contextId] = c - self.learn_rate*alpha*grad_c
		self.target[wordId] = t - self.learn_rate*alpha*grad_t

		# compute the loss
		loss = np.log( 1 + np.exp(-np.dot(c,t)) ) + np.sum( np.log( 1 + np.exp(np.dot(t,n.T)) ) )
		self.accLoss += loss

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