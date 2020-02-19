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
from multiprocessing import Pool, Process, Queue # for multiprocessing
import string

import warnings
warnings.filterwarnings("error")
from functools import partial


__authors__ = ['Mewe-Hezoudah KAHANAM','Mouad BOUCHATTAOUI']
__emails__  = ['mewe-hezoudah.kahanam@student.ecp.fr','mouad.bouchattaoui@student.ecp.fr']

translator = str.maketrans('', '', string.punctuation)

def get_neg_sample(negativeRate, omit, vocab, vocab_pert):
	v_size = len(vocab)

	vocab_pert += np.sum(vocab_pert[omit]) / (v_size - len(omit))
	vocab_pert[omit] = 0

	return np.random.choice(v_size, size=negativeRate, replace=False, p=vocab_pert)


def sentences_2data(sentences, negativeRate, winSize, vocab, vocab_id, vocab_pert):
	data = []
	for sentence in sentences:
		wIds = list()
		negativeIds = list()
		ctxtIds = list()
		for wpos, word in enumerate(sentence):
			wIdx = vocab_id[word]
			winsize = np.random.randint(winSize//2) + 1
			start = max(0, wpos - winsize)
			end = min(wpos + winsize + 1, len(sentence))
			ctxtIds_ = [ vocab_id[w] for w in sentence[start:end] if vocab_id[w] != wIdx ]
			if len(ctxtIds_) == 0: continue # avoid sentences with all <unk>
			wIds.append(wIdx)
			negativeIds.append([get_neg_sample(negativeRate, [wIdx, ctxtId], vocab, vocab_pert) for ctxtId in ctxtIds_])
			ctxtIds.append(ctxtIds_)
		data.append([wIds, ctxtIds, negativeIds])
	return data


def get_train_data(sentences, negativeRate, winSize, vocab, vocab_id, vocab_pert, n_jobs=3):
	n = len(sentences)
	result = []

	for i in range(0, n, 1000):
		t = time()
		data = sentences[i:i+1000]

		chunks = []
		for j in range(n_jobs):
			chunk = data[j::n_jobs]
			chunks += [(chunk, negativeRate, winSize, vocab, vocab_id, vocab_pert)]
		pool = Pool(processes=n_jobs)
		res = pool.starmap(sentences_2data, chunks)
		pool.close()
		result += sum(res, [])
		
		e = time() - t
		print("> loaded %d of %d in %.4f(s)"% (i+1000, n, e) )
	print(len(result))
	return result

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
		docs.append(doc.split())
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
			try:
				counts[word] += sentence.count(word)
			except:
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
		self.vocab = list(counts.keys())
		# word to ID mapping
		self.w2id = {word: counter for (counter, word) in enumerate(self.vocab)}
		# distribution for negative sampling
		scores = get_scores(counts)
		self.scores = np.array([scores[word] for word in self.vocab])

		# model words that are less than minCount as <unk>
		# set of training sentences
		self.trainset = []
		for sentence in sentences:
			for pos, word in enumerate(sentence):
				try:
					counts[word]
				except:
					sentence[pos] = '<unk>'
			self.trainset.append(tuple(sentence))

	def initialize(self, low=-0.8, high=0.8, learn_rate=0.01):
		self.learn_rate = learn_rate
		self.trainWords = 0
		self.accLoss = 0
		self.loss = []

		# initialize variables for training process
		v_size = len(self.vocab)
		self.target = np.random.uniform(low, high, (v_size, self.nEmbed))
		self.context = np.random.uniform(low, high, (v_size, self.nEmbed))

	def train(self, learn_rate=0.01, low=-0.8, high=0.8, initialize=True):
		# if we want to initialize the model parameters
		if initialize:
			self.initialize(low=low, high=high, learn_rate=learn_rate)

		train_data = get_train_data(self.trainset, self.negativeRate, self.winSize, self.vocab, self.w2id, self.scores, n_jobs=3)
		# train the corpus
		for epoch in range(4):
			print("Epoch ", epoch)
			t=time()
			for counter, data in enumerate(train_data):
				wIds, ctxtIds, negativeIds = data
				for pos, wIdx in enumerate(wIds):

					for cpos, ctxtId in enumerate(ctxtIds[pos]):
						nIds = negativeIds[pos][cpos]
						self.trainWord(wIdx, ctxtId, nIds)
						self.trainWords += 1

				if counter % 1000 == 999:
					print(' > training %d of %d' % (counter, len(self.trainset)))
					self.loss.append(self.accLoss / self.trainWords)
					self.trainWords = 0
					self.accLoss = 0.
					print("time = ", time() - t)
					t = time()
			print(self.loss)
			self.trainWords = 0
			self.accLoss = 0
			self.loss = []

	def trainWord(self, wordId, contextId, negativeIds):
		c = self.context[contextId]
		n = self.context[negativeIds]
		t = self.target[wordId]

		grad_c = (expit(np.dot(c,t)) - 1)*t
		grad_n = expit(np.dot(n,t)).reshape(-1,1)*t
		grad_t = (expit(np.dot(c,t)) - 1)*c + np.sum(expit(np.dot(n,t)).reshape(-1,1)*n, 0)

		self.context[contextId] -= self.learn_rate*grad_c
		self.context[negativeIds] -= self.learn_rate*grad_n
		self.target[wordId] -= self.learn_rate*grad_t

		self.accLoss += -( np.log( expit(np.dot(c,t)) ) + np.sum(np.log( expit(-np.dot(n,t)) )) )
		pass

	def save(self,path):
		with open(path, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
		
	def similarity(self,word1,word2):
		# create a function to get word id return id of <unk> if word not in vocab
		word_id = lambda word: self.w2id[word] if word in self.vocab else self.w2id['<unk>']

		# get the embeddings
		w1 = self.target[word_id(word1)]
		w2 = self.target[word_id(word2)]

		score = np.dot(w1, w2) / (norm(w1) * norm(w2))
		score = (score+1)*np.exp(score+1)/(2*np.exp(2))
		score = round(score, 4)

		# print(self.vocab[word_id(word1)], ' -- ', self.vocab[word_id(word2)], ' -- ', score)
		
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
		sg.train(learn_rate=0.05)
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