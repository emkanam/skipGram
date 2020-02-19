---
title: SkipGram Word2Vec embedding
date: "February 2020"
author: Mewe-Hezoudah KAHANAM, Mouad BOUCHATTAOUI 
---

# I- Data loading

## A- Sentences loading
The training data is loaded from a text file where each line contains a sentence. This is done by calling the function `text2sentences`. Each sentence in the file is prepocessed by removing punctuations and lowercasing each word. We the split the sentence so it becomes alist of words (see function `tokenize`).

The data loading process can take a lot of time as the number of sentences grows. Hence we decided to parallelize the task by using multyprocessing (3 processes by default) and we achieved 100k sentences within about 4.8 seconds.

## B- Vocabulary and noise distribution

### 1- Vocabulary
The function `get_vocab` creates a dictionnary where the key are the words of our vocabulary and the values are the count of each word in the corpus. For each sentence, we loop over each word and add the word and its count from the sentence to our dictionnary (if the key already exists, we add the count).

All the words that appears less than `minCount` times are removed and the key `<unk>` (for representing unknown words) is added with value the sum of the values of removed words.

### 2- Noise distribution
The noise distribution is defined in [1] as :

$$
P(w) = \frac{c(w)^\alpha}{\sum_{w^{'}} c(w^{'})^\alpha}
$$

Where $c(w)$ is the number of times the word $w$ appears in the corpus and $\alpha$ a hyperparameter (chosen equal to 0.75 from [1]).

The distribution is used when sampling negative words (see [negative sampling](#negative-sampling)) by guiving high probabilities to rare words and low probability to recurent words. The function `get_scores` implements this method and is quite easy to handle.

# II- The SkipGram Model

## A- Model initialization
We use `get_vocab` to get the vocabulary of our model. We also store the noise distribution from `get_scores` as an attribute. We then loop over all the sentences passed to the constructor and replace all the words that are not in our vocab by `<unk>`. This process is quite fast as it took around 2 seconds for 100k words.

## B- Training the model

### 1- Training data
The primal opproach of the training process is quite slow (took around 30s for a corpus of 1000 sentences and 700s for 10k sentences). This is due to the fact that we are creating the trainning data as we loop over each word in each sentence. Hence we decided to create the training data one time before the training begins. By doing so, we were able to use some multiprocessing and speeded up the trainig process (since we speed up the data loading). We obtained 10s for 1000 sentences and 127s for 10k sentences (train data loading takes from 5s to 12s for 1000 sentences depending on the size of the corpus, but the training loop takes around 6s for 1000 words).

Let $(s_1, \cdots, s_K)$ be the set of sentences in the corpus. A sentence $s_k$ is represented by $(w_{k,1}, \cdots, w_{k,N_k})$, where $w_{k,n}$ are the words and $N_k$ the size of the sentence (total words). For the word $w_{k,n}$ in sentence $s_k$ we associate the context words $(c_{k,n,1}, \cdots, c_{k,n,L_n})$ and for each context word $c_{k,n,l}$ we get the negative samples $(m_{k,n,l,1}, \cdots, m_{k,n,l,J})$. One can see that the $k-th$ value in the train data is:

$$
\begin{bmatrix}
\pi(w_{k,1}) & \cdots & \pi(w_{k,N_k})
\end{bmatrix}
$$

$$
\begin{bmatrix}
\pi(c_{k,1,1}) & \cdots & \pi(c_{k,1,L_1})\\
 & \vdots & \\
\pi(c_{k,N_k,1}) & \cdots & \pi(c_{k,N_k,L_{N_k}})
\end{bmatrix}\\
$$

$$
\begin{bmatrix}
\begin{bmatrix}
\pi(m_{k,1,1,1}) & \cdots & \pi(m_{k,1,1,J})\\
& \vdots & \\
\pi(m_{k,1,L_1,1}) & \cdots & \pi(m_{k,1,L_1,J})
\end{bmatrix}
&
\cdots
&
\begin{bmatrix}
\pi(m_{k,N_k,1,1}) & \cdots & \pi(m_{k,N_k,1,J})\\
& \vdots & \\
\pi(m_{k,N_k,L_{N_k},1}) & \cdots & \pi(m_{k,N_k,1,J})
\end{bmatrix}
\end{bmatrix}
$$

Where $\pi(w)$ is the index of the word in the vocabulary. This process is implemented in `get_train_data`.

#### Negative sampling
The negative sampling is implemented in `get_neg_sample`. We zero out the scores of words we want to omit and we add the mean of their scores to remaining words so that the total score remains 1 (we could have also rescaled the scores array). We then use `numpy.random.choice` to sample the negative words.

### 2- The training process
For each data in the training data, we recover the words indexes, the context words and the corresponding negative words. We send each word and the corresponding context and negative words to `trainWord` which will process the forward and backward pass.

#### a- Forward pass
Here we compute the loss of the model given by:

$$
L(t,c,n) = -\log(\sigma(c.t)) - \sum_{n \in W_{neg}} \log(\sigma(-n.t))
$$

Where $\sigma$ is the sigmoid function, $t$ the target embedding for the word, $c$ and $n$ the context embedings for the context word and negative words.

#### b- Backward pass
Here we compute the derivative of the loss with respect to $t$, $c$ and $n$ and we update the model's weights. The derivatives are given by:

$$
\frac{\partial L}{\partial c}(t,c,n) = (\sigma(c.t)-1)t
$$

$$
\frac{\partial L}{\partial n}(t,c,n) = \sigma(n.t)t
$$

$$
\frac{\partial L}{\partial t}(t,c,n) = (\sigma(c.t)-1)c + \sum_{n \in W_{neg}} \sigma(n.t)n
$$

The update is done by using a gradient descent, where $\eta$ is the learning rate.:

$$
t_{k+1} = t_k - \eta\frac{\partial L}{\partial t}(t_k,c_k,n_k)  
$$

$$
c_{k+1} = c_k - \eta\frac{\partial L}{\partial c}(t_k,c_k,n_k) 
$$

$$
n_{k+1} = n_k - \eta\frac{\partial L}{\partial n}(t_k,c_k,n_k)
$$

## C- Computing similarity
We compute the similarity between two words by using the dot product between their target embedings. We could have used 1 + the cosine similarity.

[1]: http://slashdot.org