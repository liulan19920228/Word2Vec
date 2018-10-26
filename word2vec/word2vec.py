#!/usr/bin/python
import sys
import time
import math
from collections import Counter
import numpy as np

#start_time = time.time()
#input_file = open(sys.argv[1],"r").readlines()

class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0

class Vocab:
    def __init__(self, input_file, vocab_txt, min_count):
        self.vocab_item = []
        self.index = []
        self.output_index=[]
        self.build_vocab(input_file, vocab_txt, min_count)

    def build_vocab(self, input_file, vocab_txt, min_count):
        corpus = open(input_file,'r').read()
        vocab_hash = {}
        vocab_items = []
        words = corpus.split()
        num_words = 0
    # Count the number of all words in the document
        for word in words:
            if word not in vocab_hash:
                vocab_hash[word] = len(vocab_items)
                vocab_items.append(Word(word))
            vocab_items[vocab_hash[word]].count += 1
            num_words += 1

    # This is the subsampling, the subsampled word is stored in subsampled_dict
        subsampled_dict = {}
        for word in vocab_items:
            if (math.sqrt(word.count/(0.001*num_words))+1)*(0.001*num_words/word.count) < np.random.rand():
                subsampled_dict[word.word] = 1

    # The given vocabulary is stored in vocab_given_dict
        vocab_given_file = open(vocab_txt, 'r').read()
        vocab_given = vocab_given_file.split()
        vocab_given_dict = {}
        for word in vocab_given:
            vocab_given_dict[word] = 1
    
    # Filter the word occurs less than min count. Remove the subsampled word. Leave the given vocabulary.
        temp = []
        temp.append(Word('UNK'))
        for word in vocab_items:
            if word.word in vocab_given_dict:
                temp.append(word)
            elif word.word in subsampled_dict:
                continue
            elif word.count>=min_count:
                temp.append(word)
            else:
                temp[0].count += word.count
        temp[1:].sort(key=lambda word: word.count, reverse = True)
        vocab_dict = {}
        for i, word in enumerate(temp):
            vocab_dict[word.word] = i

        index = []
        output_index = []
        for word in words:
            if word in vocab_dict:
                index.append(vocab_dict[word])
            elif word in subsampled_dict:
                continue
            else:
                index.append(vocab_dict['UNK'])
        for word in vocab_given:
            if word in vocab_dict:           
                output_index.append(vocab_dict[word])
            else:
                output_index.append(vocab_dict['UNK'])

        self.vocab_items = temp
        self.index = index
        self.output_index = output_index


    def __len__(self):
        return len(self.vocab_items)

class NegativeSamples:
    def __init__(self, vocab):
        z = sum([math.pow(word.count, 0.75) for word in vocab])
        length = int(1e8)
        sample_list = np.zeros(length, dtype = np.uint32)
        prob = 0
        i = 0
        for j, word in enumerate(vocab):
            prob += float(math.pow(word.count, 0.75))/z
            while i<length and float(i)/length < prob:
                table[i] = j
                i += 1
        self.sample_list = sample_list 

    def sample_set(self, sample_size):
        indices = np.random.randint(low = 0, high = len(self.sample_list), size = sample_size)
        return [self.sample_list[i] for i in indices]



def sigmoid(x):
    return 1/(1+math.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# training
if __name__ == '__main__':

    max_window_size = 1
    embed_size = 100
    sample_size = 10
    learning_rate = 0.025
    min_count = 5
    vocab = Vocab('text8','vocab.txt',min_count)
    vocab_size = len(vocab)
    index = vocab.index
    output_index = vocab.output_index
    print('finish accesse the vocabulary and construct index set')
    print(len(index))
    NegSample = NegativeSamples(vocab.vocab_item)
    in_weight = np.random.uniform(low = -1, high = 1, size = (vocab_size, embed_size))
    out_weight = np.random.uniform(low = -0.1, high = 0.1, size = (embed_size, vocab_size))
   

    vocab_given_file = open('vocab.txt', 'r').read()
    vocab_given = vocab_given_file.split()

    print("end writing")
    epoch = 0
    while epoch < 6:
        print('start trainig, epoch number', epoch)
        for i in range(3, len(index)-3):
            pos = index[i]
            context1 = index[i-1]
            context2 = index[i+1]
            context3 = index[i-2]
            context4 = index[i+2]
            context5 = index[i+3]
            context6 = index[i+3]

            EH = np.zeros(embed_size)
            #neg_sam = []
            #window_size = np.random.randint(low = 1, high = max_window_size + 1)
            #context_set = index[i-window_size:i]+index[i+1:i+window_size+1]
            #while len(neg_sam) < sample_size:
            #    w = NegSample.sample_list[np.random.randint(int(1e8))]
            #    if not (w in context_set or w == pos):
            #        neg_sam.append(w)
            classifier =[(context1,1),(context2,1), (context3,1),(context4,1),(context5,1),(context6,1)]
            for k in NegSample.sample_set(sample_size):
                if not (k in [context1,context2,context3,context4,context5, context6,pos]):
                    classifier += [(k,0)]
            for j,t in classifier:
                EI = t - sigmoid(np.dot(out_weight[:,j],in_weight[pos,:]))
                EH += EI*out_weight[:,j]
                out_weight[:,j] += learning_rate*EI*in_weight[pos,:]
            in_weight[pos,:] += learning_rate*EH
        #time2=time.time()
            if i % 100000 == 0:
                learning_rate = max(learning_rate - 0.025/3000,0000, 0.001*0.25)
                loss = 0
                for j, t in classifier:
                    if t == 0:
                        t = -1
                    loss -= math.log(sigmoid(t*out_weight[:,j].dot(in_weight[pos,:])))         
                print(epoch, i)
                print(loss)
        epoch = epoch + 1

    f = open('vectors.txt', 'w')
    for i in range(len(vocab_given)):
        predict = ' '.join(str("%.3f"%(k)) for k in in_weight[output_index[i]])
        f.write(vocab_given[i])
        f.write(' ')
        f.write(predict)
        f.write('\n')
    f.close()




        





