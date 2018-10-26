
#!/usr/bin/python
import sys
import time
import math
from collections import Counter
import numpy as np
from scipy import sparse, io

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
        self.vocab_dict = {}
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
        #temp.append(Word('UNK'))
        for word in vocab_items:
            if word.word in vocab_given_dict:
                temp.append(word)
            #elif word.word in subsampled_dict:
            #    continue
            #elif word.count>=min_count:
            #    temp.append(word)
            #else:
                #temp[0].count += word.count
            else:
            	continue
        temp.sort(key=lambda word: word.count, reverse = True)
        vocab_dict = {}
        for i, word in enumerate(temp):
            vocab_dict[word.word] = i

        index = []
        output_index = []
        for word in words:
            if word in vocab_dict:
                index.append(vocab_dict[word])
            #elif word in subsampled_dict:
            #    continue
            else:
            	continue
            #    index.append(vocab_dict['UNK'])

        for word in vocab_given:
            if word in vocab_dict:           
                output_index.append(vocab_dict[word])
            else:
                output_index.append(np.random.randint(len(vocab_dict)))

        self.vocab_items = temp
        self.index = index
        self.output_index = output_index
        self.vocab_dict = vocab_dict


    def __len__(self):
        return len(self.vocab_items)


def build_cooccur(size, index, window_size, min_count):
	vocab_size = size
	cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),dtype=np.float64)
	for i in range(len(index)):
		pos = index[i]
		context = index[max(0,i - window_size):i]
		context_len = len(context)
		for j, context_id in enumerate(context):
			increment = 1.0/(context_len - j)
			cooccurrences[pos, context_id] += increment
			cooccurrences[context_id, pos] += increment
	return cooccurrences

# training
if __name__ == '__main__':
    window_size = 10
    min_count = 200
    learning_rate = 0.015
    x_max = 100
    alpha = 0.75
    embed_size = 100
    vocab = Vocab('text8','vocab.txt',min_count)
    vocab_size = len(vocab)
    index = vocab.index
    output_index = vocab.output_index
    print "vocab_size: ", vocab_size, ", len(index): ", len(index)
    print('building coccurrence matrix')
    #cooccurrences = build_cooccur(vocab_size, index, window_size, min_count)

    #io.mmwrite("coocc_extend.mtx", cooccurrences)
    cooccurrences = io.mmread("coocc.mtx")
    #cooccurrences = io.mmread("coocc_extend.mtx")
    #vocab_size = cooccurrences.shape[0]


    c = sparse.coo_matrix(cooccurrences)
    print('finish accesse the vocabulary and construct index set')

    W = (np.random.rand(vocab_size*2, embed_size) - 0.5) / float(embed_size + 1)
    b = (np.random.rand(vocab_size * 2) - 0.5) / float(embed_size + 1)
    gradsq = np.ones((vocab_size * 2, embed_size), dtype=np.float64)    
    gradsq_biases = np.ones(vocab_size * 2, dtype=np.float64)

    iteration = 0
    while iteration < 80:
        print('start training', iteration)
        global_loss=0
        for i,k,coocur in zip(c.row, c.col, c.data):
        	j = k + vocab_size
        	if coocur < x_max:
        		weight = (coocur/x_max)**alpha
        	else:
        		weight = 1
        	inner = W[i].dot(W[j])+b[i]+b[j]-math.log(coocur)
        	global_loss += 0.5*weight*(inner**2)
        	weight = weight*inner
        	coeff = learning_rate*weight
        	W[i] -= coeff*W[j]/np.sqrt(gradsq[i])
        	W[j] -= coeff*W[i]/np.sqrt(gradsq[j])
        	b[i] -= coeff/np.sqrt(gradsq_biases[i])
        	b[j] -= coeff/np.sqrt(gradsq_biases[j])
        	gradsq[i] += np.square(weight*W[j])
        	gradsq[j] += np.square(weight*W[i])
        	gradsq_biases[i] += (weight)**2
        	gradsq_biases[j] += (weight)**2
        print(global_loss)
        iteration = iteration + 1

    np.save('weight1.npy',W)

    vocab_given_file = open('vocab.txt', 'r').read()
    vocab_given = vocab_given_file.split()

    f = open('vectors.txt', 'w')
    for i in range(len(vocab_given)):
    	f.write(vocab_given[i])
    	f.write(' ')
        if vocab_given[i] in vocab.vocab_dict:
        	predict = ' '.join(str("%.3f"%(k)) for k in W[output_index[i]])
        	f.write(predict)
        f.write('\n')
    f.close()




        





