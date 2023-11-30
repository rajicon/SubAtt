#The goal of this script is to split vocbulary into a train and validation set
#Subwords will only be trained/extracted from train set
#Main model will only be trained on train set and validate words on valid set
#This is how Hice does it and how I will do AM


#Input: gensim vectors and training corpus vocab (just to make sure vocabs are same)
#Output: a dictionary where key is vocab word and value is either 'train' or 'val'


import gensim
import random
import numpy as np
import pickle
import random
import sys

train_folder = sys.argv[1]

#print('should split based on stem, use split_vocabulary_into_train_and_val_BY_STEM.py'-8)


#train_folder = '/scratch/rpatel17/BERTRAM_2023_OUT/WWC_orig_preprocess/' 
#train_folder = '/scratch/rpatel17/LLAMA_OUT_OOV_2023/retokenized_corpuses/gpt2/retokenized_corpus/'

train_file_vocab = 'train.vwc100'



vocab_list = []
word_freq_dict = {}
word2id_dict = {}
f = open(train_folder+train_file_vocab)
for line in f:
    a = line.split(' ')
    word = a[0]
    freq = a[1]
    word_freq_dict[word] = int(freq)
    
    vocab_list.append(word)

f.close

print(len(word_freq_dict))

train_val_dict = {}
train_count = 0
val_count = 0
test_count = 0
for word in vocab_list:
    #import random
    aaa = random.uniform(0, 1)
    
    if aaa <= .8:
        train_val_dict[word] = 'train'
        train_count = train_count + 1
        print('train')
    elif aaa > .8 and aaa <= 1.0:
        train_val_dict[word] = 'val'
        val_count = val_count + 1
        print('val')
    '''
    else
        train_val_dict[word] = 'test'
        test_count = test_count + 1
        print('test')
    '''

        
total_count = train_count + val_count
print(total_count)
print(train_count/total_count)
print(val_count/total_count)

import pickle
pickle.dump( train_val_dict, open( train_folder+"train_val_dict.p", "wb" ) )

print(done)
