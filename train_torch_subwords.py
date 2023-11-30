#torch version of train_sub_embeddings.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np
import gensim
import os
import pickle

import sys
from tqdm import tqdm

import random
import math

from Subword_Torch import Subword_Torch



sys.path.insert(1, 'Combined_AttentionExperiments/')

from sub_data_loader import SubDataLoader


out_directory = sys.argv[1]
train_folder = sys.argv[2]
emb_file = sys.argv[3]
num_ep = int(sys.argv[4])  #number of epochs
dropout= float(sys.argv[5]) #dropout (if any)
min_amount_of_context = float(sys.argv[6]) #doesn't use
max_amount_of_context = float(sys.argv[7]) #doesn't use
input_lr = float(sys.argv[8])
min_file_ind = int(sys.argv[9])
max_file_ind = int(sys.argv[10])
my_seed = int(sys.argv[11])
min_word_freq = int(sys.argv[12])
train_val_dict_filename = sys.argv[13]

batch_size = 64

if not os.path.exists(out_directory):
	os.makedirs(out_directory)


def NonZero(sent):
    for word in sent:
        if word != 0:
            return True

    return False
    
def remove_blank_contexts_and_subs(subwords_list, contexts_set_list, labels_list):

    new_context_set_list2 = []
    new_sub_list2 = []
    new_label_list2 = []



    for aaa in range(len(contexts_set_list)):
        for sent in contexts_set_list[aaa]:
            if NonZero(sent) == True:
                new_context_set_list2.append(contexts_set_list[aaa])
                new_sub_list2.append(subwords_list[aaa])
                new_label_list2.append(labels_list[aaa])
                break

    new_context_set_list2 = np.array(new_context_set_list2)
    new_sub_list2 = np.array(new_sub_list2)
    new_label_list2 = np.array(new_label_list2)

    new_context_set_list3 = []
    new_sub_list3 = []
    new_label_list3 = []




    for bbb in range(len(new_sub_list2)):
        if NonZero(new_sub_list2[bbb]) == True:
            new_context_set_list3.append(new_context_set_list2[bbb])
            new_sub_list3.append(new_sub_list2[bbb])
            new_label_list3.append(new_label_list2[bbb])
        


    new_context_set_list3 = np.array(new_context_set_list3)
    new_sub_list3 = np.array(new_sub_list3)
    new_label_list3 = np.array(new_label_list3)
    
    return new_sub_list3, new_context_set_list3, new_label_list3

 
word_emb_file = emb_file

train_file_vocab = 'train.vwc100'

train_val_dict = pickle.load( open( train_val_dict_filename, "rb" ) )

train_count = 0
val_count = 0
for wrd in train_val_dict:
    if train_val_dict[wrd] == 'train':
        train_count = train_count + 1

    if train_val_dict[wrd] == 'val':
        val_count = val_count + 1
        
print('train count : '+str(train_count))
print('val count : '+str(val_count))
print('total : '+str(len(train_val_dict.keys())))
print('--')        

data = SubDataLoader(vocab_path=train_folder+train_file_vocab, word_emb_file=word_emb_file, train_val_dict=train_val_dict, subword_type='ngram', ngram_dropout = dropout, min_freq=min_word_freq)
#data = SubDataLoader(vocab_path=train_folder+train_file_vocab, word_emb_file=word_emb_file, train_val_dict=train_val_dict, subword_type='ngram', ngram_dropout = dropout, min_freq=min_word_freq, min_sub_freq=10)

vocab_size = len(data.word_embeddings.wv.vocab)

subword_vocab_size = len(data.sub2id_dict)
word_emb_dim = data.word_embeddings.wv.vector_size  #400 #300


print('voc size : '+str(vocab_size))
print('sub voc size : '+str(subword_vocab_size))




data.ngram_dropout = dropout #train on ngram dropout
train_subwords_list, train_contexts_set_list, train_labels_list, DUMMY_val_subwords_list, DUMMY_val_contexts_set_list, DUMMY_val_labels_list = data.load_from_file(train_folder, min_amount_of_context, max_amount_of_context, max_words_in_context=50, randomly_dropout_one_side = False, min_file_ind=min_file_ind, max_file_ind=max_file_ind)

del(DUMMY_val_subwords_list)
del(DUMMY_val_contexts_set_list)
del(DUMMY_val_labels_list)

data.ngram_dropout = 0.0 #don't dropout on val
DUMMY_train_subwords_list, DUMMY_train_contexts_set_list, DUMMY_train_labels_list, val_subwords_list, val_contexts_set_list, val_labels_list = data.load_from_file(train_folder, 1, 64, max_words_in_context=50, randomly_dropout_one_side = False, min_file_ind=min_file_ind, max_file_ind=max_file_ind)  #Val Set should ALWAYS be varying range of context sizes, as in real life we have varying context sizes

del(DUMMY_train_subwords_list)
del(DUMMY_train_contexts_set_list)
del(DUMMY_train_labels_list)

train_subwords_list, train_contexts_set_list, train_labels_list = remove_blank_contexts_and_subs(train_subwords_list, train_contexts_set_list, train_labels_list)
val_subwords_list, val_contexts_set_list, val_labels_list = remove_blank_contexts_and_subs(val_subwords_list, val_contexts_set_list, val_labels_list)



print(np.shape(train_subwords_list))
print(np.shape(train_contexts_set_list))
print(np.shape(val_subwords_list))
print(np.shape(val_contexts_set_list))

lr_decay=0.5
threshold=1e-3
patience=4
lr_early_stop=1e-7

lr_init = input_lr


    
device = torch.device("cuda:%d" % 0)
print(device)

#LOAD MODELS
our_model = Subword_Torch(data, dim=word_emb_dim).to(device)


##########

optimizer = torch.optim.Adam(our_model.parameters(), lr = lr_init, eps=1e-07)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay, patience = patience, mode='max', threshold=threshold)

best_valid_cosine = -1

 
my_num_batch_train = int(len(train_subwords_list)/batch_size) + 1 #want to go through whole data set, not random
my_num_batch_val = int(len(val_subwords_list)/batch_size) + 1 #want to go through whole data set, not random




for epoch in range(num_ep):
    print('epoch  :  '+str(epoch)+'--------------------')  
    
    #shuffle each epoch together
    temp = list(zip(train_subwords_list, train_contexts_set_list, train_labels_list))
    random.shuffle(temp)
    train_subwords_list, train_contexts_set_list, train_labels_list = zip(*temp)
    
    print('=' * 100)
    train_cosine = []
    valid_cosine = []
    our_model.train()
    with tqdm(np.arange(my_num_batch_train), desc='Train') as monitor:
        for batch in monitor:
            start_ind = batch * batch_size
            end_ind = (batch + 1) * batch_size

            train_sub_tensor = torch.LongTensor(train_subwords_list[start_ind:end_ind]).to(device)
            #train_cc_tensor = torch.LongTensor(train_contexts_set_list[start_ind:end_ind]).to(device)
            train_label_tensor = torch.FloatTensor(train_labels_list[start_ind:end_ind]).to(device)
                              
                           
            optimizer.zero_grad()
            


            pred_emb = our_model.forward(train_sub_tensor)
            assert not torch.isnan(pred_emb).any()
            loss = -F.cosine_similarity(pred_emb, train_label_tensor).mean() 
            full_loss = loss
            assert not torch.isnan(full_loss).any()
            del pred_emb
            del train_sub_tensor
            del train_label_tensor

            full_loss.backward()
            optimizer.step()
            train_cosine += [[-loss.cpu().detach().numpy()]]
            monitor.set_postfix(train_status = train_cosine[-1])
            del loss
            del full_loss
    our_model.eval()

    with torch.no_grad():
        with tqdm(np.arange(my_num_batch_val), desc='Valid') as monitor:
            for batch in monitor:
                start_ind = batch * batch_size
                end_ind = (batch + 1) * batch_size


                val_sub_tensor = torch.LongTensor(val_subwords_list[start_ind:end_ind]).to(device)
                val_label_tensor = torch.FloatTensor(val_labels_list[start_ind:end_ind]).to(device)
             

                pred_emb = our_model.forward(val_sub_tensor)
            
                assert not torch.isnan(pred_emb).any()

            
                loss = -F.cosine_similarity(pred_emb, val_label_tensor).mean()
            
                assert not torch.isnan(loss).any()
                    
                                

                del pred_emb
                del val_sub_tensor
                del val_label_tensor


                valid_cosine += [[-loss.cpu().numpy()]]
                monitor.set_postfix(valid_status = valid_cosine[-1])
                del loss
    

    print('-' * 100)
    avg_train, avg_valid = np.average(train_cosine, axis=0)[0], np.average(valid_cosine, axis=0)[0]
    print(("Epoch: %d: Train Cosine: %.4f; Valid Cosine: %.4f; LR: %f") \
            % (epoch, avg_train, avg_valid, optimizer.param_groups[0]['lr']))
    scheduler.step(avg_valid)
    
    if avg_valid > best_valid_cosine:
        print('best ep saved : '+str(epoch))
        best_saved_epoch = str(epoch)
        best_valid_cosine = avg_valid
        with open(os.path.join(out_directory, 'model.pt'), 'wb') as f:
            torch.save(our_model, f)
        with open(os.path.join(out_directory, 'optimizer.pt'), 'wb') as f:
            torch.save(optimizer.state_dict(), f)
        


pickle.dump( data.word2id_dict, open( out_directory+"/word2id_dict.p", "wb" ) )
pickle.dump( data.sub2id_dict, open( out_directory+"/sub2id_dict.p", "wb" ) )
pickle.dump( data, open( out_directory+"/full_data_loader.p", "wb" ) )

    
my_filename = out_directory + '/finished.txt'
outfile = open(my_filename,'w')
outfile.write('finished'+'\n')


outfile.write(str(best_valid_cosine)+'\n')
outfile.write('best ep '+best_saved_epoch)


outfile.close()


loaded_model = torch.load(out_directory+'/model.pt')

#recently added
subword_file = open(out_directory+'/best_ep_subword_emb_gensim.txt' ,'w')  
subword_file.write('{} {}\n'.format(len(data.id2sub_dict), data.word_embeddings.wv.vector_size))


for sub_ind in data.id2sub_dict:  #shouldn't have mask but just in case!
	if sub_ind != 0: #this is a mask
		subword = data.id2sub_dict[sub_ind]  #the actual subword
		sub_emb = loaded_model.sub_emb.weight[sub_ind].cpu().detach().numpy()
	
		str_vec = ' '.join(map(str, list(sub_emb)))
		subword_file.write('{} {}\n'.format(subword, str_vec))
subword_file.close()


