
from sub_data_loader import SubDataLoader
from sub_embedding_model import Sub_Embedding_Model
#from Sub_Embedding_Model_MSE import Sub_Embedding_Model_MSE
import numpy as np
import pickle
import h5py
import keras
import random
import sys
from keras.callbacks import ModelCheckpoint

model_type = sys.argv[1]
out_directory = sys.argv[2]
train_folder = sys.argv[3]
emb_file = sys.argv[4]
num_ep = int(sys.argv[5])  #number of epochs
dropout= float(sys.argv[6]) #dropout (if any)
min_amount_of_context = float(sys.argv[7]) #min context size, not really used here
max_amount_of_context = float(sys.argv[8]) #max context size, not really used here
input_lr = float(sys.argv[9])
min_file_ind = int(sys.argv[10])
max_file_ind = int(sys.argv[11])
my_seed = int(sys.argv[12])
min_word_freq = int(sys.argv[13])

#word_emb_file = "/scratch/rpatel17/August26_Herbelot_and_Baroni_Embs/herbelot_and_baroni400.txt"

word_emb_file = emb_file

train_file_vocab = 'train.vwc100'

train_val_dict = pickle.load( open( '/scratch/rpatel17/April9_Fixed_Min_Freq/train_val_dict.p', "rb" ) )

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

vocab_size = len(data.word_embeddings.wv.vocab)

subword_vocab_size = len(data.sub2id_dict)
word_emb_dim = data.word_embeddings.wv.vector_size  #400 #300


print('voc size : '+str(vocab_size))
print('sub voc size : '+str(subword_vocab_size))


data.ngram_dropout = dropout #train on ngram dropout
train_subwords_list, train_contexts_set_list, train_labels_list, DUMMY_val_subwords_list, DUMMY_val_contexts_set_list, DUMMY_val_labels_list = data.load_from_file(train_folder, min_amount_of_context, max_amount_of_context, max_words_in_context=50, randomly_dropout_one_side = False, input_seed=my_seed)

del(DUMMY_val_subwords_list)
del(DUMMY_val_contexts_set_list)
del(DUMMY_val_labels_list)

data.ngram_dropout = 0.0 #don't dropout on val
DUMMY_train_subwords_list, DUMMY_train_contexts_set_list, DUMMY_train_labels_list, val_subwords_list, val_contexts_set_list, val_labels_list = data.load_from_file(train_folder, 1, 64, max_words_in_context=50, randomly_dropout_one_side = False)  #Val Set should ALWAYS be varying range of context sizes, as in real life we have varying context sizes

del(DUMMY_train_subwords_list)
del(DUMMY_train_contexts_set_list)
del(DUMMY_train_labels_list)

print('dim : '+str(word_emb_dim))
#print('grar'-3)

#remove blank subword breakdowns for Train Set
new_train_sub_list = []
new_train_label_list = []
q = 0
for sub_list in train_subwords_list:
    for sub in sub_list:
        if sub != 0:
            new_train_sub_list.append(sub_list)
            new_train_label_list.append(train_labels_list[q])
                
            break #so only counted once
    q = q + 1
  
print(len(train_subwords_list))
print(len(new_train_sub_list))
new_train_sub_list = np.array(new_train_sub_list)
new_train_label_list = np.array(new_train_label_list)




#remove blank subword breakdowns for Val Set
new_val_sub_list = []
new_val_label_list = []
q = 0
for sub_list in val_subwords_list:
    for sub in sub_list:
        if sub != 0:
            new_val_sub_list.append(sub_list)
            new_val_label_list.append(val_labels_list[q])
                
            break #so only counted once
    q = q + 1
  
print(len(val_subwords_list))
print(len(new_val_sub_list))
new_val_sub_list = np.array(new_val_sub_list)
new_val_label_list = np.array(new_val_label_list)


if model_type == 'COS':
    combined_model = Sub_Embedding_Model(word_emb_dim, vocab_size, subword_vocab_size,chosen_learning_rate=input_lr)

elif model_type == 'MSE':
    combined_model = Sub_Embedding_Model_MSE(word_emb_dim, vocab_size, subword_vocab_size,chosen_learning_rate=input_lr)




print(np.shape(train_subwords_list))
print(np.shape(train_contexts_set_list))
print(np.shape(val_subwords_list))
print(np.shape(val_contexts_set_list))
print(min_word_freq)
#print('stop'-9)

oov_word = 'cat'
contexts = ['the cat is suspicious', 'he had many cat s']
aaa = combined_model.estimate_vector(oov_word, contexts, data.word2id_dict, data.sub2id_dict)

oov_word = 'seven'
contexts = ['there are seven people', 'seven is after six', 'seven is a magic number']
bbb = combined_model.estimate_vector(oov_word, contexts, data.word2id_dict, data.sub2id_dict)





mcp_save = ModelCheckpoint(out_directory+'best_ep-{epoch:03d}.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
combined_model.model.fit([new_train_sub_list], new_train_label_list, validation_data=([new_val_sub_list], new_val_label_list), epochs=num_ep, callbacks=[mcp_save], verbose=2)




print('----------------------')


combined_model.model.save_weights(out_directory+'pretrained_sub_weights.h5')
pickle.dump( data.word2id_dict, open( out_directory+"word2id_dict.p", "wb" ) )
pickle.dump( data.sub2id_dict, open( out_directory+"sub2id_dict.p", "wb" ) )
pickle.dump( data, open( out_directory+"full_data_loader.p", "wb" ) )


#save subwords as w2v model:
emb_matrix = combined_model.model.get_layer('Sub-Embedding').get_weights()[0]
print(np.shape(emb_matrix))

subword_file = open(out_directory+'/subword_emb_gensim.txt' ,'w')  
subword_file.write('{} {}\n'.format(len(data.id2sub_dict), word_emb_dim))

for sub_ind in data.id2sub_dict:  #shouldn't have mask but just in case!
    if sub_ind != 0: #this is a mask
        subword = data.id2sub_dict[sub_ind]  #the actual subword
        sub_emb = emb_matrix[sub_ind]
        
        str_vec = ' '.join(map(str, list(sub_emb)))
        subword_file.write('{} {}\n'.format(subword, str_vec))
subword_file.close()
        
        


