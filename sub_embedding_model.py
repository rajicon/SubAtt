#Trains the subword embeddings based on word2vec model
#Input: wordvec file, training vs val wordset


import tensorflow as tf
#import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.special import softmax as my_softmax
import keras
import numpy as np

from keras.layers import Input, Lambda, Dense, Layer, Embedding, Add, Flatten, Masking, Average, TimeDistributed, Concatenate
from keras.models import Model
import keras.backend as K

from keras_pos_embd import TrigPosEmbedding
import keras_transformer

from keras_transformer import transformer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_embed_sim import EmbeddingRet, EmbeddingSim



#my_cos_loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_similarity')


def custom_cos_distance_loss_function(y_actual, y_predicted):
    print(np.shape(y_actual))
    print(np.shape(y_predicted))
    y_actual = K.l2_normalize(y_actual, axis=-1)
    y_predicted = K.l2_normalize(y_predicted, axis=-1)
    return -K.mean(y_actual * y_predicted, axis=-1, keepdims=True)
    
from keras import backend as K
from keras.layers import Layer

'''
class SumInternal(Layer):

    def __init__(self, **kwargs):
        super(SumInternal, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(SumInternal, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        #return mask
        return None  #We don't need the current masking after this step!

    def call(self, x, mask):

        #mask = tf.Print(mask, [mask], message='Sum 1 Mask', summarize=1500) 

        mask = K.cast(mask, dtype = "float32")


        #mask = tf.Print(mask, [mask], message='Sum 1 Mask', summarize=1500) 

        mask = tf.expand_dims(mask, -1)

        #mask = tf.Print(mask, [mask], message='Sum 2 Mask', summarize=1500) 
        
        #x = tf.Print(x, [x], message='Sum Input Before Mask', summarize=1500)

        masked_vecs = x * mask


        #masked_vecs = tf.Print(masked_vecs, [masked_vecs], message='Sum Input After Mask', summarize=1500)

     
        return K.sum(masked_vecs, axis=1, keepdims=False)      
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
'''
        
class MeanInternal(Layer):

    def __init__(self, **kwargs):
        super(MeanInternal, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(MeanInternal, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        #return mask
        return None  #We don't need the current masking after this step!


    def call(self, x, mask):

        mask = K.cast(mask, dtype = "float32")

        #mask = tf.Print(mask, [mask], message='Mean Internal 1', summarize=1500)

        mask = tf.expand_dims(mask, -1)

        #mask = tf.Print(mask, [mask], message='Mean Internal 2', summarize=1500)

        #x = tf.Print(x, [x], message='Mean before mask !!!', summarize=1500)

        masked_vecs = x * mask

        #masked_vecs = tf.Print(masked_vecs, [masked_vecs], message='mean after mask !!!', summarize=1500)

        #print('quetzocoatl')
        #print(tf.count_nonzero(mask, axis=1) )
        
        total_count = tf.count_nonzero(mask, axis=1, dtype = "float32")  
        
        total_count = tf.Print(total_count, [total_count], message='total_count !!!', summarize=1500)

        
        return K.sum(masked_vecs, axis=1, keepdims=False) / total_count  
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

'''
class ZeroVectorMasker(Layer):

    def __init__(self, **kwargs):
        super(ZeroVectorMasker, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(ZeroVectorMasker, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask=None):

        
        #calculate zero vector masks

        zero_vector_mask =  tf.not_equal(tf.count_nonzero(inputs, axis=2), 0)
        #zero_vector_mask = K.cast(zero_vector_mask, dtype = "bool")


        #zero_vector_mask = tf.Print(zero_vector_mask, [zero_vector_mask], message='Value of Zero Vec Mask !!!', summarize=1500) 

        return zero_vector_mask  #We don't need the current masking after this step!

    def call(self, x, mask=None):

        #here we ignore the mask, the compute_mask will output the real one (based on 0 vectors inputed through here)
        #print('0vec call')
        #print(x)
        #print(mask)
        return x     
        
    def compute_output_shape(self, input_shape):
        return input_shape


class My_Printer(Layer):

    def __init__(self, in_message, **kwargs):
        self.in_message = in_message
        super(My_Printer, self).__init__(**kwargs)
      

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(My_Printer, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask=None):

        


        return mask  #We don't need the current masking after this step!

    def call(self, x, mask=None):

        x = tf.Print(x, [x], message=self.in_message, summarize=1500) 

        return x     
        
    def compute_output_shape(self, input_shape):
        return input_shape
'''
        
        
class Sub_Embedding_Model:
  def __init__(self, word_emb_dim, vocab_size, subword_vocab_size, max_amount_of_subwords=None, chosen_learning_rate=0.001):
    self.word_emb_dim = word_emb_dim
    self.max_amount_of_subwords = max_amount_of_subwords

    #self.word2id_dict = word2id_dict
    #self.word_embedding_model = word_embedding_model

    self.subword_vocab_size = subword_vocab_size

    self.chosen_learning_rate = chosen_learning_rate

  

    self.model = self.build_model()



         
  
  def build_model(self):

    subwords = Input(shape=(self.max_amount_of_subwords,), dtype=tf.int64)  
    #contexts = Input(shape=(self.max_num_context, self.max_num_words_per_context, ), dtype=tf.int64) #num_context x num_words
        
    sub_embs_np = np.zeros((self.subword_vocab_size + 1, self.word_emb_dim))  #MASK
    for rrr in range(self.subword_vocab_size):
        sub_embs_np[rrr+1] = np.random.random(self.word_emb_dim)



            

    
    sub_embs  = EmbeddingRet(
            input_dim=self.subword_vocab_size+1, #is this correct?
            output_dim=self.word_emb_dim,
            mask_zero=True, #is this what we want?  Think about this
            weights=[sub_embs_np],  #self.embedding_matrix,
            trainable=True,
            name='Sub-Embedding',
        )(subwords)



    final_estimate = MeanInternal()(sub_embs[0])
    #print(np.shape(final_estimate)) 
    


    model = Model(inputs=[subwords], outputs=final_estimate)
    
    

    my_adam = keras.optimizers.Adam(lr=self.chosen_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    #model.compile(loss=custom_cos_distance_loss_function, optimizer='adam', metrics=[])
    model.compile(loss=custom_cos_distance_loss_function, optimizer=my_adam, metrics=[])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])

    model.summary()

    return model

  def estimate_vector(self, word, contexts, word2id_dict, sub2id_dict, ngram_min=3, ngram_max=5, max_words_per_context = 50):
      extended_word = '<' + word + '>'
      subwords_list = self.extract_subwords(extended_word, ngram_min=ngram_min, ngram_max=ngram_max)
      #print('sub_list : '+str(subwords_list))

      subword_ind_list = []
      filtered_subword_list = []
      sub_pad_list = []
      context_pad_list = []
      for sub in subwords_list:
          if sub in sub2id_dict:
              subword_ind_list.append(sub2id_dict[sub])
              filtered_subword_list.append(sub)
      

      #print('implement nltk tokenize for each context, dont remove anything!'-9)
      context_inds_list = []
      for c in contexts:
          c_list = []
          '''
          for wrd in c.split():
              wrd = wrd.lower()
              wrd = wrd.replace('.','')
              wrd = wrd.replace('?','')
              wrd = wrd.replace(',','')
          '''

          tokens = c.split()

          
          for wrd in tokens:
              wrd = wrd.lower()
              wrd = wrd.replace('\n','')
              wrd = wrd.replace('\r','')

              if wrd in word2id_dict and wrd != word:
                  #print(wrd)
                  c_list.append(word2id_dict[wrd])

          
          while len(c_list) < max_words_per_context:
              c_list.append(0)

          if len(c_list) > max_words_per_context:
              c_list = c_list[0:max_words_per_context]
          

          context_inds_list.append(c_list)
      
      '''
      while len(context_inds_list) < 20:
          context_inds_list.append(np.zeros(max_words_per_context))
          context_pad_list.append('cont_pad')
      '''

      while len(subword_ind_list) < 84:
          subword_ind_list.append(0)

      subword_ind_list = np.array(subword_ind_list)
      context_inds_list = np.array(context_inds_list)

      
      new_contexts = []
      ci = 1
      for c in contexts:
          new_contexts.append('C'+str(ci) + ' : ' + c)
          ci = ci + 1


      #print(len(subword_ind_list))
      #print(len(context_inds_list))

      bbbbb = self.model.predict([np.array([subword_ind_list])])[0]
      return bbbbb





  def extract_subwords(self, extended_word, ngram_min=3, ngram_max=5):
      extracted_list = []
      for n in range(ngram_min, ngram_max+1):
          for i in range(len(extended_word)):
              if i+n <= len(extended_word):  #make sure not out of bounds
                  subword = extended_word[i:i+n]
                  extracted_list.append(subword)
                  
      return extracted_list



