#This is analogous to data_loader, but used for the subword training initially
#This one will take in the train vs val as input, and will output sub_dict

#data_loader
#for now, using FCM preprocessing!
import gensim
import random
import numpy as np
import pickle
import random

class SubDataLoader():
  def __init__(self, vocab_path, word_emb_file, train_val_dict, min_freq=5, min_sub_freq=4, subword_type='ngram', min_ngram=3, max_ngram=5, ngram_dropout=0.0):
    self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(word_emb_file, binary=False)

    self.min_freq = min_freq
    self.min_sub_freq = min_sub_freq
    self.subword_type = subword_type

    self.min_ngram = min_ngram
    self.max_ngram = max_ngram
    
    self.train_val_dict = train_val_dict

    if subword_type == 'segment':
        print('breaking words into morpheme')
        import morfessor
        morfessor_model_file = "CombinedAttentionExperiments/morfessor_model.bin"
        morfessor_io = morfessor.MorfessorIO()
        self.morfessor_model = morfessor_io.read_binary_model_file(morfessor_model_file)

    self.ngram_dropout = ngram_dropout


    self.max_amount_of_subwords = 0
    self.word2id_dict, self.id2word_dict, self.word_freq_dict = self.build_word2id_dict(vocab_path, self.word_embeddings)
    self.sub2id_dict, self.id2sub_dict, self.sub_freq_dict, self.sub_breakdown = self.build_sub2id_dict(vocab_path, self.min_sub_freq, self.train_val_dict)
    non_match = 0
    for word in self.word_embeddings.wv.vocab:
        if word not in self.word2id_dict:
            print(word)
            #non_match = non_match + 1
            
    for word in self.word2id_dict:
        if word not in self.word_embeddings.wv.vocab:
            print(word)

   
  #creates word2id and word_freq_dict
  #used before Herbelot word embs
  '''
  def build_word2id_dict(self,vocab_path):
      ind = 1
      word_freq_dict = {}
      word2id_dict = {}
      f = open(vocab_path)
      for line in f:
          a = line.split(' ')
          word = a[0]
          freq = a[1]
          word_freq_dict[word] = int(freq)
          word2id_dict[word] = ind
          ind = ind + 1
      f.close
      
      #with only indexes with enough freq!
      new_word2id_dict = {}
      id2word_dict = {}
      ind = 1
      for key in word2id_dict:

          if word_freq_dict[key] >= self.min_freq:
              new_word2id_dict[key] = ind
              id2word_dict[ind] = key
              ind = ind + 1
      return new_word2id_dict, id2word_dict, word_freq_dict
    '''
    
  #frequency from vocab path, word2id from w2v vecs
  def build_word2id_dict(self, vocab_path, w2v):
      #ind = 1 #0 is pad
      word_freq_dict = {}
      word2id_dict = {}
      id2word_dict = {}
      f = open(vocab_path)
      for line in f:
          a = line.split(' ')
          word = a[0]
          freq = a[1]
          word_freq_dict[word] = int(freq)
          #word2id_dict[word] = ind
          #ind = ind + 1
      f.close
      
      #no frequency restriction here, that should be ok!
      ind = 1   #0 is pad
      for w2v_word in w2v.wv.vocab:
          word2id_dict[w2v_word] =  ind
          id2word_dict[ind] = w2v_word
          ind = ind + 1
      
      
      return word2id_dict, id2word_dict, word_freq_dict    
    
    
    
    
    

  def build_sub2id_dict(self, vocab_path, min_sub_freq, train_val_dict, start_token='<', end_token='>'):
   
      if len(start_token) > 1 or len(end_token) > 1:
          raise ValueError('start_token and end_token should only be at most 1 character')
      sub_ind = 1
      sub2id_dict = {}
      sub_freq_dict = {}
      sub_breakdown = {}
      f = open(vocab_path)
      for line in f:
          bbb = line.split(' ')
          word = bbb[0]
          
          if train_val_dict[word] == 'train':
          
              extended_word = start_token+word+end_token

              if self.subword_type == 'segment':
                  extracted_subwords = self.extract_subwords_segment(word)  #doesn't use extended_word word!


              else:
                  extracted_subwords = self.extract_subwords(extended_word)

              if len(extracted_subwords) > self.max_amount_of_subwords:
                  self.max_amount_of_subwords = len(extracted_subwords)

              sub_breakdown[word] = extracted_subwords
              for e in extracted_subwords:
                  if e not in sub2id_dict:
                      sub2id_dict[e] = sub_ind
                      sub_ind = sub_ind + 1
                      sub_freq_dict[e] = 1
                  else:
                      sub_freq_dict[e] = sub_freq_dict[e] + 1
                      
          else:
               if train_val_dict[word] != 'val':
                   print('error, train_val_dict not train or val for this word:')
                   print(word)
                   print(train_val_dict[word])
                   print('stop'-9)      
                
      f.close
      
      new_sub2id_dict = {}
      new_sub_breakdown = {}
      id2sub_dict = {}
      sub_ind = 1
      for key in sub2id_dict:
          if sub_freq_dict[key] >= self.min_sub_freq:
              new_sub2id_dict[key] = sub_ind
              #new_sub_breakdown[key] = sub_breakdown[key]
              id2sub_dict[sub_ind] = key
              sub_ind = sub_ind + 1
      
      return new_sub2id_dict, id2sub_dict, sub_freq_dict, sub_breakdown
  
  def extract_subwords(self, extended_word):
      extracted_list = []
      for n in range(self.min_ngram, self.max_ngram+1):
          for i in range(len(extended_word)):
              if i+n <= len(extended_word):  #make sure not out of bounds
                  subword = extended_word[i:i+n]
                  extracted_list.append(subword)
                  
      return extracted_list

  def extract_subwords_segment(self, word):
      extracted_list = self.morfessor_model.viterbi_segment(word)[0]      
      return extracted_list



  #based on form-context model code
  def load_from_file(self, corpuspath, min_cont_size, max_cont_size, max_words_in_context=50, randomly_dropout_one_side=False, start_token='<', end_token='>', input_seed=10, min_file_ind=0, max_file_ind=25):
      train_labels_list = []
      train_contexts_set_list = []
      train_subwords_set_list = []
    
      val_labels_list = []
      val_contexts_set_list = []
      val_subwords_set_list = []
      
      random.seed(input_seed)

      for qq in range(min_file_ind, max_file_ind):
          print(qq)
          f = open(corpuspath+'train.bucket'+str(qq)+'.txt')
          for line in f:
            #comps = re.split(r'\t', line)
            comps = line.split('\t')


            word = comps[0]
            

            extended_word = start_token+word+end_token
            
            if self.subword_type == 'segment':
                subwords_list = self.extract_subwords_segment(word)

            else:
                subwords_list = self.extract_subwords(extended_word)
            
            #verify working correctly
            if self.train_val_dict[word] == 'train':
                aaaa = set(self.sub_breakdown[word])
                bbbb = set(subwords_list)
                
                if aaaa != bbbb:
                    print(aaaa)
                    print(bbbb)
                    print('sets dont match for some reason'-9)
            

            #print(word)
            #print(subwords_list)
            #print('--')
            
            new_subwords_list = []  #only take subwords we are using
            for sss in subwords_list:
                if sss in self.sub2id_dict:
                    
                    new_subwords_list.append(sss)

            #print(new_subwords_list)
            #print('-----')

              


                  
            ind_subwords_list = []
            for s in range(self.max_amount_of_subwords):
                if s < len(new_subwords_list):
                    ind_subwords_list.append(self.sub2id_dict[new_subwords_list[s]])

                    if self.sub2id_dict[new_subwords_list[s]] == 0:
                        print('selected subword zero!!!' - 4)
                else:
                    ind_subwords_list.append(0)
            all_contexts = comps[1:]
            

            
            random.shuffle(all_contexts)

            if len(comps) == 2 and comps[1] == '\n':
                continue

            occurrences = self._get_occurrences(word)
            
            #if self.train_val_dict[word] == 'train':
                #print(str(word)+' : '+str(occurrences))
            
            for _ in range(occurrences):
                label = self.word_embeddings.wv[word] #if occurences are 0, shouldn't get here!

                number_of_contexts = random.randint(min_cont_size, max_cont_size) # should be 20
                contexts = all_contexts[:number_of_contexts]
                
                if len(contexts) > 0:
                    contexts_list = []

                    for c in contexts:
                        context_inds = []
                        #print(len(c.split()))
                        for wrd in c.split():
                            if wrd in self.word2id_dict and wrd != word:  #need to remove our target word (maybe replace it with a mask token)
                                context_inds.append(self.word2id_dict[wrd])
                        #print(len(context_inds))
                        while len(context_inds) < max_words_in_context:
                            context_inds.append(0)
                        if len(context_inds) > max_words_in_context:
                            context_inds = context_inds[0:max_words_in_context]

                        contexts_list.append(context_inds)

                
                    while len(contexts_list) < max_cont_size:
                        contexts_list.append(np.zeros(len(contexts_list[0])))
                    
                    
                    
                    new_ind_subwords_list = []
                    for p in ind_subwords_list: 
                        
                        if self.ngram_dropout == 0.0:
                            new_ind_subwords_list.append(p)
                        else:
                            if random.random() > self.ngram_dropout:
                                new_ind_subwords_list.append(p)

                    #append 0s so correct size again
                    if self.ngram_dropout != 0.0:  
                        while len(new_ind_subwords_list) < self.max_amount_of_subwords:
                            new_ind_subwords_list.append(0)


                    
                    #labels_list.append(label)
                    if self.train_val_dict[word] == 'train':
                        train_labels_list.append(label)
                    elif self.train_val_dict[word] == 'val':
                        val_labels_list.append(label)
                    else:
                        print('word not train or val:')
                        print(word)
                        print(self.train_val_dict[word])
                        print('stop'-9)

                    if randomly_dropout_one_side == False:

                        #contexts_set_list.append(contexts_list)
                        #subwords_set_list.append(new_ind_subwords_list)
                        if self.train_val_dict[word] == 'train':
                            train_contexts_set_list.append(contexts_list)
                            train_subwords_set_list.append(new_ind_subwords_list)
                        elif self.train_val_dict[word] == 'val':
                            val_contexts_set_list.append(contexts_list)
                            val_subwords_set_list.append(new_ind_subwords_list)
                        else:
                            print('word not train or val:')
                            print(word)
                            print(self.train_val_dict[word])
                            print('stop'-9)                        
                    

                    
                    else:
                        print('dropout commented out, adapt to train and val'-9)
                        '''
                        dropped_contexts_list = []
                        for r in range(20):
                            dropped_contexts_list.append(np.zeros(len(contexts_list[0])))
                        
                        dropped_sub_list = []
                        for r in range(self.max_amount_of_subwords):
                            dropped_sub_list.append(0)

                        a = random.random()

                        if a <= .2:
                            #make sure sub isn't blanked
                            sub_blank = True
                            for r in new_ind_subwords_list:
                                if r != 0:
                                    sub_blank = False
                                    break
                            if sub_blank == False:
                                contexts_set_list.append(dropped_contexts_list)
                                subwords_set_list.append(new_ind_subwords_list)

                        elif a <= .4:
                            #make sure context isn't blank
                            context_blank = True
                            for r1 in contexts_list:
                                for r2 in r1:
                                    if r2 != 0:
                                        context_blank = False
                                        break
                            if context_blank == False:
                                contexts_set_list.append(contexts_list)
                                subwords_set_list.append(dropped_sub_list)

                        else:
                            contexts_set_list.append(contexts_list)
                            subwords_set_list.append(new_ind_subwords_list)
                        '''
                       


                    del (all_contexts[:number_of_contexts])
      
      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))   
      


      c_train = list(zip(train_subwords_set_list, train_contexts_set_list, train_labels_list))
      
      train_subwords_set_list, train_contexts_set_list, train_labels_list= zip(*c_train)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))    

      train_subwords_set_list = np.array(train_subwords_set_list)
      train_contexts_set_list = np.array(train_contexts_set_list)
      train_labels_list = np.array(train_labels_list)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))               
      c_val = list(zip(val_subwords_set_list, val_contexts_set_list, val_labels_list))
      
      val_subwords_set_list, val_contexts_set_list, val_labels_list= zip(*c_val)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))    

      val_subwords_set_list = np.array(val_subwords_set_list)
      val_contexts_set_list = np.array(val_contexts_set_list)
      val_labels_list = np.array(val_labels_list)      
      
      return train_subwords_set_list, train_contexts_set_list, train_labels_list, val_subwords_set_list, val_contexts_set_list, val_labels_list
  
  def _get_occurrences(self, word):
      if word not in self.word_freq_dict or word not in self.word_embeddings:
          return 0
      word_count = self.word_freq_dict[word]
      return int(max(1, min(word_count / self.min_freq, 5)))
    #load corpus
    

    
  def load_one_example_per_val_word(self, corpuspath, min_cont_size, max_cont_size, max_words_in_context=50, randomly_dropout_one_side=False, start_token='<', end_token='>', input_seed=10):
      train_labels_list = []
      train_contexts_set_list = []
      train_subwords_set_list = []
      train_words = []
      
      val_labels_list = []
      val_contexts_set_list = []
      val_subwords_set_list = []
      val_words = []
      
      random.seed(input_seed)

      for qq in range(25):
          print(qq)
          f = open(corpuspath+'train.bucket'+str(qq)+'.txt')
          for line in f:
            #comps = re.split(r'\t', line)
            comps = line.split('\t')


            word = comps[0]
            
            #if word == '121,000' and word not in self.word_embeddings.wv.vocab:
            #    continue
                
                
            #subwords_list = self.sub_breakdown[word]  #can't do this bc val set doesn't work here
            
            extended_word = start_token+word+end_token
            
            if self.subword_type == 'segment':
                subwords_list = self.extract_subwords_segment(word)

            else:
                subwords_list = self.extract_subwords(extended_word)
            
            #verify working correctly
            if self.train_val_dict[word] == 'train':
                aaaa = set(self.sub_breakdown[word])
                bbbb = set(subwords_list)
                
                if aaaa != bbbb:
                    print(aaaa)
                    print(bbbb)
                    print('sets dont match for some reason'-9)
            

            #print(word)
            #print(subwords_list)
            #print('--')
            
            new_subwords_list = []  #only take subwords we are using
            for sss in subwords_list:
                if sss in self.sub2id_dict:
                    
                    new_subwords_list.append(sss)

            #print(new_subwords_list)
            #print('-----')

              


                  
            ind_subwords_list = []
            for s in range(self.max_amount_of_subwords):
                if s < len(new_subwords_list):
                    ind_subwords_list.append(self.sub2id_dict[new_subwords_list[s]])

                    if self.sub2id_dict[new_subwords_list[s]] == 0:
                        print('selected subword zero!!!' - 4)
                else:
                    ind_subwords_list.append(0)
            all_contexts = comps[1:]
            

            
            random.shuffle(all_contexts)

            if len(comps) == 2 and comps[1] == '\n':
                continue

            occurrences = self._get_occurrences(word)
            for _ in range(min(1, occurrences)):
                label = self.word_embeddings.wv[word]

                number_of_contexts = random.randint(min_cont_size, max_cont_size) # should be 20
                contexts = all_contexts[:number_of_contexts]
                
                if len(contexts) > 0:
                    contexts_list = []

                    for c in contexts:
                        context_inds = []
                        #print(len(c.split()))
                        for wrd in c.split():
                            if wrd in self.word2id_dict and wrd != word:  #need to remove our target word (maybe replace it with a mask token)
                                context_inds.append(self.word2id_dict[wrd])
                        #print(len(context_inds))
                        while len(context_inds) < max_words_in_context:
                            context_inds.append(0)
                        if len(context_inds) > max_words_in_context:
                            context_inds = context_inds[0:max_words_in_context]

                        contexts_list.append(context_inds)

                
                    while len(contexts_list) < max_cont_size:
                        contexts_list.append(np.zeros(len(contexts_list[0])))
                    
                    
                    
                    new_ind_subwords_list = []
                    for p in ind_subwords_list: 
                        
                        if self.ngram_dropout == 0.0:
                            new_ind_subwords_list.append(p)
                        else:
                            if random.random() > self.ngram_dropout:
                                new_ind_subwords_list.append(p)

                    #append 0s so correct size again
                    if self.ngram_dropout != 0.0:  
                        while len(new_ind_subwords_list) < self.max_amount_of_subwords:
                            new_ind_subwords_list.append(0)


                    
                    #labels_list.append(label)
                    if self.train_val_dict[word] == 'train':
                        train_labels_list.append(label)
                        train_words.append(word)
                        
                    elif self.train_val_dict[word] == 'val':
                        val_labels_list.append(label)
                        val_words.append(word)
                    else:
                        print('word not train or val:')
                        print(word)
                        print(self.train_val_dict[word])
                        print('stop'-9)

                    if randomly_dropout_one_side == False:

                        #contexts_set_list.append(contexts_list)
                        #subwords_set_list.append(new_ind_subwords_list)
                        if self.train_val_dict[word] == 'train':
                            train_contexts_set_list.append(contexts_list)
                            train_subwords_set_list.append(new_ind_subwords_list)
                        elif self.train_val_dict[word] == 'val':
                            val_contexts_set_list.append(contexts_list)
                            val_subwords_set_list.append(new_ind_subwords_list)
                        else:
                            print('word not train or val:')
                            print(word)
                            print(self.train_val_dict[word])
                            print('stop'-9)                        
                    

                    
                    else:
                        print('dropout commented out, adapt to train and val'-9)
                        '''
                        dropped_contexts_list = []
                        for r in range(20):
                            dropped_contexts_list.append(np.zeros(len(contexts_list[0])))
                        
                        dropped_sub_list = []
                        for r in range(self.max_amount_of_subwords):
                            dropped_sub_list.append(0)

                        a = random.random()

                        if a <= .2:
                            #make sure sub isn't blanked
                            sub_blank = True
                            for r in new_ind_subwords_list:
                                if r != 0:
                                    sub_blank = False
                                    break
                            if sub_blank == False:
                                contexts_set_list.append(dropped_contexts_list)
                                subwords_set_list.append(new_ind_subwords_list)

                        elif a <= .4:
                            #make sure context isn't blank
                            context_blank = True
                            for r1 in contexts_list:
                                for r2 in r1:
                                    if r2 != 0:
                                        context_blank = False
                                        break
                            if context_blank == False:
                                contexts_set_list.append(contexts_list)
                                subwords_set_list.append(dropped_sub_list)

                        else:
                            contexts_set_list.append(contexts_list)
                            subwords_set_list.append(new_ind_subwords_list)
                        '''
                       


                    del (all_contexts[:number_of_contexts])
      
      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))   
      


      c_train = list(zip(train_subwords_set_list, train_contexts_set_list, train_labels_list, train_words))
      
      train_subwords_set_list, train_contexts_set_list, train_labels_list, train_words= zip(*c_train)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))    

      train_subwords_set_list = np.array(train_subwords_set_list)
      train_contexts_set_list = np.array(train_contexts_set_list)
      train_labels_list = np.array(train_labels_list)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))               
      c_val = list(zip(val_subwords_set_list, val_contexts_set_list, val_labels_list, val_words))
      
      val_subwords_set_list, val_contexts_set_list, val_labels_list, val_words = zip(*c_val)

      #print(np.shape(subwords_set_list))
      #print(np.shape(contexts_set_list))
      #print(np.shape(labels_list))    

      val_subwords_set_list = np.array(val_subwords_set_list)
      val_contexts_set_list = np.array(val_contexts_set_list)
      val_labels_list = np.array(val_labels_list)      
      
      return train_subwords_set_list, train_contexts_set_list, train_labels_list, train_words, val_subwords_set_list, val_contexts_set_list, val_labels_list, val_words


#for am 
  def load_one_example_per_val_word_text(self, corpuspath, min_cont_size, max_cont_size, max_words_in_context=50, randomly_dropout_one_side=False, start_token='<', end_token='>', input_seed=10):

      wordList = []
      contextList = []
      
      
      
      random.seed(input_seed)

      for qq in range(25):
          print(qq)
          f = open(corpuspath+'train.bucket'+str(qq)+'.txt')
          for line in f:
            #comps = re.split(r'\t', line)
            comps = line.split('\t')


            word = comps[0]
            
            #if word == '121,000' and word not in self.word_embeddings.wv.vocab:
            #    continue
                
                
            #subwords_list = self.sub_breakdown[word]  #can't do this bc val set doesn't work here
            
            extended_word = start_token+word+end_token
            
            if self.subword_type == 'segment':
                subwords_list = self.extract_subwords_segment(word)

            else:
                subwords_list = self.extract_subwords(extended_word)
            
            #verify working correctly
            if self.train_val_dict[word] == 'train':
                aaaa = set(self.sub_breakdown[word])
                bbbb = set(subwords_list)
                
                if aaaa != bbbb:
                    print(aaaa)
                    print(bbbb)
                    print('sets dont match for some reason'-9)
            

            #print(word)
            #print(subwords_list)
            #print('--')
            
            new_subwords_list = []  #only take subwords we are using
            for sss in subwords_list:
                if sss in self.sub2id_dict:
                    
                    new_subwords_list.append(sss)

            #print(new_subwords_list)
            #print('-----')

              


                  
            ind_subwords_list = []
            for s in range(self.max_amount_of_subwords):
                if s < len(new_subwords_list):
                    ind_subwords_list.append(self.sub2id_dict[new_subwords_list[s]])

                    if self.sub2id_dict[new_subwords_list[s]] == 0:
                        print('selected subword zero!!!' - 4)
                else:
                    ind_subwords_list.append(0)
            all_contexts = comps[1:]
            

            
            random.shuffle(all_contexts)

            if len(comps) == 2 and comps[1] == '\n':
                continue

            occurrences = self._get_occurrences(word)
            for _ in range(min(1, occurrences)):
                label = self.word_embeddings.wv[word]

                number_of_contexts = random.randint(min_cont_size, max_cont_size) # should be 20
                contexts = all_contexts[:number_of_contexts]
                
                if self.train_val_dict[word] == 'val':
                    wordList.append(word)
                    contextList.append(contexts)
                    
            
                
      return wordList, contextList
                


  