
#torch version of subword 


import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np
import gensim




class Subword_Torch(nn.Module):
    def __init__(self, data, dim=400):
        super(Subword_Torch, self).__init__()
        
        self.data = data
        
        self.dim = dim
        
       
        
        sub_num_embeddings = len(data.sub2id_dict)

            
        self.sub_emb = nn.Embedding(sub_num_embeddings+1, dim, padding_idx=0)              
        self.sub_emb.weight.requires_grad = True            



        
    def forward(self, subs):
        

        subs_mask = (subs != 0)
        subs_mask = subs_mask.unsqueeze(-2).float()
        subs_mask = subs_mask.unsqueeze(-2)
        subs_mask = subs_mask.squeeze().unsqueeze(-1)


        sub_out = self.sub_emb(subs)   

        #print(torch.sum(subs_mask, dim=1).size())
        #print(torch.sum(subs_mask, dim=1))

        sub_mean = torch.sum(sub_out * subs_mask, dim=1) / torch.sum(subs_mask, dim=1)            


        return sub_mean



    def estimate_vector(self, word, sub2id_dict, device, ngram_min=3, ngram_max=5):
    
      word_list = [word, word]
      
      self.eval()
    
      list_of_subword_ind_list = []

        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
        
          extended_word = '<' + word + '>'
          subwords_list = self.extract_subwords(extended_word, ngram_min=ngram_min, ngram_max=ngram_max)
          #print('sub_list : '+str(subwords_list))

          subword_ind_list = []
          filtered_subword_list = []
          sub_pad_list = []
          for sub in subwords_list:
              if sub in sub2id_dict:
                  subword_ind_list.append(sub2id_dict[sub])
                  filtered_subword_list.append(sub)
      


          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)

          list_of_subword_ind_list.append(subword_ind_list)      
        

      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)


      bbbbb = self.forward(sub_tensor)
    
      #print(bbbbb.size())
      print(bbbbb[0].size())
      print(bbbbb[0][0].size())

 
      return bbbbb.cpu().detach().numpy()[0]
    





    def estimate_multiple_vectors(self, word_list, sub2id_dict, device, ngram_min=3, ngram_max=5):
    
      self.eval()
      
      one_item_in_list = False
      
      #handle one scenario
      if len(word_list) == 1:
          one_item_in_list = True
          word_list = [word_list[0], word_list[0]]
    
      list_of_subword_ind_list = []

        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
        
          extended_word = '<' + word + '>'
          subwords_list = self.extract_subwords(extended_word, ngram_min=ngram_min, ngram_max=ngram_max)

          subword_ind_list = []
          filtered_subword_list = []
          sub_pad_list = []
          for sub in subwords_list:
              if sub in sub2id_dict:
                  subword_ind_list.append(sub2id_dict[sub])
                  filtered_subword_list.append(sub)
      


          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)

          list_of_subword_ind_list.append(subword_ind_list)      
        
      print(list_of_subword_ind_list)
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)

      print(sub_tensor.size())
 
      
      bbbbb = self.forward(sub_tensor)


      if one_item_in_list == True:
          return [bbbbb.cpu().detach().numpy()[0]] # (2 x 400) -> (1 x 400)
      else:
          return bbbbb.cpu().detach().numpy()

    def extract_subwords(self, extended_word, ngram_min=3, ngram_max=5):
      extracted_list = []
      for n in range(ngram_min, ngram_max+1):
          for i in range(len(extended_word)):
              if i+n <= len(extended_word):  #make sure not out of bounds
                  subword = extended_word[i:i+n]
                  extracted_list.append(subword)
                  
      return extracted_list
      
    def mask_pad(self, x, pad = 0):
        "Create a mask to hide padding"
        return (x != pad).unsqueeze(-2).unsqueeze(-2)        
 