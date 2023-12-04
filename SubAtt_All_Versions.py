#This is a version of Crossword_Contr_Lin where 
#we have model parameters for NoCross, LinTrans (true for now)
#CrossOnly, Cross + Contrastive Loss

#this model will take in each uncontextualized 
#bert layer embeddings, and estimate OOV at each layer
#and then aggregate 

#for now, eval this model, compare to AM and only layer 0 of this model
#compare to WLNLaMPro test: https://github.com/timoschick/am-for-bert

#this model uses a similar model to attentive mimicking, but uses a transformer to aggregate context (probably should also do SubAtt)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import numpy as np
import gensim



from HiCE_Transformer_Methods import EncoderLayer, LayerNorm, MultiHeadedAttention, MultiHeadedAttention, SublayerConnection, PositionalEncoding, PositionalAttention, CharCNN, CrossEncoderLayer, get_qkv_transformed_data



#https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

class SubAtt_All_Versions(nn.Module):
    def __init__(self, data, sub_emb_model, use_sub_attention, dim=400, nhead=10, num_layer=8, use_pos=True, fast=False):
        super(SubAtt_All_Versions, self).__init__()
        
        self.fast = fast
        self.data = data
        self.sub_emb_model = sub_emb_model
        
        self.n_hid   = dim
        self.n_head = nhead
        
        self.num_layer = int(num_layer)
        
        self.use_sub_attention = use_sub_attention
        
        if self.num_layer != 8:
            print('Num_Layer not correct, figure out'-9)
        
        word_num_embs = len(data.word_embeddings.wv.vocab)
        word_embs = np.zeros((word_num_embs+1, self.n_hid)) #include 0 index for masking
        for word in data.word_embeddings.wv.vocab:
            chosen_ind = data.word2id_dict[word]
            word_embs[chosen_ind] = data.word_embeddings.wv[word]   
            
        word_embs = torch.FloatTensor(word_embs)            
        self.emb = nn.Embedding.from_pretrained(word_embs, padding_idx=0)              
        self.emb.weight.requires_grad = False          
        
        sub_num_embeddings = len(data.sub2id_dict)
        
        if sub_emb_model != None:
            sub_embs = np.zeros((sub_num_embeddings+1, self.n_hid)) #include 0 index for masking
            for subword in sub_emb_model.wv.vocab:
                chosen_ind = data.sub2id_dict[subword]
                sub_embs[chosen_ind] = sub_emb_model.wv[subword]
            
            sub_embs = torch.FloatTensor(sub_embs)            
            self.sub_emb = nn.Embedding.from_pretrained(sub_embs, padding_idx=0)              
            self.sub_emb.weight.requires_grad = False            

        else:
            self.sub_emb = nn.Embedding(sub_num_embeddings+1, dim, padding_idx=0)              
            self.sub_emb.weight.requires_grad = True     


          
        self.alpha_weights_final_combiner = nn.Linear(self.n_hid *2, 1)


        
        self.use_pos = use_pos
 
        self.context_encoder   = nn.ModuleList([EncoderLayer(self.n_head, self.n_hid) for _ in range(2)])

            
        self.context_aggegator = nn.ModuleList([EncoderLayer(self.n_head, self.n_hid) for _ in range(self.num_layer)])
        self.sub_aggegator = nn.ModuleList([EncoderLayer(self.n_head, self.n_hid) for _ in range(self.num_layer)])

            
        self.pos_att = PositionalAttention(50)
        self.pos_enc = PositionalEncoding(self.n_hid)


        
    def forward(self, subs, contexts):
        
        masks = self.mask_pad(contexts, 0).transpose(0,1)
        
        if self.use_pos == True:
            x = self.pos_enc(self.pos_att(self.emb(contexts))).transpose(0,1)
        else:
            x = self.emb(contexts).transpose(0,1)

                

        if self.fast == True:

            x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1))  # (samples * timesteps, input_size)
            mask_reshape = masks.contiguous().view(-1, masks.size(-3), masks.size(-2), masks.size(-1))  # (samples * timesteps, input_size)

            for layer in self.context_encoder:
                x_reshape = layer(x_reshape, mask_reshape)
            mask_value = mask_reshape.squeeze().unsqueeze(-1).float()

            div_val = torch.sum(mask_value, dim=1)
            div_val[div_val==0] = 1 #replace div_vals of 0 with 1, to avoid divide by 0 error.  these should be 0 vecs anyway        
            res = torch.sum(x_reshape * mask_value, dim=1) / div_val        
            res = res.view(-1, x.size(1), res.size(-1))  # (timesteps, samples, output_size)
            res = res.transpose(0,1)
        
        
        else:
            res = []
            for xi, mask in zip(x, masks):
        
                #print('xi '+str(xi.size()))  
                #print('mask '+str(mask.size()))  
        
                for layer in self.context_encoder:
                    xi = layer(xi, mask)
                mask_value = mask.squeeze().unsqueeze(-1).float()

                #print('xi '+str(xi.size()))  
                #print('mask_value '+str(mask_value.size())) 
            
                #print('--')

                div_val = torch.sum(mask_value, dim=1)
                div_val[div_val==0] = 1 #replace div_vals of 0 with 1, to avoid divide by 0 error.  these should be 0 vecs anyway

            
                res += [torch.sum(xi * mask_value, dim=1) / div_val]
        

            
            #  K * B * n_hid  -> B * K * n_hid
            res = torch.stack(res).transpose(0,1)
        

        
        sent_masks = masks.transpose(0, 1) #masks torch.Size([32, 64, 1, 1, 50])
        sent_masks = sent_masks.transpose(1,3)
        sent_masks = torch.sum(sent_masks, dim=-1)
        sent_masks[sent_masks!=0] = 1
        
        

        for layer in self.context_aggegator:
            res = layer(res, sent_masks)
            



        subs_mask = (subs != 0)
        subs_mask = subs_mask.unsqueeze(-2).float()
        subs_mask = subs_mask.unsqueeze(-2)

        sub_out = self.sub_emb(subs)   

        if self.use_sub_attention:
            for layer in self.sub_aggegator:
                sub_out = layer(sub_out, subs_mask)        


        sent_masks = sent_masks.squeeze().unsqueeze(-1)
        div_val2 = torch.sum(sent_masks, dim=1)
        subs_mask = subs_mask.squeeze().unsqueeze(-1)
    
    
        sub_mean = torch.sum(sub_out * subs_mask, dim=1) / torch.sum(subs_mask, dim=1)            

        contexts_mean = torch.sum(res * sent_masks, dim=1) / div_val2


        
        
        a = torch.cat([sub_mean, contexts_mean], dim=1)
        alpha = self.alpha_weights_final_combiner(a)
        alpha = torch.sigmoid(alpha)
        final_out = (alpha * sub_mean) + ((1-alpha) * contexts_mean)

        return final_out


    def forward_get_sub_vs_ctx_scores(self, subs, contexts):
        
        masks = self.mask_pad(contexts, 0).transpose(0,1)
        
        if self.use_pos == True:
            x = self.pos_enc(self.pos_att(self.emb(contexts))).transpose(0,1)
        else:
            x = self.emb(contexts).transpose(0,1)

                

        if self.fast == True:

            x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1))  # (samples * timesteps, input_size)
            mask_reshape = masks.contiguous().view(-1, masks.size(-3), masks.size(-2), masks.size(-1))  # (samples * timesteps, input_size)

            for layer in self.context_encoder:
                x_reshape = layer(x_reshape, mask_reshape)
            mask_value = mask_reshape.squeeze().unsqueeze(-1).float()

            div_val = torch.sum(mask_value, dim=1)
            div_val[div_val==0] = 1 #replace div_vals of 0 with 1, to avoid divide by 0 error.  these should be 0 vecs anyway        
            res = torch.sum(x_reshape * mask_value, dim=1) / div_val        
            res = res.view(-1, x.size(1), res.size(-1))  # (timesteps, samples, output_size)
            res = res.transpose(0,1)
        
        
        else:
            res = []
            for xi, mask in zip(x, masks):
        
                #print('xi '+str(xi.size()))  
                #print('mask '+str(mask.size()))  
        
                for layer in self.context_encoder:
                    xi = layer(xi, mask)
                mask_value = mask.squeeze().unsqueeze(-1).float()

                #print('xi '+str(xi.size()))  
                #print('mask_value '+str(mask_value.size())) 
            
                #print('--')

                div_val = torch.sum(mask_value, dim=1)
                div_val[div_val==0] = 1 #replace div_vals of 0 with 1, to avoid divide by 0 error.  these should be 0 vecs anyway

            
                res += [torch.sum(xi * mask_value, dim=1) / div_val]
        

            
            #  K * B * n_hid  -> B * K * n_hid
            res = torch.stack(res).transpose(0,1)
        

        
        sent_masks = masks.transpose(0, 1) #masks torch.Size([32, 64, 1, 1, 50])
        sent_masks = sent_masks.transpose(1,3)
        sent_masks = torch.sum(sent_masks, dim=-1)
        sent_masks[sent_masks!=0] = 1
        
        

        for layer in self.context_aggegator:
            res = layer(res, sent_masks)
            



        subs_mask = (subs != 0)
        subs_mask = subs_mask.unsqueeze(-2).float()
        subs_mask = subs_mask.unsqueeze(-2)

        sub_out = self.sub_emb(subs)   

        if self.use_sub_attention:
            for layer in self.sub_aggegator:
                sub_out = layer(sub_out, subs_mask)        


        sent_masks = sent_masks.squeeze().unsqueeze(-1)
        div_val2 = torch.sum(sent_masks, dim=1)
        subs_mask = subs_mask.squeeze().unsqueeze(-1)
    
    
        sub_mean = torch.sum(sub_out * subs_mask, dim=1) / torch.sum(subs_mask, dim=1)            

        contexts_mean = torch.sum(res * sent_masks, dim=1) / div_val2


        
        
        a = torch.cat([sub_mean, contexts_mean], dim=1)
        alpha = self.alpha_weights_final_combiner(a)
        alpha = torch.sigmoid(alpha)
        final_out = (alpha * sub_mean) + ((1-alpha) * contexts_mean)

        return final_out, alpha


    def forward_before_cross(self, subs, contexts):
        
        
        print('implement this'-2)


        return sub_out, subs_mask, res, sent_masks
        



    def forward_ctx(self, subs, contexts):
        

        print('implement this'-2)

          

        
        return final_cc_out


    def forward_sub(self, subs, contexts):
        
        print('implement this'-2)


        
        return final_sub_out


    def forward_get_cross_qkv(self, subs, contexts):
        
        print('implement this'-2)



        
        return sc_q, sc_k, sc_v, cs_q, cs_k, cs_v, sent_masks, subs_mask


 
    def estimate_vector(self, word, contexts, word2id_dict, sub2id_dict, device, ngram_min=3, ngram_max=5, max_words_per_context = 50):
    
      word_list = [word, word]
      contexts_list = [contexts, contexts]
      
      self.eval()
    
      list_of_subword_ind_list = []
      list_of_context_inds_list  = []
      
      
      max_amt_of_cons = 2; #one causes problems with squeeze
      for con in contexts_list:
          if len(con) > max_amt_of_cons :
             max_amt_of_cons = len(con)
        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
          contexts = contexts_list[my_ind]
        
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

              #print(c_list)
              #print(len(word2id_dict))
              #print('subatt'-2)
              while len(c_list) < max_words_per_context:
                  c_list.append(0)

              if len(c_list) > max_words_per_context:
                  c_list = c_list[0:max_words_per_context]
          

              context_inds_list.append(c_list)
      
          
          while len(context_inds_list) < max_amt_of_cons:
            context_inds_list.append(np.zeros(max_words_per_context))
      

          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)
          context_inds_list = np.array(context_inds_list)

          list_of_subword_ind_list.append(subword_ind_list)      
          list_of_context_inds_list.append(context_inds_list)
        
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))

      #bbbbb = self.model.predict([np.array(list_of_subword_ind_list), np.array(list_of_context_inds_list)])
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)
      cc_tensor = torch.LongTensor(list_of_context_inds_list).to(device)

      #print(sub_tensor.size())
      #print(cc_tensor.size())

      bbbbb = self.forward(sub_tensor, cc_tensor)
    
      #print(bbbbb.size())
      #print(bbbbb[0].size())
      #print(bbbbb[0][0].size())


      return bbbbb.cpu().detach().numpy()[0]
    



 
    def estimate_vector_alpha_scores(self, word, contexts, word2id_dict, sub2id_dict, device, ngram_min=3, ngram_max=5, max_words_per_context = 50):
    
      word_list = [word, word]
      contexts_list = [contexts, contexts]
      
      self.eval()
    
      list_of_subword_ind_list = []
      list_of_context_inds_list  = []
      
      
      max_amt_of_cons = 2; #one causes problems with squeeze
      for con in contexts_list:
          if len(con) > max_amt_of_cons :
             max_amt_of_cons = len(con)
        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
          contexts = contexts_list[my_ind]
        
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

              #print(c_list)
              #print(len(word2id_dict))
              #print('subatt'-2)
              while len(c_list) < max_words_per_context:
                  c_list.append(0)

              if len(c_list) > max_words_per_context:
                  c_list = c_list[0:max_words_per_context]
          

              context_inds_list.append(c_list)
      
          
          while len(context_inds_list) < max_amt_of_cons:
            context_inds_list.append(np.zeros(max_words_per_context))
      

          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)
          context_inds_list = np.array(context_inds_list)

          list_of_subword_ind_list.append(subword_ind_list)      
          list_of_context_inds_list.append(context_inds_list)
        
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))

      #bbbbb = self.model.predict([np.array(list_of_subword_ind_list), np.array(list_of_context_inds_list)])
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)
      cc_tensor = torch.LongTensor(list_of_context_inds_list).to(device)

      #print(sub_tensor.size())
      #print(cc_tensor.size())

      bbbbb, alpha = self.forward_get_sub_vs_ctx_scores(sub_tensor, cc_tensor)
    
      #print(bbbbb.size())
      #print(bbbbb[0].size())
      #print(bbbbb[0][0].size())


      return bbbbb.cpu().detach().numpy()[0], alpha.cpu().detach().numpy()[0]



    def estimate_vector_sub(self, word, contexts, word2id_dict, sub2id_dict, device, ngram_min=3, ngram_max=5, max_words_per_context = 50):
    
      word_list = [word, word]
      contexts_list = [contexts, contexts]
      
      self.eval()
    
      list_of_subword_ind_list = []
      list_of_context_inds_list  = []
      
      
      max_amt_of_cons = 2; #one causes problems with squeeze
      for con in contexts_list:
          if len(con) > max_amt_of_cons :
             max_amt_of_cons = len(con)
        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
          contexts = contexts_list[my_ind]
        
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
      
          
          while len(context_inds_list) < max_amt_of_cons:
            context_inds_list.append(np.zeros(max_words_per_context))
      

          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)
          context_inds_list = np.array(context_inds_list)

          list_of_subword_ind_list.append(subword_ind_list)      
          list_of_context_inds_list.append(context_inds_list)
        
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))

      #bbbbb = self.model.predict([np.array(list_of_subword_ind_list), np.array(list_of_context_inds_list)])
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)
      cc_tensor = torch.LongTensor(list_of_context_inds_list).to(device)

      #print(sub_tensor.size())
      #print(cc_tensor.size())

      bbbbb = self.forward_sub(sub_tensor, cc_tensor)
    
      #print(bbbbb.size())
      #print(bbbbb[0].size())
      #print(bbbbb[0][0].size())

      return bbbbb[0].cpu().detach().numpy()
      
    def estimate_vector_ctx(self, word, contexts, word2id_dict, sub2id_dict, device, ngram_min=3, ngram_max=5, max_words_per_context = 50):
    
      word_list = [word, word]
      contexts_list = [contexts, contexts]
      
      self.eval()
    
      list_of_subword_ind_list = []
      list_of_context_inds_list  = []
      
      
      max_amt_of_cons = 2; #one causes problems with squeeze
      for con in contexts_list:
          if len(con) > max_amt_of_cons :
             max_amt_of_cons = len(con)
        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
          contexts = contexts_list[my_ind]
        
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
      
          
          while len(context_inds_list) < max_amt_of_cons:
            context_inds_list.append(np.zeros(max_words_per_context))
      

          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)
          context_inds_list = np.array(context_inds_list)

          list_of_subword_ind_list.append(subword_ind_list)      
          list_of_context_inds_list.append(context_inds_list)
        
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))

      #bbbbb = self.model.predict([np.array(list_of_subword_ind_list), np.array(list_of_context_inds_list)])
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)
      cc_tensor = torch.LongTensor(list_of_context_inds_list).to(device)

      #print(sub_tensor.size())
      #print(cc_tensor.size())

      bbbbb = self.forward_ctx(sub_tensor, cc_tensor)
    
      #print(bbbbb.size())
      #print(bbbbb[0].size())
      #print(bbbbb[0][0].size())

      return bbbbb[0].cpu().detach().numpy()

        
    def estimate_multiple_vectors(self, word_list, contexts_list, word2id_dict, sub2id_dict, device, ngram_min=3, ngram_max=5, max_words_per_context = 50, fast=True):
    
      self.eval()
      
      one_item_in_list = False
      
      #handle one scenario
      if len(word_list) == 1:
          one_item_in_list = True
          word_list = [word_list[0], word_list[0]]
          contexts_list = [contexts_list[0], contexts_list[0]]
    
      list_of_subword_ind_list = []
      list_of_context_inds_list  = []
      
      
      max_amt_of_cons = 2;
      for con in contexts_list:
          if len(con) > max_amt_of_cons :
             max_amt_of_cons = len(con)
        
      max_amt_of_subs = 0;
      for sss in word_list:
           extended_sss = '<' + sss + '>'
           sss_len = len(self.extract_subwords(extended_sss, ngram_min=ngram_min, ngram_max=ngram_max))
           if sss_len > max_amt_of_subs:
               max_amt_of_subs = sss_len
      

      
      for my_ind in range(len(word_list)):
          word = word_list[my_ind]
          contexts = contexts_list[my_ind]
        
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
      
          
          while len(context_inds_list) < max_amt_of_cons:
            context_inds_list.append(np.zeros(max_words_per_context))
      

          while len(subword_ind_list) < max_amt_of_subs:
            subword_ind_list.append(0)

          subword_ind_list = np.array(subword_ind_list)
          context_inds_list = np.array(context_inds_list)

          list_of_subword_ind_list.append(subword_ind_list)      
          list_of_context_inds_list.append(context_inds_list)
        
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))

      #bbbbb = self.model.predict([np.array(list_of_subword_ind_list), np.array(list_of_context_inds_list)])
      #print(np.shape(list_of_subword_ind_list))
      #print(np.shape(list_of_context_inds_list))
      sub_tensor = torch.LongTensor(list_of_subword_ind_list).to(device)
      cc_tensor = torch.LongTensor(list_of_context_inds_list).to(device)

      #print(sub_tensor.size())
      #print(cc_tensor.size())
      #print('--')
      
      bbbbb = self.forward(sub_tensor, cc_tensor)


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
 
