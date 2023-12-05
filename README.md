# SubAtt
Implementation of SubAtt.  For more information, see our paper : https://aclanthology.org/2023.findings-acl.221/

Quick Usage:

1)  Build a corpus using preprocess from https://github.com/timoschick/form-context-model.

2)  Call split_vocabulary_into_train_and_val.py on the preprocessed corpus folder to create a dict that splits training and validation words.

3)  Train subword embeddings using train_torch_subwords.py

4)  Train SubAtt model using train_SubAtt_All_Versions.py

5)  Use trained model to predict new embeddings using its estimate_multiple_vectors() method. (If only one estimate, put word and context in a list).

Some Examples:

python3 split_vocabulary_into_train_and_val.py $TrainCorpus  #Step 2, creates TRAIN_VAL_DICT file

python3 train_torch_subwords.py $SubwordOutputDir $TrainCorpus $WordEmbs 30 0.7 1 64 1e-2 0 25 0 100 $TRAIN_VAL_DICT #Step 3

python3 train_SubAtt_All_Versions.py Pretrain_SubAtt $ModelOutputDir $TrainCorpus $SubwordOutputDir $WordEmbs 30 0.7 1 64 1e-4 0 25 0 100 8 fast 768 12 #Step 4

