# SubAtt
Implementation of SubAtt.  For more information, see our paper : https://aclanthology.org/2023.findings-acl.221/

Quick Usage:

1)  Build a corpus using preprocess from https://github.com/timoschick/form-context-model.

2)  Split vocabulary into train and val dict, saved as train_val_dict.p .

3)  Train subword embeddings using train_torch_subwords.py

4)  Train SubAtt model using train_SubAtt_All_Versions.py

5)  Use trained model to predict new embeddings using its estimate_vector() method.

Examples:

python3 train_torch_subwords.py $SubwordOutputDir $TrainCorpus $WordEmbs 30 0.7 1 64 1e-2 0 25 0 100 $TRAIN_VAL_DICT

python3 train_SubAtt_All_Versions.py Pretrain_SubAtt $ModelOutputDir $TrainCorpus $SubwordOutputDir $WordEmbs 30 0.7 1 64 1e-4 0 25 0 100 8 fast 768 12

