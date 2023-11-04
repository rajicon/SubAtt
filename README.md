# SubAtt
Implementation of SubAtt.  For more information, see our paper : https://aclanthology.org/2023.findings-acl.221/

Quick Usage:

1)  Build a corpus using preprocess from https://github.com/timoschick/form-context-model.

2)  Train subword embeddings using train_sub_embedding.py

3)  Train SubAtt model using train_SubAtt_All_Versions.py

4)  Use trained model to predict new embeddings using its estimate_vector() method.
