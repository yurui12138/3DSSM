## 3DSSM

#### Sentence Semantic Matching based on 3D CNN for Human-robot Language Interaction
Wenpeng Lu, Rui Yu, Shoujin Wang, Can Wang, Ping Jian, Heyan Huang  
The paper has been received by ACM Transaction on Internet Technology.

#### Prerequisites
python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0  

#### Example to run the codes
Run 3DSSM.py  
`python3 3DSSM.py`  

#### Dataset
We used two datasets: BQ & LCQMC.  
1. "The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.  
2. "LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.

### Note
Due to the differences between the two data sets, some parameters used by the model in the training of the two data sets are not exactly the consistent. Therefore, we provide two versions of the code.
