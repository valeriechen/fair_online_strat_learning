Accompanying code for paper: "Learning Strategy-Aware Linear Classifiers" by Yiling Chen, Yang Liu, and Chara Podimata, which is accepted at NeurIPS2020. The paper is also publicly available here: https://arxiv.org/abs/1911.04004.


Folder cont_code/ contains the implementation for the continuous variant, and folder discr_code/ contains the implementation for the discretized variant.

It is run using python 2.7. and it has the following required packages: 

*** cont_code/ ***

-- numpy

-- cvxpy 

-- polytope

-- multiprocessing 

-- sklearn

-- logging

-- os

*** extra packages required for discr_code/ ***

-- networkx

-- sys



File that controls the distribution of points: params.py.

File that controls the power of the agent (i.e., \delta): runner-dp.py

In order to run the code: python runner-dp.py.

Please see individual .py files for explanations regarding the functionality of each file.





