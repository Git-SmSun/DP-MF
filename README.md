# DP-MF
Affirm that our project is based on the open source recommender system LibMF.
About how to use LibMF, refer to https://www.csie.ntu.edu.tw/~cjlin/libmf/.

We have written our proposed method in the mf.cpp file, if you want to use our dynamic pruning method, you should：
  1. In function SolverBase::calc_z(), uncomment the pruning part and comment out the no pruning part.
  2. In function fpsg_core(), uncomment the comment part.
  3. In function MFSolver::sg_update(), uncomment the pruning part and comment out the no pruning part.

In addition, we provide some parameter settings：
  1. In function init_model(), we provide uniform and normal distributions to initialise the feature matrices:
     If you want to use normal distribution to initialize the feature marices P and Q，comment out Equation (1) and uncomment Equation (2). If you want to use uniform distribution，comment out Equation (2) and uncomment Equation (1).
     *ptr = (mf_float)(distribution(generator) * scale); (1)
     *ptr = (mf_float)(dist(generator)); (2)
  2. We provide the Adam and Adadelta methods . 
  
