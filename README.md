# enCorrLog

Demo code of our TIP 2016 paper "Correlated Logistic Model with Elastic Net Regularization for Multilabel Image Classification"

%% Code authors: Wei Bian and Qiang Li

%% Release time: Nov. 11th, 2016

%% Current version: CorrLog_v1

1. Run the Main code.

Note the difference between 'script_CV.m' and 'script_TT.m',

the first corresponds to cross-validation based experiments,

the second corresponds to train/test based experiments.

The datasets and results are in 'data/' folder.

2. Datasets and Feature normalization.

We only used MULANscene in this demo. For other datasets, please follow the guidance in our TIP paper.

For MULANscene and XXXX-PHOW, better to use "whitening" normalization.

For XXXX-CNN, better to apply no normalization.

3. Two methods are implemented.

CorrLog.m, corresponds to the previous model using L2 regularization.

enCorrLog.m, corresponds the updated model using elastic net regularization.

%% Reference noticement:

If you have used the code, please cite both of the two papers:

[1] Qiang Li, Bo Xie, Jane You, Wei Bian, and Dacheng Tao,

"Correlated Logistic Model with Elastic Net Regularization for Multilabel Image Classification,"

IEEE Trans. on Image Processing (T-IP), vol.25, no.8, pp.3801-3813, 2016.

[2] Wei Bian, Bo Xie, and Dacheng Tao,

"CorrLog: Correlated Logistic Models for Joint Prediction of Multiple Labels,"

in Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2012, pp.109¨C117.

%% Supporting information:

If any questions and comments, feel free to send your email to

Qiang Li (leetsiang.cloud@gmail.com)