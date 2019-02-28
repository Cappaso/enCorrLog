function [ label_pred, run_time, perf ] = Lite_enCorrLog(data_train, label_train, data_test, label_test)
%   label_train:    nSamp * nLab
%   data_train:     nSamp * nFeat

rho1 = 0.001; rho2 = 0.001;

featr = [data_train*1 ones(size(data_train,1), 1)];
feats = [data_test*1 ones(size(data_test,1), 1)];

% training
star_t = tic;
[B,A] = enCorrLog(featr,label_train,rho1,rho2);
run_time.train = toc(star_t);

% prediction
star_t = tic;
label_pred = Prd_CorrLog(feats,B,A);
run_time.test = toc(star_t);

% Calculate performance if required
if nargout == 3
    perf = get_perform(label_test,label_pred);
end