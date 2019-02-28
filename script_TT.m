close all; clear;
% experiment script for train/test case

data_name = 'Emotions';
data_path = ['./data/' data_name];
load(data_path)

% data, y={0,1}
[data_train,data_test] = FeatNormalize(data_train,data_test,'whitening');
featr = [data_train, ones(size(data_train,1),1)];
gndtr = label_train;
feats = [data_test, ones(size(data_test,1),1)];
gndts = label_test;

% param tuning
rho1 = logspace(-4,0,5); %%% parameter
rho2 = logspace(-4,0,5); %%% parameter

perf_ham = []; perf_set = []; perf_mac_f1 = []; perf_mic_f1 = []; perf_f1 = [];  perf_accu = [];
for iTune1 = 1:length(rho1)
    for iTune2 = 1:length(rho2)
        % training
        [B,A] = enCorrLog(featr,gndtr,rho1(iTune1),rho2(iTune2));
        
        % prediction
        Yprd = Prd_CorrLog(feats,B,A);
        
        % evaluation
        tmp_perf = get_perform(gndts,Yprd);
        
        % record average cross-validatiaon results
        perf_ham(iTune1,iTune2) = tmp_perf.ham;
        perf_set(iTune1,iTune2) = tmp_perf.set;
        perf_accu(iTune1,iTune2) = tmp_perf.accu;
        perf_f1(iTune1,iTune2) = tmp_perf.f1;
        perf_mac_f1(iTune1,iTune2) = tmp_perf.mac_f1;
        perf_mic_f1(iTune1,iTune2) = tmp_perf.mic_f1;
    end
end


% visualize results
figure;imagesc(perf_ham);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['Hamming Loss Aug Logi ' data_name])

figure;imagesc(perf_set);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['Zero-One Loss Aug Logi ' data_name])

figure;imagesc(perf_accu);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['Accuracy Aug Logi ' data_name])

figure;imagesc(perf_f1);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['F1 Aug Logi ' data_name])

figure;imagesc(perf_mac_f1);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['Macro-F1 Aug Logi ' data_name])

figure;imagesc(perf_mic_f1);colorbar;
set(gca,'XTick',1:length(rho2));set(gca,'YTick',1:length(rho1));
set(gca,'XTickLabel',rho2);set(gca,'YTickLabel',rho1);
xlabel('rho2');ylabel('rho1');
title(['Micro-F1 Aug Logi ' data_name])

[a,b] = find(perf_set == min(perf_set(:))); a=a(1);b=b(1);
perf_t.ham = perf_ham(a,b);
perf_t.set = perf_set(a,b);
perf_t.accu = perf_accu(a,b);
perf_t.f1 = perf_f1(a,b);
perf_t.mac_f1 = perf_mac_f1(a,b);
perf_t.mic_f1 = perf_mic_f1(a,b);
perf_t

% save([data_path '_result_enCorrlog1'])