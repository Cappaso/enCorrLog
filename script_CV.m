close all; clear;
% experiment script for cross-validation case

numExp = 5; % 5-folds cross-validation

% Select dataset, feature, MLC method,
dataNameAll = {'Scene','LabelMe','PASCAL07','PASCAL12'};
featNameAll = {'mulanFeat','phow','cnn'};
methodNameAll = {'ILRs','IBLR','MLkNN','CC','MMOC',...
	'LIFT','PLEM','CGM','CorrLog','enCorrLog'};

tmp = [];
for glb_di = 1
dataName = dataNameAll{glb_di}; % select dataset
for glb_fi = 1 % select feature
featName = featNameAll{glb_fi};
for glb_mi = 9 % global method index
methodName = methodNameAll{glb_mi};

% == PreSetup dataset info
load conf_Scene.mat;

% == Setting flags
flag.splitFile = sprintf('%s%s_CV_Splits.mat', conf.featDir, dataName);
flag.doSplit = false;
flag.doExtrFeat = false;

% == Random split train/test
trainIdx = cell(numExp,1);
testIdx = cell(numExp,1);
if ~exist(flag.splitFile,'file') || flag.doSplit
    % K-folds cross-validation
    allIdx = randperm( conf.numImages );
    cvPart = [0, round(conf.numImages * (1:numExp) / numExp)];
    for kk = 1:numExp
        testIdx{kk} = sort( allIdx(cvPart(kk)+1:cvPart(kk+1)) ); % 1 test fold
        trainIdx{kk} = setdiff( 1:conf.numImages, testIdx{kk} ); % (K-1) train folds
    end
	save(flag.splitFile, 'trainIdx', 'testIdx');
else
    fprintf('Load Split File: %s\n',flag.splitFile);
	load(flag.splitFile);
end

% == Define MLC output file name.
mlcOutFileName = sprintf('%s%s_%s_%s', conf.featDir, dataName, featName, methodName);

% == Main MLC procedure
disp(['Main: ', mlcOutFileName]);
label_pred = cell(numExp,1);
run_time = cell(numExp,1); % running time
for kk = 1:numExp
	trainIdx_kk = trainIdx{kk};
	testIdx_kk = testIdx{kk};
	
    % Extract GIST/CNN features in one time (split-independent).
    % Extract PHOW features according to train/test split (split-dependent).
    switch featName
        case 'phow'
            if ~exist(sprintf('%s%s_%s_%02d.mat', conf.featDir, dataName, featName, kk),'file') ...
                || flag.doExtrFeat
                ComputePhow( conf, trainIdx_kk, testIdx_kk, kk );
            end
            load(sprintf('%s%s_%s_%02d.mat', conf.featDir, dataName, featName, kk));
            % IMPORTANT CHECK while using PHOW, since the features are
            % dependent on training subset.
            if ~isequal( label_test, conf.labels(testIdx{kk},:) ); error('Inconsistent using PHOW...'); end
        case 'cnn'
            if ~exist(sprintf('%s%s_%s.mat', conf.featDir, dataName, featName),'file') ...
                || ( flag.doExtrFeat && (kk==1) )
                ComputeCnn( conf );
            end
            load(sprintf('%s%s_%s.mat', conf.featDir, dataName, featName));

            data_train = Xcnn(trainIdx_kk,:);
            data_test = Xcnn(testIdx_kk,:);
            label_train = conf.labels(trainIdx_kk,:);
            label_test = conf.labels(testIdx_kk,:);
        case 'mulanFeat'
            load(sprintf('%s%s_%s.mat', conf.featDir, dataName, featName));
            
            data_train = XmulanFeat(trainIdx_kk,:);
            data_test = XmulanFeat(testIdx_kk,:); clear XmulanFeat;
            label_train = conf.labels(trainIdx_kk,:);
            label_test = conf.labels(testIdx_kk,:);
        otherwise
            return;
    end
    
    % Feature normalization
    if strcmp(featName,'cnn') || strcmp(methodName,'CGM') % This is better for CGM
        disp('No normalization or whitening.');
    elseif strcmp(methodName,'PLEM') % This is better for PLEM
        disp('Do normalization to [0,1].');
        [data_train,data_test] = FeatNormalize(data_train,data_test,'scaling');
    else
        disp('Do whitening to zero mean and unit variance.'); % Fixed big issue on NaN.
        [data_train,data_test] = FeatNormalize(data_train,data_test,'whitening');
    end
    
    % Run MLC methods
    switch methodName
        case 'CorrLog'
            disp('Run CorrLog...');
            [ label_pred{kk}, run_time{kk} ] = Lite_CorrLog(data_train, label_train, data_test, label_test);
        case 'enCorrLog'
            disp('Run enCorrLog...');
            [ label_pred{kk}, run_time{kk} ] = Lite_enCorrLog(data_train, label_train, data_test, label_test);
        otherwise
            disp('No method choosed.');
    end
    
    % Save predictions & Record running time
    save([mlcOutFileName,'_pred.mat'],'label_pred');
    save([mlcOutFileName,'_time.mat'],'run_time');
    
end

% Evaluation measures
disp(['Evaluation: ', mlcOutFileName]);
load([mlcOutFileName '_pred.mat']); % load 'label_pred'
load([mlcOutFileName '_time.mat']); % load 'run_time'

perf = zeros(numExp,12);
for kk = 1:numExp;
    label_test = conf.labels(testIdx{kk},:);
    perf(kk,:) = cell2mat(struct2cell( get_perform(label_test,label_pred{kk}) ));
end
tmp = [tmp; mean(perf(:,[1 2 6 5 9 12]),1)]
save([mlcOutFileName,'.mat'],'perf');
disp(['Finished: ', mlcOutFileName]);

end
end
end