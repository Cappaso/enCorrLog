function [data_train,data_test] = FeatNormalize(data_train,data_test,flag)

switch flag
    case 'scaling'
        % disp('Do normalization to [0,1].');
        minTrainData = min(data_train(:));
        rngTrainData = max(data_train(:)) - min(data_train(:));
        tmp_data = bsxfun(@minus,data_train,minTrainData);
        data_train = bsxfun(@times,tmp_data,1./rngTrainData);
        tmp_data = bsxfun(@minus,data_test,minTrainData);
        data_test = bsxfun(@times,tmp_data,1./rngTrainData);
    case 'whitening'
        % disp('Do whitening to zero mean and unit variance.'); % Fixed big issue on NaN.
        meanTrainData = mean(data_train); % average feature on train set.
        tmp_data = bsxfun(@minus,data_train,meanTrainData);
        tmp_std = std(tmp_data); tmp_std(tmp_std==0) = 1; % Important: avoid zero denominator.
        data_train = bsxfun(@times,tmp_data,1./tmp_std);
        tmp_data = bsxfun(@minus,data_test,meanTrainData);
        tmp_std = std(tmp_data); tmp_std(tmp_std==0) = 1;
        data_test = bsxfun(@times,tmp_data,1./tmp_std);
    otherwise
        disp('No normalization or whitening.');
end