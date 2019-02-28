function perf = get_perform(gndts,gndprd,score)

perf.ham=loss_ham(gndts,gndprd);
perf.set = mean(any(gndts~=gndprd,2));
[perf.prec,perf.recal ] = prec_recal(gndts,gndprd);
[perf.f1,perf.accu ] = f1_accu(gndts,gndprd);
[perf.mac_prec, perf.mac_recal, perf.mac_f1] = macro_prf(gndts,gndprd);
[perf.mic_prec, perf.mic_recal, perf.mic_f1] = micro_prf(gndts,gndprd);

if nargin == 3
    Outputs = (score.*gndprd)'; test_target = gndts';
    
    perf.one_err = One_error(Outputs,test_target);
    perf.cvg = coverage(Outputs,test_target);
    perf.rank_loss = Ranking_loss(Outputs,test_target);
    perf.avg_prec = Average_precision(Outputs,test_target);
end

% Example based measures
function l = loss_ham(Y0,Y)
l = sum(Y(:)~=Y0(:))/length(Y0(:));

function [prec,recal] = prec_recal(Y0,Y)
prec = mean(sum(Y0.*Y,2)./(sum(Y,2)+eps));
recal = mean(sum(Y0.*Y,2)./(sum(Y0,2)+eps));

function [f1, accu] = f1_accu(Y0,Y)
f1 = 2*mean(sum(Y0.*Y,2)./(sum(Y,2) + sum(Y0,2) + eps));
accu = mean(sum(Y0.*Y,2)./(sum((Y0+Y)>0,2)+eps));

% Label based measures
function [mac_prec, mac_recal, mac_f1] = macro_prf(Y0,Y)
mac_prec = mean(sum(Y0.*Y,1)./(sum(Y,1)+eps));
mac_recal = mean(sum(Y0.*Y,1)./(sum(Y0,1)+eps));
mac_f1 = 2*mac_prec*mac_recal / (mac_prec + mac_recal);

function [mic_prec, mic_recal, mic_f1] = micro_prf(Y0,Y)
mic_prec = sum(sum(Y0.*Y))./(sum(sum(Y))+eps);
mic_recal = sum(sum(Y0.*Y))./(sum(sum(Y0))+eps);
mic_f1 = 2*mic_prec*mic_recal / (mic_prec + mic_recal);

% Ranking based measures (Copied from 'LIFT' toolbox)
function OneError=One_error(Outputs,test_target)
%Computing the one error
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

[num_class,num_instance]=size(Outputs);
temp_Outputs=[];
temp_test_target=[];
for i=1:num_instance
    temp=test_target(:,i);
    if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
        temp_Outputs=[temp_Outputs,Outputs(:,i)];
        temp_test_target=[temp_test_target,temp];
    end
end
Outputs=temp_Outputs;
test_target=temp_test_target;
[num_class,num_instance]=size(Outputs);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

oneerr=0;
for i=1:num_instance
    indicator=0;
    temp=Outputs(:,i);
    [maximum,index]=max(temp);
    for j=1:num_class
        if(temp(j)==maximum)
            if(ismember(j,Label{i,1}))
                indicator=1;
                break;
            end
        end
    end
    if(indicator==0)
        oneerr=oneerr+1;
    end
end
OneError=oneerr/num_instance;

function Coverage=coverage(Outputs,test_target)
%Computing the coverage
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

[num_class,num_instance]=size(Outputs);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

cover=0;
for i=1:num_instance
    temp=Outputs(:,i);
    [tempvalue,index]=sort(temp);
    temp_min=num_class+1;
    for m=1:Label_size(i)
        [tempvalue,loc]=ismember(Label{i,1}(m),index);
        if(loc<temp_min)
            temp_min=loc;
        end
    end
    cover=cover+(num_class-temp_min+1);
end
Coverage=(cover/num_instance)-1;

function RankingLoss=Ranking_loss(Outputs,test_target)
%Computing the hamming loss
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

[num_class,num_instance]=size(Outputs);
temp_Outputs=[];
temp_test_target=[];
for i=1:num_instance
    temp=test_target(:,i);
    if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
        temp_Outputs=[temp_Outputs,Outputs(:,i)];
        temp_test_target=[temp_test_target,temp];
    end
end
Outputs=temp_Outputs;
test_target=temp_test_target;
[num_class,num_instance]=size(Outputs);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

rankloss=0;
for i=1:num_instance
    temp=0;
    for m=1:Label_size(i)
        for n=1:(num_class-Label_size(i))
            if(Outputs(Label{i,1}(m),i)<=Outputs(not_Label{i,1}(n),i))
                temp=temp+1;
            end
        end
    end
    rl_binary(i)=temp/(m*n);
    rankloss=rankloss+temp/(m*n);
end
RankingLoss=rankloss/num_instance;

function Average_Precision=Average_precision(Outputs,test_target)
%Computing the average precision
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

[num_class,num_instance]=size(Outputs);
temp_Outputs=[];
temp_test_target=[];
for i=1:num_instance
    temp=test_target(:,i);
    if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
        temp_Outputs=[temp_Outputs,Outputs(:,i)];
        temp_test_target=[temp_test_target,temp];
    end
end
Outputs=temp_Outputs;
test_target=temp_test_target;
[num_class,num_instance]=size(Outputs);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

aveprec=0;
for i=1:num_instance
    temp=Outputs(:,i);
    [tempvalue,index]=sort(temp);
    indicator=zeros(1,num_class);
    for m=1:Label_size(i)
        [tempvalue,loc]=ismember(Label{i,1}(m),index);
        indicator(1,loc)=1;
    end
    summary=0;
    for m=1:Label_size(i)
        [tempvalue,loc]=ismember(Label{i,1}(m),index);
        summary=summary+sum(indicator(loc:num_class))/(num_class-loc+1);
    end
    ap_binary(i)=summary/Label_size(i);
    aveprec=aveprec+summary/Label_size(i);
end
Average_Precision=aveprec/num_instance;