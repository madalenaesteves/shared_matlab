function [stats,ind_clean,OR,CI_low,CI_up,conf_matrix,correct_predic,sensitivity,specificity,precision]=logistic_regression_general(dep,ind);
%the independent variable is a matrix, where each row is one trial and
%each column is one varible for the model
%the dependent variable is a two-column matrix. The first column is the
%outcome of each trial (0s and 1s), the second is ones (I confirmed with jasp that this
%is the correct form of doing this)

%perform logistic regression
[b,dev,stats]=glmfit(ind,[dep],'binomial','logit');

%remove outliers
    ind_clean=ind;
    clear A B
    A=isnan(stats.resid);
    B=find(A==0);
    res_=NaN(length(stats.resid),1);
    res_(B,1)=abs(zscore(stats.resid(B,1)));
    
    outlier=find(res_>3);
    if isempty(outlier)==0;
        ind_clean(outlier,:)=NaN;
    end

%repeat logistic regression without the outliers
[b,dev,stats]=glmfit(ind_clean,[dep],'binomial','logit');

%calculate probability of prem in each trial
elements=zeros(size(ind,1),size(ind,2));
for i=1:size(ind,2);
    elements(:,i)=stats.beta(i+1)*ind(:,i);
end

clear A B
prob=zeros(size(ind,1),1);
for i=1:size(ind,1);
    A=isnan(elements(i,:));
    B=find(A==1);
    if isempty(B)==0;
        prob(i,1)=NaN;
    else
        prob(i,1)=exp(stats.beta(1)+sum(elements(i,:)))/(1+exp(stats.beta(1)+sum(elements(i,:))));
    end
end

%determine predicted response based on prob
predicted_resp=NaN(length(prob),1);
for i=1:length(prob);
    if prob(i)>0.5;
        predicted_resp(i)=1;
    else if prob(i)<0.5;
        predicted_resp(i)=0;
        end
    end
end

%calculate OR and confidence intervals
OR=exp(stats.beta);
CI_low=exp(stats.beta-stats.se*1.96);
CI_up=exp(stats.beta+stats.se*1.96);

%create confusion matrix
pred1_tru1=0;
pred1_tru0=0;
pred0_tru0=0;
pred0_tru1=0;
for i=1:length(dep);
    if predicted_resp(i)==1 && dep(i)==1;
        pred1_tru1=pred1_tru1+1;
    else if predicted_resp(i)==1 && dep(i)==0;
            pred1_tru0=pred1_tru0+1;
        else if predicted_resp(i)==0 && dep(i)==0;
                pred0_tru0=pred0_tru0+1;
            else if predicted_resp(i)==0 && dep(i)==1;
                    pred0_tru1=pred0_tru1+1;
                end
            end
        end
    end
end

total_resp=pred1_tru1+pred1_tru0+pred0_tru0+pred0_tru1;
conf_matrix(1,1)=pred0_tru0/total_resp;
conf_matrix(2,1)=pred0_tru1/total_resp;
conf_matrix(1,2)=pred1_tru0/total_resp;
conf_matrix(2,2)=pred1_tru1/total_resp;

%calculating proportion of correct predictions, sensitivity, specificity and precision
correct_predic=conf_matrix(1,1)+conf_matrix(2,2); %what is the proportion of correct predicitons?
sensitivity=pred1_tru1/(pred1_tru1+pred0_tru1); %from all positives, how many were correctly predicted?
specificity=pred0_tru0/(pred0_tru0+pred1_tru0); %from all negatives, how many were correctly predicted?
precision=pred1_tru1/(pred1_tru1+pred1_tru0); %when we say a given trial is positive, how many times was that correct?
end
