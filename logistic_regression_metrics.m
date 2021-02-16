function [stats,ind_clean,OR,CI_low,CI_up,conf_matrix,correct_predic,sensitivity,specificity,precision]=logistic_regression_metrics(dep,ind);
%this function performs logistic regression using glmfit with one or
%multiple predictors. Removes outliers (abs(zscore(res))>3) and performs
%the regression again (if you do not want this feature, comment rows 45 to
%54).
%Returns stats (returned directly from the glmfit function), as well as
%additional metrics of interest (see OUTPUTS). All output metrics are in 
%accordance with results attained in Jasp software.

%INPUTS:
%- dep - dependent variable. Is a two-column matrix. Column 1 is a vector of
%0s (failures) and 1s (successes). Column 2 is constituted of only 1s.
%- ind - independent variables. Is a matrix with the same number of rows as
%dep, where each column is a variable of interest.

%OUTPUTS
%- stats - returned directly from the glmfit function
%- ind_clean - independent variables without the outliers (if you do not want
%this feature, it will just retrieve a copy of ind)
%- OR - odds ratio of intercept (row 1) and independent variables (in the
%inputed order)
%- CI_low - lower bound of the 95% confidence interval of intercept (row 1)
%and independent variables (in the inputed order)
%- CI_up - upper bound of the 95% confidence interval of intercept (row 1)
%and independent variables (in the inputed order)
%- conf_matrix - confusion matrix of real and predicted outcomes proportions: 
%(1,1) true negatives; (2,1) false negatives; (1,2) false positives; (2,2)
%true positives
%- correct_predic - proportion of correctly predicted outcomes (i.e. sum of
%true negatives and true positives)
%- sensitivity - proportion of successes correctly predicted
%- specificity - proportion of failures correctly predicted
%- precision - proportion of predicted successes that was correct

%Created by Madalena Esteves: madalena.curva.esteves@gmail.com

%Cite as: Madalena Esteves (2021). logistic_regression_metrics
%(https://github.com/madalenaesteves/shared_matlab/blob/main/logistic_regression_metrics.m).

%correction 15/02/2021 - correction of the database used for metrics 
%calculation (now without outliers)

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
elements=zeros(size(ind_clean,1),size(ind_clean,2));
for i=1:size(ind_clean,2);
    elements(:,i)=stats.beta(i+1)*ind_clean(:,i);
end

clear A B
prob=zeros(size(ind_clean,1),1);
for i=1:size(ind_clean,1);
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
