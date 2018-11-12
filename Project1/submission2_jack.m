%% I) LDA/QDA CLASSIFIER
clearvars; close all;

inputDataSet='trainSet.mat';
inputLabel='trainLabels.mat';

TrainDataSet=open(inputDataSet);
TrainDataSet=TrainDataSet.trainData;

Labels=open(inputLabel);
Labels=Labels.trainLabels;  % 1=error, 0=correct

FeatSubset=TrainDataSet(:, 1:10:end);
LabSubset=Labels(:, 1:10:end);


%% Specifying priortype = 'uniform' or 'empyrical' (default)

% prior probabilities (PriorType) can be set to:
% --> empirical: the class prior probabilities are the class relative
% frequencies in the label vector
% --> uniform: all class prior probabilities are equal to 1/K, where K is
% the number of the classes
% Prior probability: P(Ck)=sum_j(n_kj)/N

Methods={'linear' 'diaglinear' 'diagquadratic'};

PriorType_uni='uniform';
% --> !!! this correspond to the default setting (i.e. without specifying
% priortype
PriorType_emp='empirical'; 
% ratio to split the data in 2 groups (training and test subsets)
ratio=0.7;
classErrorRatios=[0.7 0.3];

% ERROR computed on the TEST SUBSET, test the 2 prior prob. distributions
classificationErrors_uni=zeros(length(Methods), 1);
classErrors_uni=zeros(length(Methods), 1);
classificationErrors_emp=zeros(length(Methods), 1);
classErrors_emp=zeros(length(Methods), 1);

for t=1:size(classificationErrors_uni, 1)
    % uniform prior
    [classificationErrors, classErrors] = errors(TrainDataSet, Labels, ...
        ratio, PriorType_uni, Methods{t}, classErrorRatios);
    classificationErrors_uni(t)=classificationErrors(2);
    classErrors_uni(t)=classErrors(2);
    
    % empirical prior
    [classificationErrors, classErrors] = errors(TrainDataSet, Labels, ...
        ratio, PriorType_emp, Methods{t}, classErrorRatios);
    classificationErrors_emp(t)=classificationErrors(2);
    classErrors_emp(t)=classErrors(2);
end
fprintf('\n--> (I) LDA/QDA CLASSIFIERS: \n');
[minClassificationError_uni, idx_min_uni]=min(classificationErrors_uni(:));
fprintf('Divide samples in a test and training subsets with a random permutation, build models of different kinds with the same permutation so that are comparable \n');
fprintf('\nUNIFORM PRIOR DISTRIBUTION \n');
fprintf('The minimal classification error is obtained with method: %-s and is %-10.4f \n', Methods{idx_min_uni}, minClassificationError_uni);
[minClassError_uni, idx_min_uni]=min(classErrors_uni(:));
fprintf('The minimal class error is obtained with method: %-s and is %-10.4f \n', Methods{idx_min_uni}, minClassError_uni);

[minClassificationError_emp, idx_min_emp]=min(classificationErrors_emp(:));
fprintf('\nEMPIRICAL PRIOR DISTRIBUTION \n');
fprintf('The minimal classification error is obtained with method: %-s and is %-10.4f \n', Methods{idx_min_emp}, minClassificationError_emp);
[minClassError_emp, idx_min_emp]=min(classErrors_emp(:));
fprintf('The minimal class error is obtained with method: %-s and is %-10.4f \n', Methods{idx_min_emp}, minClassError_emp);
% fprintf output with prior = "uniform": 
% % The minimal classification error is obtained with method: diagquadratic and is 0.5587     
% % The minimal class error is obtained with method: diagquadratic and is 0.7777   

%%%%%%%% OUTCOME %%%%%%%%
% --> based on the classification error we would choose the linear
% classifier. This is as expected due to the linearity of our data. This is
% confirmed by the class error. This outcomes strongly depends on the
% random permutation for the subsets creation. The majority of the
% permutations give the linear model as the best. However, sometimes the
% best model is the diaglinear or diagquadratic. To justify the choice of
% the linear model, see the next part (part II).
% --> by using an empyrical prior distribution we can generally observe 
% lower class and classification errors. This observation strongly depends
% on the random permutation used to divide the 2 subsets. But this is as
% expected. Our data are not equally distributed, we have ~70% of 0-labels 
% and ~30% of 1-labels. Therefore an empirical prior distribution better
% describe the nature of our data
% --> With both distributions, the best model is the linear one.
% --> In all the cases, the class error is higher than the classification
% error. The usage of the classification or the class errors depends on
% what we want to study. The classification error is more useful if the
% study is about the performances of the whole classifier. If the goal is
% not to study the general performances but to really investigate about
% each class classification, the class error is more suited. In fact, this
% takes into account the weight (size) of each class and reflects in a 
% more appropriate and accurate way the "individual class classification
% accuracy".
% If the class error weights are modified to [0.7 0.3], this error lowers.
% This is as expected because by this way the classes weights are better
% represented than the default [0.5 0.5] that supposes an equal partition
% of samples.
%%%%%%%%%%%%%%%%%%%%%%%%%

%% HANDS ON (TRISTAN)

% Check the help function of fitcdiscr() to find out how to specify prior probabilities. Then, based
% on the chosen classifers, repeat again by specifying uniform prior when calling fitcdiscr(). This
% time, look at both classifcation error and class error and compare the cases with and without
% uniform prior. What do you observe and how do you explain the values based on the prior. Also,
% regarding the classifcation error and the class error, would you argue that one is more useful than
% the other. Make some assumption if necessary, and choose one for the later part of this guide sheet. 

% ==> theoretical prior P(Ck) = # of sample in class Ck / # of sample tot
% ==> approximal prior P(Ck) = 'uniform' = 1 / # of class = 1/2
% ==> default prior P(Ck) = 'empirical' = The class prior probabilities are 
%                                         the class relative frequencies in 
%                                         TrainLabels.

% COMPARISION BETWEEN ERROR ( priortype = 'uniform' vs. priortype = 'empirical' (default value))
% - We observe that when priortype isn't precised, error probabilities are a
% little bit higher than those obtained when priortype = 'uniform'
%       o   priortype = 'uniform'  -->  min classification err = 0.5587 
%                                  -->  min class err = 0.7777
%       o   priortype = 'empirical'-->  min classification err = 0.5689
%                                  -->  min class err = 0.7933
% - Moreover, the classifier method with the lowest values of error are not
% the same: 
%       o   priortype = 'uniform'   -->  diagquadratic 
%       o   priortype = 'empirical' -->  diaglinear

% READ ME (QUESTION): We should tell why we obtain thoses results, personnally
% (Tristan), I think that empirical prior probability better correspond to 
% real value of prior, since they are calculated with the relative
% frequency in TrainLabels (== theoretical prior), but we obtain error
% bigger than those obtain with the approximation made with 'uniform' that
% prior where egal to 1/2. Any idea why??? 


%% II) TRAINING TESTING ERROR

Methods={'linear' 'diaglinear' 'diagquadratic'};

% Use empyrical since we observed a lower error compared to uniform
PriorType='empirical'; 
% ratio to split the data in 2 groups (training and test subsets). Here
% split into 2 equal groups
ratio=0.5;
classErrorRatios=[0.7 0.3];

% ERROR computed on the TEST SUBSET, test the 2 prior prob. distributions
% 2 columns, the first for train subset error, the second for the test
% subset error. Each column correspond to a Method
classificationErrors=zeros(length(Methods), 2);
classErrors=zeros(length(Methods), 2);

fprintf('\n--> (II) TRAINING AND TESTING ERRORS: \n');
for t=1:size(classificationErrors, 1)
    [classificationErrors(t, :), classErrors(t, :)] = errors(TrainDataSet, Labels, ...
        ratio, PriorType, Methods{t}, classErrorRatios);   
    fprintf('\nUsed prior probability: %-s with method: %-s \n', PriorType, Methods{t});
    fprintf('\t\t\t %-10s \t %-10s \n', 'Traing subset', 'Test subset');
    fprintf('Classification error: \t %-10.4f \t %-10.4f \n', classificationErrors(t, :));
    fprintf('Class error: \t\t %-10.4f \t %-10.4f \n', classErrors(t, :));
end

% [minClassificationError, idx_min]=min(classificationErrors, [], 1);
% [minClassError, idx_min]=min(classErrors, [], 1);

%%%%%%%% OUTCOME %%%%%%%%
% --> We can observe that the error of the model tested on the training
% subset is always lower. This is as expected! In fact, if the model is
% tested on the subset that was used to build it, the error would be lower
% since it is optimized for this subset of data. However, a perfect linear
% model cannot be built (unless very specific cases), therefore the error,
% even on the training subset is never zero. If another type of model would
% be used, this could be the case. Here the quadratic model cannot be used
% because the covariance matrix is not inversible. By the way, this is not
% always a good idea to have a model that tested on the training set gives
% a zero-error. This could in fact lead to overfitting: a perfect fit of
% training samples. This is a problem because even if a small data
% difference is present in an unseen dataset, because of the perfect
% optimization on a specific training data, this would lead to very large
% error: poor representation of unseen data.
% As before, the class error is slightly higher than the classification
% error. By modifying the class error weights with [0.7 0.3] (standard: 
% [0.5 0.5]), the class error lowers a little bit as explained in the
% previous section but still stays higher than the classification error.
% --> !!! The error on the training set can help us to choose the best
% model type. In fact, the model that most approximate the nature of our
% data would have the lower training subset errors. If the datas perfectly
% fit to a given model, the error on this subset would be zero. We can see
% it very well by running a classifier construction with multiple models
% type and by comparing the results. We can see that with the linear model,
% the training subset errors are almost zero. We therefore chose the LINEAR
% model. Our data follow a LINEAR distribution.
% --> The improvement (lowering) on the training error does not improve the
% testing error for the reason explained above. The testing subset is an
% "unseen" subset that is only used to test the model performances and NOT
% to build it or to optimize it. To optimize it a validation subset should
% be created.
% --> the quadratic classifier cannot be used because the covariance matrix
% is not invertible
% --> model complexity: a linear model is fast and easy to build because it
% has a low complexity. In many cases a linear model can be sufficient to
% estimate a classifier or to do a first rough data classification. But in
% many cases a more complex model is needed. More complex models imply an
% higher computation time and power but in SOME cases give better
% classifiers. This is not always the case. If the complexity is too high
% the risk of overfitting is present. Therefore, a compromise between
% complexity and accuracy must be found.
% --> since the permutation used to divide the 2 subsets is randomly
% generated, if the set1 is used to train and build the model and the set2
% is used to test it ot vice-versa, the variability of the performance
% should stay stable.
% --> with uniform prior probability we optain lower performances as
% describet in section I
%%%%%%%%%%%%%%%%%%%%%%%%%

%% Testing minError function

% classifErr_test=zeros(3, 1);
% classErr_test=zeros(3, 1);
% 
% classifErr_train=zeros(3, 1);
% classErr_train=zeros(3, 1);
% 
% ratio = 0.5;
% priortype = 'uniform';
% niter=10;
% 
% [classifErr_test, classifErr_train, classErr_test, classErr_train] = minErrors(ratio, priortype, niter) ;

%%Result with 1000 iters and ratio = 0.5 priortype='uniform': 
% % % test_set min classification error --> method = diagquadratic; prior = uniform; value = 0.5829     
% % % test_set min class error --> method = diagquadratic ; prior = uniform; value = 0.8113     
% % % train_set min classification error --> method = Linear; prior = uniform; value = 0.0180     
% % % train_set min class error --> method = Linear; prior = uniform; value = 0.0249  

%%Result with 1000 iters and ratio = 0.7 priortype='uniform'
% % % test_set min classification error --> method = Linear; prior = uniform; value = 0.5692     
% % % test_set min class error --> method = Linear ; prior = uniform; value = 0.7945     
% % % train_set min classification error --> method = Linear; prior = uniform; value = 0.0887     
% % % train_set min class error --> method = Linear; prior = uniform; value = 0.1231   

%%Result with 1000 iters and ratio = 0.5 priortype='empirical'
% % % test_set min classification error --> method = diagquadratic; prior = empirical; value = 0.5709     
% % % test_set min class error --> method = diagquadratic ; prior = empirical; value = 0.7951     
% % % train_set min classification error --> method = Linear; prior = empirical; value = 0.0151     
% % % train_set min class error --> method = Linear; prior = empirical; value = 0.0208 

%%Result with 1000 iters and ratio = 0.7 priortype='empirical'
% % test_set min classification error --> method = Linear; prior = empirical; value = 0.5247     
% % test_set min class error --> method = Linear ; prior = empirical; value = 0.7313     
% % train_set min classification error --> method = Linear; prior = empirical; value = 0.0717     
% % train_set min class error --> method = Linear; prior = empirical; value = 0.0995   

%% CROSS-VALIDATION FOR  PERFORMANCE ESTIMATION
close all;

% --> one of the disavantages of having too small training set could be a
% wrong or less-precise probability distribution extrapolation in the case
% of a probabilistic approach

% k-fold cross validation:
% --> An object of the cvpartition class defines a random partition on a
% set of data of a specified size.  This partition can be used to
% define test and training sets for validating a statistical model
% using cross-validation

% create random partitions for a k-fold validation
nbPartitions=10;
cp=cvpartition(Labels, 'kfold', nbPartitions);

PriorType='empirical';
Method={'linear'};
classErrorRatios=[0.5 0.5];

% 2 columns, the first for train subset error, the second for the test
% subset error. 
classificationErrors_all=zeros(nbPartitions, 2);
classErrors_all=zeros(nbPartitions, 2);
%% 

for i=1:nbPartitions
    train_subset=cp.training(i);
    train_subset_data=TrainDataSet(train_subset, :);
    train_labels=Labels(train_subset);
    
    test_subset=cp.test(i);
    test_subset_data=TrainDataSet(test_subset, :);
    test_labels=Labels(test_subset);
    
    % !!! !could reduce dimensionality!!! (NOT reduced anymore in errors_subsets
    % function) !!!!
    
    if (sum(train_subset)+sum(test_subset)) ~= numel(Labels)
        error('ERROR, verify how to use the cvpartition class, the sum of the train and test subset samples is < nb. total samples !!');
    end
    [classificationErrors_all(i, :), classErrors_all(i, :)] = ...
        errors_subsets(train_subset_data, test_subset_data, train_labels, test_labels,...
        PriorType, Method{1}, classErrorRatios);
end

std_classification=std(classificationErrors_all, 1);
std_class=std(classErrors_all, 1);

folds=[1:1:10];
k_fold_fig=figure; 
plot(folds, classificationErrors_all(:, 1)); hold on;
plot(folds, classificationErrors_all(:, 2));
plot(folds, classErrors_all(:, 1));
plot(folds, classErrors_all(:, 2));
aa=xlim;
line([aa(1) aa(2)], [std_classification(2) std_classification(2)], 'Color', 'm', 'Linestyle', '--');
line([aa(1) aa(2)], [std_class(2) std_class(2)], 'Color', 'c', 'Linestyle', '--');
legend({'Train subset classification error', 'Test subset classification error', ...
    'Train subset class error', 'Test subset class error', ...
    'STD test subset classification error', 'STD test subset class error'});
xlabel('k-folds'); ylabel('Error');
title('STANDARD k-fold validation');

print(k_fold_fig, '10-fold cross validation errors', '-dtiffn', '-r400');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% repeat cross-validation using repartition at each time

% 2 columns, the first for train subset error, the second for the test
% subset error. 
classificationErrors_all=zeros(nbPartitions, 2);
classErrors_all=zeros(nbPartitions, 2);

for i=1:nbPartitions
    cp_new=repartition(cp);
    train_subset=cp_new.training(i);
    train_subset_data=TrainDataSet(train_subset, :);
    train_labels=Labels(train_subset);
    
    test_subset=cp_new.test(i);
    test_subset_data=TrainDataSet(test_subset, :);
    test_labels=Labels(test_subset);
    
    % !!! !could reduce dimensionality!!! (NOT reduced anymore in errors_subsets
    % function) !!!!
    
    if (sum(train_subset)+sum(test_subset)) ~= numel(Labels)
        error('ERROR, verify how to use the cvpartition class, the sum of the train and test subset samples is < nb. total samples !!');
    end
    [classificationErrors_all(i, :), classErrors_all(i, :)] = ...
        errors_subsets(train_subset_data, test_subset_data, train_labels, test_labels,...
        PriorType, Method{1}, classErrorRatios);
end

std_classification=std(classificationErrors_all, 1);
std_class=std(classErrors_all, 1);

folds=[1:1:10];
k_fold_repartition_fig=figure; 
plot(folds, classificationErrors_all(:, 1)); hold on;
plot(folds, classificationErrors_all(:, 2));
plot(folds, classErrors_all(:, 1));
plot(folds, classErrors_all(:, 2));
aa=xlim;
line([aa(1) aa(2)], [std_classification(2) std_classification(2)], 'Color', 'm', 'Linestyle', '--');
line([aa(1) aa(2)], [std_class(2) std_class(2)], 'Color', 'c', 'Linestyle', '--');
legend({'Train subset classification error', 'Test subset classification error', ...
    'Train subset class error', 'Test subset class error', ...
    'STD test subset classification error', 'STD test subset class error'});
xlabel('k-folds'); ylabel('Error');
title('REPARTITION k-fold validation');

print(k_fold_repartition_fig, '10-fold cross validation errors _ REPARTITION', '-dtiffn', '-r400');


%%%%%%%% OUTCOME %%%%%%%%
% --> cvpartition returns 10 test subdivisions and 10 train subdivisions.
% One test partition of sike A is used for testing the model that is
% trained with the other [NbPartitions-1] train partitions of sizes 
% [B_1, ..., B_NbPartitions-1]. The partitions combinations are chosed to 
% have always sum(B)+A=N=number of samples. Moreover, Bi=round(N/NbPartitions)
% Both cvpartition(N, 'kfold', nbPartitions) and 
% cvpartition(Labels, 'kfold', nbPartitions) syntaxes gives the same
% result; With the same notation, each subclass will have rougly the same
% class proportion as in Labels vector. This notation is therefore
% suggested.
% --> repartition usage in k-fold validation allow to deefine a new random
% partition of the input samples. It means that samples are devided into 2
% classes following a repartition that is different from the initial
% predicted one. Repartitioning is useful for Monte-Carlo repetitions of
% cross-validation analysis. The randomnes is used in a Monte-Carlo
% simulation to solve a problem having a probabilistic interpretation. In
% fact, by the law of large numbers, "the samples generated by the
% randomness will be the ones that follow the right separation pattern".
% --> By constructing this classification system that vary depending on the
% randomness of some parameter, there are multiple advantages!! To optimize
% the outcome, an iterative simulation could in fact be launched with the
% goal of error minimization. The simulation can be runned until
% convergence to a minimal stable error!
% --> The type of model is needed (linear, ...)
% --> We don't need to chose one specific f-cross validation permutation.
% The principle is tu run it iteratively until min error convergence
%%%%%%%%%%%%%%%%%%%%%%%%%

%% TRY TO ITERATIVELY RUN THE MONTECARLO SIMULATION UNTIL MIN ERROR CONVERGENCE (only on classification error)

MaxIteration=500;

fprintf('\n');
fprintf('\n--> (III) CROSS-VALIDATION FOR PERFORMANCE ESTIMATION: \n');
fprintf('\n');
disp(strcat('Running Montecarlo simulation for 10-fold cross validation...(', ...
    num2str(MaxIteration*nbPartitions), '__iterations)...'));

% 2 columns, the first for train subset error, the second for the test
% subset error. 

minClassificationError=2;
minClassificationError_iterations=zeros(MaxIteration, 1);
BestClassifier=[];

tic
for j=1:MaxIteration
    j
    classificationErrors_all=zeros(nbPartitions, 2);
    for i=1:nbPartitions
        cp_new=repartition(cp);
        train_subset=cp_new.training(i);
        train_subset_data=TrainDataSet(train_subset, :);
        train_labels=Labels(train_subset);

        test_subset=cp_new.test(i);
        test_subset_data=TrainDataSet(test_subset, :);
        test_labels=Labels(test_subset);
        
        [classificationErrors_all(i, :), ~] = ...
            errors_subsets(train_subset_data, test_subset_data, train_labels, test_labels,...
            PriorType, Method{1}, classErrorRatios);
        
        if classificationErrors_all(i, 2)<minClassificationError
            minClassificationError=classificationErrors_all(i, 2);
            BestClassifier=fitcdiscr(train_subset_data, train_labels, 'discrimtype', Method{1} ,...
            'Prior', PriorType);
        end
    end
    minError_j=min(classificationErrors_all(:, 2));
    minClassificationError_iterations(j)=minError_j;
end
toc


iterations=[1:1:MaxIteration];
absoluteMin_idx=find(minClassificationError_iterations==min(minClassificationError_iterations));
absoluteMin=ones(1, numel(absoluteMin_idx))*min(minClassificationError_iterations); % size adapted to plot

fig_montecarlo=figure;
plot(iterations, minClassificationError_iterations);
hold on; 
p1=plot(iterations(absoluteMin_idx), absoluteMin, ...
    'Marker', 'p', 'MarkerSize', 10, 'LineStyle', 'none');
legend(p1, 'Absolute min classification error');
ylim([0.05, 0.5]);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);
xlabel('Iteration number');
ylabel('Min classification error of 10-fold cross-validation');

% save figure
print(fig_montecarlo, 'Montecarlo 10-fold cross validation simulation, min classification error=0,1000', '-dtiffn', '-r400');


%%%%%%%% OUTCOME %%%%%%%%
% --> the min error can by this way found and the best model selected. BUT
% this is not always a good idea, the min error could be found for a
% specific subset division that not represent well the datas. Therefore the
% outcome is not constant (depend on randomness) after the montecarlo
% simulation to find the best classifier. Solution: see course 22 --> the
% cross validation should only be used to evaluate if the model is good
% (evaluate the classification error). If the model is evaluated as good
% because of acceptable errors, since every sample is very precious (and
% especially here where we don't have a lot of samples), the real used
% model is evaluated on ALL the input training samples. By this way we
% should obtain a most reliable model. This was not done here, consider to
% do it to have a stable outcome that does not depend on "the chance of
% falling on the right samples division that gives the min error but does
% not represent at the best the data". (In any case, with this method used
% here, even if we don't have a stable outcome, we obtain however quite
% good results).
%%%%%%%%%%%%%%%%%%%%%%%%%
%% CLASSIFY SAMPLES OF TESTSET USING THE BEST LINEAR MODEL FOUND BEFORE, CREATE CSV PREDICTION FILE FOR KAGGLE UPLOAD
testSet=open('testSet.mat');
testSet=testSet.testData;
output = predict(BestClassifier, testSet);
labelToCSV(output, 'Session2.csv', './');

