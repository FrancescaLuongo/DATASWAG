close all;  clearvars;

%% I) Cross validation for hyperparameter selection
% each classifer trained with a different Nsel will be considered as
% different model, the goal is to select the best ones. for this we will
% study the class-averaged classification error and see how it evolves with
% Nsel. rankfeat() will be used to rank the features with the Fisher
% score method, that gives to all the features a score that measures how
% well a feature has both similar values for the same class and different
% values for other classes. 

SAVE_FIGURES=true;

inputDataSet='trainSet.mat';
inputLabel='trainLabels.mat';

trainData=open(inputDataSet);
trainData=trainData.trainData;

Labels=open(inputLabel);
Labels=Labels.trainLabels;  % 1=error, 0=correct

%%

[s_m, s_n] = size(trainData);

% transform input space into a low-dimensional feature space by a feature
% selection method. This is inportant to create a better, faster and easier
% to understand model. Multiple methods can be used, here we use the Fisher
% score method, that gives to all the features a score that measures how
% well a feature has both similar values for the same class and different
% values for other classes. Do it on ALL input data (all features, no
% subset division) because features have to be considered entirely and not
% partially.

% N=nb of feature subset
% K=nb of models=nbPartitions. The model type change but the model
% parameters change
% !!! Features dimensionality reduction: this is needed if the number of
% samples is low. In fact, the higher the complexity of the model, the more
% parameters have to be estimated. But to have a good estimation, a lot of
% samples are needed!!

% create random partitions for a k-fold validation
nbPartitions=10;
cp=cvpartition(trainLabels, 'kfold', nbPartitions);
nbFeaturesExtractionIterations=50;

% define matrix of test errors : a matrix of test errors of size [N, K], 
% where N is the number of models tried out and K is the number of folds.
% N is also the number of models tried out : each different subset of
% features gives a different model
% use only classification errors since are lower (see guidesheet 2)
trainErrors_featExtraction=zeros(nbFeaturesExtractionIterations, nbPartitions);
testErrors_featExtraction=zeros(nbFeaturesExtractionIterations, nbPartitions);
PriorType='empirical';
discrimType='linear';
classErrorRatios=[0.5 0.5]; % but we don't look at the class error.

% in each fold, test the classifier with the single best feature, save
% training and test error. Repeat with the first 2 best features and so on
disp(strcat('Iterating on_', num2str(nbPartitions), '_folds...'));
for i=1:nbPartitions
    disp(strcat('Fold nb_', num2str(i)));
    cp_new=repartition(cp);
    train_subset_idx=cp_new.training(i);
    train_labels=trainLabels(train_subset_idx);
    train_subset_data=trainData(train_subset_idx, :);

    test_subset_idx=cp_new.test(i);
    test_labels=trainLabels(test_subset_idx);
    test_subset_data=trainData(test_subset_idx, :);
    
    % order features from 1 to nbFeatures with rankfeat(). We put the
    % feature selection inside the cross validation loop because the
    % ranking of the features must be calculated from the same train set as
    % the one used to train and compute the classification error in each partition
    [orderedInd, orderedPower] = rankfeat(train_subset_data, train_labels, 'fisher');

    for j=1:length(orderedInd(1:nbFeaturesExtractionIterations))
        current_features=trainData(:, orderedInd(1, 1:j));
        
        train_subset_data_feat_lowDim=train_subset_data(:, orderedInd(1, 1:j));
        test_subset_data_feat_lowDim=test_subset_data(:, orderedInd(1, 1:j));
        

        [classificationErrors, ~] = ...
            errors_subsets(train_subset_data_feat_lowDim, test_subset_data_feat_lowDim, train_labels, test_labels,...
            PriorType, discrimType, classErrorRatios);
        trainErrors_featExtraction(j, i)=classificationErrors(1);
        testErrors_featExtraction(j, i)=classificationErrors(2);
    end
end


fig_train_errors_all=figure; 
p=plot(trainErrors_featExtraction, 'b');
hold on; 
m=plot(mean(trainErrors_featExtraction, 2), 'r', 'LineWidth', 2);
ylabel('classification errors');
xlabel('Nb. included best features');
legend([p(1), m], {'folds train classification errors', 'average'});
title('train errors for each fold');

fig_test_errors_all=figure; 
p=plot(testErrors_featExtraction, 'b');
hold on;
m=plot(mean(testErrors_featExtraction, 2), 'r', 'LineWidth', 2);
ylabel('classification errors');
xlabel('Nb. included best features');
legend([p(1), m], {'folds test classification errors', 'average'});
title('test errors for each fold');

fig_avg_train_error=figure;
plot(mean(trainErrors_featExtraction, 2), 'r');
ylabel('average classification error');
xlabel('Nb. included best features');
title('average train error');

fig_avg_tests_error=figure;
plot(mean(testErrors_featExtraction ,2), 'r');
ylabel('average classification error');
xlabel('Nb. included best features');
title('average test error');

if SAVE_FIGURES
    print(fig_train_errors_all, 'Training errors 10-folds cross validation with avg', '-dtiffn', '-r400');
    print(fig_test_errors_all, 'Test errors 10-folds cross validation with avg', '-dtiffn', '-r400');    print(fig_train_errors_all, 'Training errors 10-folds cross validation with avg', '-dtiffn', '-r400');
    print(fig_avg_train_error, 'Average training error 10-folds cross validation', '-dtiffn', '-r400');
    print(fig_avg_tests_error, 'Average test error 10-folds cross validation', '-dtiffn', '-r400');
end

% comparison between test and training error graphs: the value is not
% really comparable but the tread yes. At the beginning (only one feature),
% the risk of overfitting is very low. This risk increase by adding
% features (more complicated model). Therefore is normal that the training
% error decrease. We compare the gap between the 2 graphs. It becomes more
% and more large. 
% !!! if the initial training error is quite large but the test error stays
% stable around its initial value, it is better to chose the model with
% only one feature: more simple.Also, adding more parameters than needed will 
% not give a robust model (model that will reduce the chance of fitting noise)
% In all cases an evaluation criterium must be taken into account 
% (we need to defined the max variation % that allow the variation of the 
% test error to be considered as "stable"). 
% Analysis of the test (or validation) error curve:
% if the test error has two local minimas at 40 and 20 features. We choose the
% model with the smaller number of features (20) because it is less likely
% that this model will overfit.
% Comparison of the two curves: The train error is initially high and
% decreases with the number of features. This is because by increasing the
% number of features, we increase the noise and thus the chance that
% overfitting will occur. The test decreases fast and then stabilized
% before re-increasing when the number of features is high. This is because
% an overfitted model will be less efficient at classifying unseed data.
% Thus, it would be good to choose the model with a small enough number of
% features that can still minimize the test error. Even with 200 features,
% the test error does not decrease more.
% Note: this choice of hyperparameter is risky because based on a one time simulation,
% running the algorithm a second time could give very different results.
% This is why we do nested cross valdiation

%% II) Nested Cross Validation for Performance Estimation, Hyperparameter=nb. of features only

% create the outer partition
nbFeaturesExtractionIterations=500;
PriorType='empirical';
discrimType='linear';
RankFeatMethod='fisher';
classErrorRatios=[.5 .5];
nb_outer_folds = 5;
nb_inner_folds = 7;
cp_outer=cvpartition(Labels, 'kfold', nb_outer_folds);

% test inner error
validationError=zeros(nbFeaturesExtractionIterations, nb_inner_folds, nb_outer_folds);
avgValidationErrorMat=zeros(nbFeaturesExtractionIterations, nb_outer_folds);
% train inner error
trainInnerError=zeros(nbFeaturesExtractionIterations, nb_inner_folds, nb_outer_folds);
avgTrainInnerErrorMat=zeros(nbFeaturesExtractionIterations, nb_outer_folds);
% vector of optimal hyperparameter (nb of features), corresponding
% minimal average validation (inner test) error and associated average
% inner training error:  OptimalHyperparam=[Nsel; MinAvgValidationError; AssociatedInnerTrainingError];
OptimalHyperparam=zeros(3, nb_outer_folds);
% outer cross validation test error (for the best selected hyperparameters
% minimizing the average validation error (=inner average test error on
% i-inner folds)
TestError=zeros(1, nb_outer_folds);

% outer loop
disp(strcat('Iterating on_', num2str(nb_outer_folds), '_outer folds...'));
for o=1:nb_outer_folds
    disp(strcat('--> Outer fold nb_', num2str(o)));
    
    cp_new = repartition(cp_outer);
   
    outer_train_subset_idx=cp_new.training(o);
    outer_train_labels=Labels(outer_train_subset_idx);
    outer_train_subset_data=trainData(outer_train_subset_idx, :);

    outer_test_subset_idx=cp_new.test(o);
    outer_test_labels=Labels(outer_test_subset_idx);
    outer_test_subset_data=trainData(outer_test_subset_idx, :);

    cp_inner=cvpartition(outer_train_labels, 'kfold', nb_inner_folds);
  
    % inner loop
    disp(strcat('Iterating on_', num2str(nb_inner_folds), '_inner folds...'));
    for i=1:nb_inner_folds
        disp(strcat('Inner fold nb_', num2str(i)));
        cp_inner_new = repartition(cp_inner);

        inner_train_subset_idx=cp_inner_new.training(i);
        inner_train_labels=outer_train_labels(inner_train_subset_idx);
        inner_train_subset_data=outer_train_subset_data(inner_train_subset_idx, :);

        inner_test_subset_idx=cp_inner_new.test(i);
        inner_test_labels=outer_train_labels(inner_test_subset_idx);
        inner_test_subset_data=outer_train_subset_data(inner_test_subset_idx, :);
        
        [orderedInd, orderedPower] = rankfeat(inner_train_subset_data, ...
            inner_train_labels, RankFeatMethod);

        for f=1:length(orderedInd(1:nbFeaturesExtractionIterations))
            
            train_subset_data_feat_lowDim=inner_train_subset_data(:, orderedInd(1, 1:f));
            test_subset_data_feat_lowDim=inner_test_subset_data(:, orderedInd(1, 1:f));

            [classificationErrors, ~] = errors_subsets(...
                train_subset_data_feat_lowDim, test_subset_data_feat_lowDim, ...
                    inner_train_labels, inner_test_labels, PriorType, discrimType, ...
                classErrorRatios);
            
            validationError(f, i, o)=classificationErrors(2);
            trainInnerError(f, i, o)=classificationErrors(1);
        end
    end  
    
    avgValidationError=mean(validationError(:, :, o), 2);
    avgTrainInnerError=mean(trainInnerError(:, :, o), 2);
    [minAvgValidationE, idx_minV]=min(avgValidationError);
    [minAvgTrainE, idx_minT]=min(avgTrainInnerError);
    
    avgValidationErrorMat(:, o)=avgValidationError;
    avgTrainInnerErrorMat(:, o)=avgTrainInnerError;
    
    % save optimal hyperparameter Nsel
    OptimalHyperparam(1, o)=idx_minV;
    OptimalHyperparam(2, o)=minAvgValidationE;
    % save associated avg inner training error
    OptimalHyperparam(3, o)=avgTrainInnerError(idx_minV);
    
    
    [orderedInd, orderedPower] = rankfeat(outer_train_subset_data, ...
        outer_train_labels, RankFeatMethod);
    outer_train_subset_data_feat_lowDim=outer_train_subset_data(:, orderedInd(1: idx_minV));
    outer_test_subset_data_feat_lowDim=outer_test_subset_data(:, orderedInd(1: idx_minV));
    
        
    [classificationErrors, ~] = errors_subsets(...
        outer_train_subset_data_feat_lowDim, outer_test_subset_data_feat_lowDim, ...
            outer_train_labels, outer_test_labels, PriorType, discrimType, ...
        classErrorRatios);
    TestError(1, o)=classificationErrors(2);
    
end

%%
close all;

fig_avg_validation_err=figure; 
LegendInfo={};
for p=1:nb_outer_folds
    plot(avgValidationErrorMat(:, p));
    hold on;
    LegendInfo{p}=strcat('Outer fold -->', num2str(p));
end
plot(mean(avgValidationErrorMat, 2), 'm', 'linewidth', 1.5);
ylim([0.2, 1.2]);
LegendInfo{end+1}='Average validation error';
yyaxis 'right';
ylim([0, 0.35]);
plot(std(avgValidationErrorMat, [], 2), 'c', 'linewidth', 0.5);
ylabel('STD');
LegendInfo{end+1}='STD of validation error';
legend(LegendInfo);
xlabel('Nb. included best features');
yyaxis 'left';
ylabel('Error value');
title('Average validation error for each outer fold');
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);

fig_avg_train_err=figure;
LegendInfo={};
for p=1:nb_outer_folds
    plot(avgTrainInnerErrorMat(:, p)); 
    hold on;
    LegendInfo{p}=strcat('Outer fold -->', num2str(p));
end
plot(mean(avgTrainInnerErrorMat, 2), 'm', 'linewidth', 1.5);
% ylim([0.2, 0.55]);
LegendInfo{end+1}='Average train error';
yyaxis 'right';
ylim([0, 0.25]);
plot(std(avgTrainInnerErrorMat, [], 2), 'c', 'linewidth', 0.5);
ylabel('STD');
LegendInfo{end+1}='STD of train error';
legend(LegendInfo);
xlabel('Nb. included best features');
yyaxis 'left';
ylabel('Error value');
title('Average train error for each outer fold');

fig_optimal_summary=figure; 
yyaxis 'right';
p1=plot(OptimalHyperparam(1, :), 'r');
hold on;
yyaxis 'left';
p2=plot(OptimalHyperparam(2, :), 'g');
p3=plot(OptimalHyperparam(3, :), 'g--');
ylabel('Error value');
yyaxis 'right';
ylabel('Nb. of optimal selected features');
xlabel('Outer fold');
legend([p1, p2, p3], {'N sel.', 'Min Avg. Validation Error', 'Associated Inner Training Error'});

fig_testError=figure; 
plot(TestError);
xlabel('Outer fold');
ylabel('Test Error');
title('Outer test subset error');

fig_boxplot=figure; 
boxplot([OptimalHyperparam(3, :)', OptimalHyperparam(2, :)', TestError'], 'Labels', ...
    {'Associated Inner Training Error', 'Min Avg Validation Error', 'Test Error'});
ylabel('Error value');
title('Different errors boxplots');
%%
if SAVE_FIGURES
    print(fig_avg_validation_err, 'Fisher, outerFolds=5, innerFolds=7, average validation error, nested cross validation', '-dtiffn', '-r400');
    print(fig_avg_train_err, 'Fisher, outerFolds=5, innerFolds=7, average test error, nested cross validation', '-dtiffn', '-r400');    
    print(fig_optimal_summary, 'Fisher, outerFolds=5, innerFolds=7, summary', '-dtiffn', '-r400');
    print(fig_testError, 'Fisher, outerFolds=5, test error', '-dtiffn', '-r400');
    print(fig_boxplot, 'Fisher, outerFolds=5, innerFolds=7, boxplots', '-dtiffn', '-r400');
end

%%
% build the final model on ALL the training labels
[minAvgValidationErr, idx]=min(OptimalHyperparam(2, :));
OptimalFeatureNb=OptimalHyperparam(1, idx);
[orderedInd, orderedPower] = rankfeat(trainData, ...
        Labels, 'fisher');
optimalFeaturesSubset=trainData(:, orderedInd(1:OptimalFeatureNb));
classifier=fitcdiscr(optimalFeaturesSubset, Labels, ...
    'discrimtype', 'linear' ,'Prior', 'empiric');
testSet=open('testSet.mat');
testSet=testSet.testData;
yhat=predict(classifier, testSet(:, orderedInd(1:OptimalFeatureNb)));
labelToCSV(yhat, 'Session3.csv', './');

%% II) Nested Cross Validation for Performance Estimation, Hyperparameter=nb. of features and classifier type (linear, ...)

% create the outer partition
nbFeaturesExtractionIterations=300;
PriorType='empirical';
% discrimType='linear';
discrimType={'linear', 'diaglinear', 'diagquadratic'};
classErrorRatios=[.5 .5];
nb_outer_folds = 5;
nb_inner_folds = 6;
cp_outer=cvpartition(Labels, 'kfold', nb_outer_folds);

% test inner error. Each position of the cell correspond to a discrimType
% (1st hyperparameter).
% Each matrix in a cell position is the validation error for all the Nb. of
% best features (2nd hyperparameter). The error is computed for all the
% inner folds and for all the outers subdivisions. Therefore, each matrix
% is M=nbFeaturesExtractionIterations x nb_inner_folds x nb_outer_folds
validationError=cell(1, length(discrimType));
% train inner error. Similar explaination that for validationError
trainInnerError=cell(1, length(discrimType));
% initialize thode 2 variables
for v=1:length(discrimType)
    validationError{v}=zeros(nbFeaturesExtractionIterations, nb_inner_folds, nb_outer_folds);
    trainInnerError{v}=zeros(nbFeaturesExtractionIterations, nb_inner_folds, nb_outer_folds);
end

% z-dimension: error for each nb. of features (1st hyperparameter) computed
% for each of the discrymtypes
avgValidationErrorMat=zeros(nbFeaturesExtractionIterations, nb_outer_folds, length(discrimType));
avgTrainInnerErrorMat=zeros(nbFeaturesExtractionIterations, nb_outer_folds, length(discrimType));

% vector of optimal hyperparameter (nb of features), corresponding
% minimal average validation (inner test) error and associated average
% inner training error:  OptimalHyperparam=[Nsel; MinAvgValidationError; AssociatedInnerTrainingError];
% z-dimension: best 1st hyperparameters (nb of features) for each
% discrimtype (2nd hyperparameter)
OptimalHyperparam=zeros(3, nb_outer_folds, length(discrimType));
% outer cross validation test error (for the best selected hyperparameters
% minimizing the average validation error (=inner average test error on
% i-inner folds)
% z-dimension: best 1st hyperparameters (nb of features) selected for each
% of the 2nd hyperparameter. TestError computed for each type of
% discrymtype
TestError=zeros(1, nb_outer_folds, length(discrimType));

% outer loop
disp(strcat('Iterating on_', num2str(nb_outer_folds), '_outer folds...'));
for o=1:nb_outer_folds
    disp(strcat('--> Outer fold nb_', num2str(o)));
    
    cp_new = repartition(cp_outer);
   
    outer_train_subset_idx=cp_new.training(o);
    outer_train_labels=Labels(outer_train_subset_idx);
    outer_train_subset_data=trainData(outer_train_subset_idx, :);

    outer_test_subset_idx=cp_new.test(o);
    outer_test_labels=Labels(outer_test_subset_idx);
    outer_test_subset_data=trainData(outer_test_subset_idx, :);

    cp_inner=cvpartition(outer_train_labels, 'kfold', nb_inner_folds);
  
    % inner loop
    disp(strcat('Iterating on_', num2str(nb_inner_folds), '_inner folds...'));
    for i=1:nb_inner_folds
        disp(strcat('Inner fold nb_', num2str(i)));
        cp_inner_new = repartition(cp_inner);

        inner_train_subset_idx=cp_inner_new.training(i);
        inner_train_labels=outer_train_labels(inner_train_subset_idx);
        inner_train_subset_data=outer_train_subset_data(inner_train_subset_idx, :);

        inner_test_subset_idx=cp_inner_new.test(i);
        inner_test_labels=outer_train_labels(inner_test_subset_idx);
        inner_test_subset_data=outer_train_subset_data(inner_test_subset_idx, :);
        
        [orderedInd, orderedPower] = rankfeat(inner_train_subset_data, ...
            inner_train_labels, 'fisher');
        
        % vary classifier type (=hyperparameter 1)
        for d=1:length(discrimType)
            % vary number of best features (=hyperparameter 2)
            for f=1:length(orderedInd(1:nbFeaturesExtractionIterations))

                train_subset_data_feat_lowDim=inner_train_subset_data(:, orderedInd(1, 1:f));
                test_subset_data_feat_lowDim=inner_test_subset_data(:, orderedInd(1, 1:f));

                [classificationErrors, ~] = errors_subsets(...
                    train_subset_data_feat_lowDim, test_subset_data_feat_lowDim, ...
                        inner_train_labels, inner_test_labels, PriorType, discrimType{d}, ...
                    classErrorRatios);

                validationError{d}(f, i, o)=classificationErrors(2);
                trainInnerError{d}(f, i, o)=classificationErrors(1);
            end
        end
    end  
    
    for d=1:length(discrimType)
        avgValidationError=mean(validationError{d}(:, :, o), 2);
        avgTrainInnerError=mean(trainInnerError{d}(:, :, o), 2);
        [minAvgValidationE, idx_minV]=min(avgValidationError);
        [minAvgTrainE, idx_minT]=min(avgTrainInnerError);
        avgValidationErrorMat(:, o, d)=avgValidationError;
        avgTrainInnerErrorMat(:, o, d)=avgTrainInnerError;
        % save optimal hyperparameter Nsel
        OptimalHyperparam(1, o, d)=idx_minV;
        OptimalHyperparam(2, o, d)=minAvgValidationE;
        % save associated avg inner training error
        OptimalHyperparam(3, o, d)=avgTrainInnerError(idx_minV);

        [orderedInd, orderedPower] = rankfeat(outer_train_subset_data, ...
        outer_train_labels, 'fisher');
        outer_train_subset_data_feat_lowDim=outer_train_subset_data(:, orderedInd(1: idx_minV));
        outer_test_subset_data_feat_lowDim=outer_test_subset_data(:, orderedInd(1: idx_minV));


        [classificationErrors, ~] = errors_subsets(...
            outer_train_subset_data_feat_lowDim, outer_test_subset_data_feat_lowDim, ...
                outer_train_labels, outer_test_labels, PriorType, discrimType{d}, ...
            classErrorRatios);
        TestError(1, o, d)=classificationErrors(2);
    end
end
%%
fig_val_error=figure; 
LegendInfo={};
colors={'r', 'g', 'b'};
colors_avg=[...
    150/255 0 19/255; ...
    0 100/255 0; ...
    0 191/255 255/255 ...
    ];
ppp=[];
for p=1:length(discrimType)
    yyaxis 'left'
    pp=plot(avgValidationErrorMat(:, :, p), colors{p}, 'linestyle', '-');
    ppp=[ppp, pp(1)];
    hold on;
    pm=plot(mean(avgValidationErrorMat(:, :, p), 2), ...
        'linewidth', 3, 'color', colors_avg(p, :), 'linestyle', '-');
    ppp=[ppp, pm];
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p});
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p}, ' -> avg');
    
    yyaxis 'right'
    ps=plot(std(avgValidationErrorMat(:, :, p), [], 2), ...
        'color', colors_avg(p, :), 'linestyle', ':');
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p}, ' -> std');
    ppp=[ppp, ps];
end
legend(ppp, LegendInfo);
xlabel('Nb. included best features');
yyaxis 'left'
ylabel('Errors, avg');
ylim([0.2 0.9]);
yyaxis 'right'
ylabel('std');
ylim([0 0.5]);
% title('Average validation error for each outer fold');
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);

fig_train_err=figure;
LegendInfo={};
colors={'r', 'g', 'b'};
colors_avg=[...
    150/255 0 19/255; ...
    0 100/255 0; ...
    0 191/255 255/255 ...
    ];
ppp=[];
for p=1:length(discrimType)
    yyaxis 'left'
    pp=plot(avgTrainInnerErrorMat(:, :, p), colors{p}, 'linestyle', '-'); 
    ppp=[ppp, pp(1)];
    hold on;
    pm=plot(mean(avgTrainInnerErrorMat(:, :, p), 2), ...
        'linewidth', 3, 'color', colors_avg(p, :), 'linestyle', '-');
    ppp=[ppp, pm];
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p});
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p}, ' -> avg');  
    
    yyaxis 'right'
    ps=plot(std(avgTrainInnerErrorMat(:, :, p), [], 2), ...
        'linewidth', 1, 'color', colors_avg(p, :), 'linestyle', ':');
    LegendInfo{end+1}=strcat('discrimType -->', discrimType{p}, ' -> std');
    ppp=[ppp, ps];
end

legend(ppp, LegendInfo);
xlabel('Nb. included best features');
yyaxis 'left'
ylabel('Errors, avg');
ylim([0 0.9]);
yyaxis 'right'
ylabel('std');
ylim([0 0.2]);
% title('Average train error for each outer fold');
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);


fig_hyperparam_selection=figure;
LegendInfo={};
colors={'r', 'g', 'b'};
ppp=[];
for p=1:length(discrimType)
    yyaxis 'left';
    p1=plot(OptimalHyperparam(1, :, p), colors{p}, 'linestyle', '-');
    ppp=[ppp p1];
    LegendInfo{end+1}=strcat('Nsel for discrymtype -->', discrimType{p});
    hold on;
    yyaxis 'right';
    p2=plot(OptimalHyperparam(2, :, p), colors{p}, 'linestyle', '--', 'marker', 'none');
    ppp=[ppp p2];
    LegendInfo{end+1}=strcat('Min avg validation error for discrymtype -->', discrimType{p});
    p3=plot(OptimalHyperparam(3, :, p), colors{p}, 'linestyle', ':', 'marker', 'none');
    ppp=[ppp p3];
    LegendInfo{end+1}=strcat('Associated Inner trainng error for discrymtype -->', discrimType{p});
end
grid on;
yyaxis 'left';
ylabel('Nb. of optimal selected features');
yyaxis 'right';
ylabel('Errors');
ylim([0.1 0.8]);
xlabel('Outer fold');
legend(ppp, LegendInfo);
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);


fig_outer_error=figure;
LegendInfo={};
colors={'r', 'g', 'b'};
ppp=[];
for p=1:length(discrimType)
	pp=plot(TestError(:, :, p), colors{p});
    hold on;
    ppp=[ppp, pp];
    LegendInfo{p}=strcat('discrimType -->', discrimType{p});
end
grid on;
xlabel('Outer fold');
ylabel('Test Error');
legend(ppp, LegendInfo);
title('Outer test subset error');
set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);


%%
if SAVE_FIGURES
    print(fig_val_error, 'Nested-cross, Average validation error for each outer fold, Nout=5, Nin=6', '-dtiffn', '-r400');
    print(fig_train_err, 'Nested-cross, Average train error for each outer fold, Nout=5, Nin=6', '-dtiffn', '-r400');    
    print(fig_hyperparam_selection, 'Nested-cross, Hyperparam. selection, Nout=5, Nin=6', '-dtiffn', '-r400');
    print(fig_outer_error, 'Nested-cross, Outer test subset error, Nout=5, Nin=6', '-dtiffn', '-r400');
end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% goal of nested cross validation:
% inner cross validation: find Nopt optimal nb. of features.
% outer cross validation: performance estimation of the model.
% Goal: have an unbiased performance estimation. With classical
% cross-validation, we have a biased estimation because Nopt was computed
% only once. There was no outer cross-validation to test the performance!
% The performance was directly tested only based on Nopt (that is here
% related only to inner!!).
% Cross validation: find the optimal hyperparameters. Stability issues: we
% select hyperparameters in a range that is more or less stable. Based on
% the stability, we might change the selection process criteria. We don't
% get any model out of CV!!! Once we optimize the model, we train the final
% model on ALL the training data.
% To select the best hyperparameter: average the validation errors on all
% the outer folds, take the min. If the min correspond to the max nb. of
% selected features (graphically), increase, max allowed nb. of best
% selected features. Always compare with training error. If we risk to have
% overfitting (too large gap between training (decays to zero) and test
% error,: risk of overfitting. In this case, choose a region where the
% validation error is STABLE. !!! compute also the std of the validation
% error on all the outer folds courbes, try to select a STABLE region where
% the variance is minimal!
%
% Final model construction:
% |---------------------------------------------------------------|
% | > CV --> optimal hyperparameters (classifier type, Nfeat, ...)|
% |                                                               |--> FINAL MODEL
% | > ALL training data                                           |
% |---------------------------------------------------------------|
