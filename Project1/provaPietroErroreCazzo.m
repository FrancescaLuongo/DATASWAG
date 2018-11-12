%% LOADING DATA & VARS

loadingInVars_pietro;

%% 

CEs = [];


cvp = cvpartition(nObservations,'kfold',10);

% setting masks
idxSetTraining = cvp.training(2);
idxSetTest = cvp.test(2);

orderedFeatures = rankfeat(trainData(idxSetTraining),trainLabels(idxSetTraining),'fisher');

for endF = 1:5:200

    % initialize classifier
    classifier_linear = fitcdiscr(trainData(idxSetTraining),trainLabels(idxSetTraining),'DiscrimType','diagquadratic');

    yhat_linear = predict(classifier_linear,trainData(idxSetTest));
    
    CEs = [CEs,findCEByLabels(trainLabels(idxSetTest),yhat_linear)];
end
figure;
plot(CEs);
    

%%
CEs = [];

% create random partitions for a k-fold validation
nbPartitions=10;
cp=cvpartition(trainLabels, 'kfold', nbPartitions);
nbFeaturesExtractionIterations=100;

trainErrors_featExtraction=zeros(nbFeaturesExtractionIterations, nbPartitions);
testErrors_featExtraction=zeros(nbFeaturesExtractionIterations, nbPartitions);
PriorType='empirical';
discrimType='linear';
classErrorRatios=[0.5 0.5]; % but we don't look at the class error.

% in each fold, test the classifier with the single best feature, save
% training and test error. Repeat with the first 2 best features and so on
disp(strcat('Iterating on_', num2str(nbPartitions), '_folds...'));

train_subset_idx=cp.training(1);
train_labels=trainLabels(train_subset_idx);
train_subset_data=trainData(train_subset_idx, :);

test_subset_idx=cp.test(1);
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

    % initialize classifier
    classifier_linear = fitcdiscr(train_subset_data_feat_lowDim,train_labels,'DiscrimType','diagquadratic');

    yhat_linear = predict(classifier_linear,test_subset_data_feat_lowDim);

    CEs = [CEs,findCEByLabels(test_labels,yhat_linear)];

end
figure;
plot(CEs);
    