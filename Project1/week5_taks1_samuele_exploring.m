%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% SETTA I DATI PER IL TRAINER
% features selection
step = 1; % step = 1 per averle tutte
selectedFeatures = 1:step:nFeatures;
nSelectedFeatures = size(selectedFeatures,2);
seed = 45;
rng(seed);

%% Set values for Nested cross-val

N = nObservations;
N_inner = 0;
k_fold = 10;

% set1 & set2 selection
cvp = cvpartition(N,'kfold',k_fold);

classErrors = [];

%% Nested Inner Cross-val

for fold = 1:k_fold
    
    N_inner = cvp.training(fold); %not sure if it's training that we have to access
    cvp_inner = cvpartition(N_inner, 'kfold',kfold);
    clear('classifierModel');
    
    % setting masks
    idxSetTraining = cvp_inner.training(fold);
    idxSetTest = cvp_inner.test(fold);
    
    % selecting data
    dataSetTraining = trainData(idxSetTraining,selectedFeatures);
    dataLabelTraining = trainLabels(idxSetTraining);
    dataSetTest = trainData(idxSetTest,selectedFeatures);
    dataLabelTest = trainLabels(idxSetTest);
    
    % initialize classifier
    classifierModel = modelClassificationClass();
    classifierModel = classifierModel.setTrainData(dataSetTraining,dataLabelTraining);
    % togliamo il quadratic che se per alcuni lo trova e per altri no la
    % dimensione della matrice finale non sarà valida
    classifierModel = classifierModel.setModelTypes({'linear','diaglinear','diagquadratic'});
    
    % train classifier
    classifierModel = classifierModel.train();
    modelTypes = classifierModel.getModelTypes();
    
    % validation on test subset
    testRes = classifierModel.structuredResults(dataSetTest,dataLabelTest);
    
    classErrors = [classErrors,testRes.classErrors];
    
    % per l'ultimo punto:
    %cvp = repartition(cvp);
    
end



%% outer Cross-val
