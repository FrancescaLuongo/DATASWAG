%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% TRAINING & TEST ERRORS %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% features selection
step = 9; % step = 1 per averle tutte
selectedFeatures = 1:step:nFeatures;
nSelectedFeatures = size(selectedFeatures,2);

modelTypes = {'linear','diaglinear','diagquadratic'};

%% AVVIA K-FOLD CROSS-VALIDATION TRAINING
N = nObservations;
k_fold = 10;

% set1 & set2 selection
cvp = cvpartition(N,'kfold',k_fold);

classErrorsTrain = [];
classErrorsTest = [];

for fold = 1:k_fold
    clear('classifierModel');
    
    % setting masks
    idxSetTraining = cvp.training(fold);
    idxSetTest = cvp.test(fold);
    
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
    classifierModel = classifierModel.setModelTypes(modelTypes);
    % SET PRIOR PROBABILITY
    classifierModel = classifierModel.setPriorProbability('empirical');
    
    % train classifier
    classifierModel = classifierModel.train();
    modelTypes = classifierModel.getModelTypes();
    
    % validation on test subset
    trainRes = classifierModel.structuredResults(dataSetTraining,dataLabelTraining);
    
    % validation on training subset
    testRes = classifierModel.structuredResults(dataSetTest,dataLabelTest);
    
    classErrorsTrain = [classErrorsTrain,trainRes.classErrors];
    classErrorsTest = [classErrorsTest,testRes.classErrors];
    
    % per l'ultimo punto:
    %cvp = repartition(cvp);
    
end

%% PLOT

meanClassErrorsTrain = mean(classErrorsTrain,2);
meanClassErrorsTestEmpirical = mean(classErrorsTest,2);

%%

figure;
bar([meanClassErrorsTestEmpirical,meanClassErrorsTestUniform,meanClassErrorsTestPrior]);
set(gca,'xticklabel',{'Empirical','Uniform','Defined'});
title('class error');
legend(modelTypes);
xlabel('(c)');



%% TREAT THE CLASS ERRORS
figure;
boxplot(classErrorsTrain',modelTypes);

