%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% TRAINING & TEST ERRORS %%%%%%%%%%%

%% TROVA LE MIGLIORI RANDOM FEATURES
% features selection

perc = 10;
selectedFeatures = 700:745;
nSelectedFeatures = sum(selectedFeatures);


% set1 & set2 selection
N = nObservations;
k_fold = 10;
cvp = cvpartition(N,'kfold',k_fold);

trials = 50;

bestError = 1;
bestFeatures = [];

tic();

for trial = 1:trials
    % elegant code per filtrare a cazzo
%     selectedFeatures = (randi([0 1000],1,nFeatures)/10)<=perc;
%     nSelectedFeatures = sum(selectedFeatures);
    
    
    classErrors = [];
    
    % AVVIA K-FOLD CROSS-VALIDATION TRAINING
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
        % teniamo solo il linear così va più veloce
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
    
    currentErrorMean = min(mean(classErrors,2));
    if currentErrorMean < bestError
        bestError = currentErrorMean;
        bestFeatures = selectedFeatures;
    end
    
end
toc();
bestError
%% TREAT THE CLASS ERRORS
figure;
boxplot(classErrors',modelTypes);

