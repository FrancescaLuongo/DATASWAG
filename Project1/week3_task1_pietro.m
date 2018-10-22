%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità
seed = 45;
rng(seed);

orderedFeatures = rankfeat(trainData,trainLabels,'fisher');

stopN = 1000;
k_fold = 10;
cvp = cvpartition(nObservations,'kfold',k_fold);

classErrors = [];

disp('iniziamo');
tic();

for nDF = 1:stopN %nDF = number of discriminant features
    selectedFeatures = orderedFeatures(1:nDF);
    
    cvClassErrors = [];
    
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
        
        
        classifierModel = classifierModel.setModelTypes({'linear','diaglinear'});

        % train classifier
        classifierModel = classifierModel.train();
        modelTypes = classifierModel.getModelTypes();

        % validation on test subset
        testRes = classifierModel.structuredResults(dataSetTest,dataLabelTest);

        cvClassErrors = [cvClassErrors,testRes.classErrors];
    end
    
    cvClassErrors = mean(cvClassErrors,2);
    
    classErrors = [classErrors,cvClassErrors];
end
toc();

%%
x = 1:stopN;

figure;
hold on;

for i = 1:length(modelTypes)
    plot(x,classErrors(i,:));
end
legend(modelTypes);






