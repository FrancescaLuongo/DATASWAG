function [Md1,classErrors,classificationError,modelTypes] = ...
    kcvClassifier(data,labels,k_fold,mTypes)
%KCVCLASSIFIER 
%   k-fold cross validation classifier: 
%   usiamo questa funzione per non dover riscrivere ogni volta tuttoz
%   ritorna il modello e i risultati del training e del test
    
    %fprintf('started a %i-fold cv \n',k_fold);

    nSamples = size(data,1);
    cvp = cvpartition(nSamples,'kfold',k_fold);

    classErrors = [];
    classificationError = [];
    
    for fold = 1:k_fold
        clear('classifierModel');

        % setting masks
        idxSetTraining = cvp.training(fold);
        idxSetTest = cvp.test(fold);

        % selecting data
        dataSetTraining = data(idxSetTraining,:);
        dataLabelTraining = labels(idxSetTraining);
        dataSetTest = data(idxSetTest,:);
        dataLabelTest = labels(idxSetTest);

        % initialize classifier
        Md1 = modelClassificationClass();
        Md1 = Md1.setTrainData(dataSetTraining,dataLabelTraining);
        
        
        Md1 = Md1.setModelTypes(mTypes);

        % train classifier
        Md1 = Md1.train();
        
        % validation on test subset
        testRes = Md1.structuredResults(dataSetTest,dataLabelTest);

        classErrors = [classErrors,testRes.classErrors];
        classificationError = [classificationError,testRes.classificationErrors];
    end
    
    modelTypes = Md1.getModelTypes();
    classErrors = mean(classErrors,2);
    classificationError = mean(classificationError,2);


end

