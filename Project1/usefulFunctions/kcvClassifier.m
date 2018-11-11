function [errors,successModelTypes] = ...
    kcvClassifier(data,labels,varargin)
%KCVCLASSIFIER 
%   k-fold cross validation classifier: 
%   usiamo questa funzione per non dover riscrivere ogni volta tutto
%   ritorna il modello e i risultati del training e del test


    % GESTIONE DEL VAR ARG IN
    default_kfold = 10;
    default_modelTypes = {'linear'};
    defaultSeed = -1;  %TODO ê giusto che il default seed sia negativo?

    p = inputParser;
    %addRequired(p,'width',validScalarPosNum);
    addOptional(p,'kfold',default_kfold);
    addParameter(p,'modelTypes',default_modelTypes);
    addParameter(p,'seed',defaultSeed);
    addParameter(p,'trainErrors',false);
    addParameter(p,'priorProbability','empiric');
    addParame
    parse(p,varargin{:});
    
    % INIZIALIZZAZIONE DELLE VARIABILI:
    k_fold = p.Results.kfold;
    mTypes = p.Results.modelTypes;
    seed   = p.Results.seed;
    priorProbability = p.Results.priorProbability;
    
    if seed < 0
        seed = 'default';%TODO aggiungi disp() che dice il cambiamento del seed
    end
    
    nSamples = size(data,1);
    
    rng(seed);
    defaultCVP = cvpartition(nSamples,'kfold',k_fold);
    
    addParameter(p,'customCVP',defaultCVP);
    parse(p,varargin{:});
    cvp = p.Results.customCVP;
    
    errors = [];
    classErrors = [];
    classificationError = [];
    trainingClassErrors = [];
    trainingClassificationError = [];
    
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
        Md1 = Md1.setPriorProbability(priorProbability);
        
        
        Md1 = Md1.setModelTypes(mTypes);

        % train classifier
        Md1 = Md1.train();
        
        % validation on test subset
        testRes = Md1.structuredResults(dataSetTest,dataLabelTest);
        
        if p.Results.trainErrors
            % test on train dataset
            trainingRes = Md1.structuredResults(dataSetTraining,dataLabelTraining);
            trainingClassErrors = [trainingClassErrors,trainingRes.classErrors];
            trainingClassificationError = [trainingClassificationError,trainingRes.classificationErrors];
        end

        classErrors = [classErrors,testRes.classErrors];
        classificationError = [classificationError,testRes.classificationErrors];
    end
    
    successModelTypes = Md1.getModelTypes();
    
    errors.classErrorsMean = mean(classErrors,2);
    errors.classificationErrorsMean = mean(classificationError,2);
    
    if p.Results.trainErrors
        errors.trainingClassErrorsMean = mean(trainingClassErrors,2);
        errors.trainingClassificationErrorsMean = mean(trainingClassificationError,2);
        errors.diffClassErrorsMean = abs(errors.trainingClassErrorsMean-errors.classErrorsMean);
        errors.diffClassificationErrorsMean = abs(errors.trainingClassificationErrorsMean-errors.classificationErrorsMean);
    end
    
end

