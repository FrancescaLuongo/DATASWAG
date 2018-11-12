function [errors,successModelTypes] = ...
    pca_kcvClassifier(data,labels,varargin)
%PCA_KCVCLASSIFIER 
%   yolo


    % GESTIONE DEL VAR ARG IN
    default_kfold = 10;
    default_modelTypes = {'linear'};
    defaultSeed = -1;

    p = inputParser;
    %addRequired(p,'width',validScalarPosNum);
    addOptional(p,'kfold',default_kfold);
    addParameter(p,'modelTypes',default_modelTypes);
    addParameter(p,'seed',defaultSeed);
    addParameter(p,'trainErrors',false);
    addParameter(p,'startN',20);
    addParameter(p,'stopN',20);
    parse(p,varargin{:});
    
    % INIZIALIZZAZIONE DELLE VARIABILI:
    k_fold = p.Results.kfold;
    mTypes = p.Results.modelTypes;
    seed   = p.Results.seed;
    startN = p.Results.startN;
    stopN   = p.Results.stopN;
    
    if seed < 0
        seed = 'shuffle';
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

        % setting masks
        idxSetTraining = cvp.training(fold);
        idxSetTest = cvp.test(fold);

        % selecting data + PCA
        mu = mean(data(idxSetTraining,:));
        [wcoeff, trainScore] = pca(data(idxSetTraining,:));
        dataLabelTraining = labels(idxSetTraining);
        dataSetTest = data(idxSetTest,:);
        dataLabelTest = labels(idxSetTest);
        
        % center test data
        testScore = bsxfun(@minus, dataSetTest, mu);
        
        % project test data onto principal components
        testScore = testScore * wcoeff;
        
        classErrorsNF = [];
        classificationErrorNF = [];
        for currentStop = startN:stopN
            % initialize classifier
            clear('Md1');
            Md1 = modelClassificationClass();
            Md1 = Md1.setTrainData(trainScore(:,startN:currentStop),dataLabelTraining);


            Md1 = Md1.setModelTypes(mTypes);

            % train classifier
            Md1 = Md1.train();

            % validation on test subset
            testRes = Md1.structuredResults(testScore(:,startN:currentStop),dataLabelTest);

            if p.Results.trainErrors
                % test on train dataset
                trainingRes = Md1.structuredResults(trainScore,dataLabelTraining);
                trainingClassErrors = [trainingClassErrors,trainingRes.classErrors];
                trainingClassificationError = [trainingClassificationError,trainingRes.classificationErrors];
            end

            classErrorsNF = [classErrorsNF;testRes.classErrors];
            classificationErrorNF = [classificationErrorNF;testRes.classificationErrors];
        end
        classErrors = [classErrors,classErrorsNF];
        classificationError = [classificationError,classificationErrorNF];
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

