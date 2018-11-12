classdef crossValidationClass
    %CROSSVALIDATIONCLASS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        errors = [];
        startPar = 1;
        stopPar = 1;
        stepPar = 1;
        parameterVariation = 'noVariation'; % nFeatures, altro
    end
    
    properties(Access = private)
        settedTypes = {'linear','quadratic','diaglinear','diagquadratic'};
        types = {};
        priorProbability = 'empirical';
        cvType = 'normal';
        k_fold = 10;
        seed = -1;
        trainErrors = false;
        data;
        labels;
        cvp;
        selectedFeatures = []; nFeatures = 0;
        dataSetTraining;dataLabelTraining;
        dataSetTest; dataLabelTest;
        
        classErrors = {};
        classificationError = {};
        trainingClassErrors = {};
        trainingClassificationError = {};
        currentVariation = 1;
        currentPar = 1;
    end
    
    methods
        function obj = setPriorProbability(obj,prior)
            obj.priorProbability = prior;
        end
        
        function obj = setModelTypes(obj,types)
            obj.settedTypes = types;
        end
        
        function types = getModelTypes(obj)
            types = obj.types;
        end
        
        function obj = setCV_type(obj,type)
            obj.cvType = type;
        end
        
        function obj = setKFold(obj,kfold)
            obj.k_fold = kfold;
        end
        
        function obj = setSeed(obj,seed)
            obj.seed = seed;
        end
        
        function obj = setTrainErrors(obj,trainErrors)
            obj.trainErrors = trainErrors;
        end
        
        function obj = setRangeFeatures(obj, range)
            obj.selectedFeatures = range;
        end
        
        function obj = setStartStop(obj,start,stop,step)
            obj.startPar = start;
            obj.stopPar = stop;
            obj.stepPar = step;
        end
        
        function obj = setData(obj,data,labels)
            obj.data = data;
            obj.labels = labels;
            obj.nFeatures = size(data,2);
            obj.selectedFeatures = 1:obj.nFeatures;
        end
        
        function obj = initPartitionByFold(obj,fold)
            % setting masks
            idxSetTraining = obj.cvp.training(fold);
            idxSetTest = obj.cvp.test(fold);

            % selecting data
            obj.dataSetTraining = obj.data(idxSetTraining,:);
            obj.dataLabelTraining = obj.labels(idxSetTraining);
            obj.dataSetTest = obj.data(idxSetTest,:);
            obj.dataLabelTest = obj.labels(idxSetTest);
        end
        
        function obj = runByFold(obj,fold)
            clear('Mdl');
            Mdl = obj.initializeClassifier();

            % train classifier
            Mdl = Mdl.train();

            % validation on test
            testRes = Mdl.structuredResults(obj.dataSetTest(:,obj.selectedFeatures),obj.dataLabelTest);

            if obj.trainErrors
                % test on train dataset
                trainingRes = Mdl.structuredResults(obj.dataSetTraining(:,obj.selectedFeatures),obj.dataLabelTraining);
                obj.trainingClassErrors{obj.currentVariation,fold} = trainingRes.classErrors;
                obj.trainingClassificationError{obj.currentVariation,fold} = trainingRes.classificationErrors;
            end

            obj.classErrors{obj.currentVariation,fold} = testRes.classErrors;
            obj.classificationError{obj.currentVariation,fold} = testRes.classificationErrors;
            
            obj.types = Mdl.getModelTypes();
        end
        
        function obj = runCV(obj)
            
            obj = obj.initializeCVP();
            
            for fold = 1:obj.k_fold
                obj = obj.initPartitionByFold(fold);
                obj = obj.processByCvType();
                
                obj.currentVariation = 1; % counter actually
                for cPar = obj.startPar:obj.stepPar:obj.stopPar
                    
                    obj = obj.changePar(cPar);
                    
                    obj = obj.runByFold(fold);
                    
                    obj.currentVariation = obj.currentVariation+1;
                end
            end
            
            obj = obj.treatError();
            
        end
        
        function obj = treatError(obj)
            for variation = 1:size(obj.classErrors,1)
                obj.errors{variation}.classErrorsMean = mean(obj.classErrors{variation},2);
                obj.errors{variation}.classificationErrorsMean = mean(obj.classificationError{variation},2);

                if obj.trainErrors
                    obj.errors{variation}.trainingClassErrorsMean = mean(obj.trainingClassErrors{variation},2);
                    obj.errors{variation}.trainingClassificationErrorsMean = mean(obj.trainingClassificationError{variation},2);
                    obj.errors{variation}.diffClassErrorsMean = abs(obj.errors{variation}.trainingClassErrorsMean-obj.errors{variation}.classErrorsMean);
                    obj.errors{variation}.diffClassificationErrorsMean = abs(obj.errors{variation}.trainingClassificationErrorsMean-obj.errors{variation}.classificationErrorsMean);
                end
            end
        end
        
        function res = getResultsByType(obj,modelType,errorType)
            iModelType = find(strcmp(obj.types,modelType));
            res = [];
            for variation = 1:size(obj.errors,2)
                errs = getfield(obj.errors{1,variation},errorType);
                res = [res,errs(iModelType)];
            end
        end
        
    end
    
    methods(Access = private)
        function obj = initializeCVP(obj)
            obj.cvp = cvpartition(obj.labels,'kfold',obj.k_fold);
        end
        
        function Mdl = initializeClassifier(obj)
            Mdl = modelClassificationClass();
            Mdl = Mdl.setTrainData(obj.dataSetTraining(:,obj.selectedFeatures),obj.dataLabelTraining);
            Mdl = Mdl.setPriorProbability(obj.priorProbability);
            Mdl = Mdl.setModelTypes(obj.settedTypes);
        end
        
        function obj = processByCvType(obj)
            switch obj.cvType
                case 'normal'
                    %non fare niente
                case 'fisher'
                    obj = obj.processByFisher();
                case 'pca'
                    obj = obj.processByPCA();
            end 
        end
        
        function obj = processByFisher(obj)
            featureOrder = rankfeat(obj.dataSetTraining,obj.dataLabelTraining,'fisher');
            %reorder training set
            obj.dataSetTraining = obj.dataSetTraining(:,featureOrder);
            %reorder validation set
            obj.dataSetTest = obj.dataSetTest(:,featureOrder);
        end
        
        function obj = processByPCA(obj)
            mu = mean(obj.dataSetTraining);
            [wcoeff, obj.dataSetTraining] = pca(obj.dataSetTraining);
            
            % center test data
            obj.dataSetTest = bsxfun(@minus, obj.dataSetTest, mu);

            % project test data onto principal components
            obj.dataSetTest = obj.dataSetTest * wcoeff;
        end
        
        function obj = changePar(obj,cPar)
            switch obj.parameterVariation
                case 'nFeatures'
                    obj.selectedFeatures = obj.startPar:obj.stepPar:cPar;
                case 'altro'
                    % fai altre cose
                case 'noVariation'
                    % do nothing
            end
        end
            
    end
end

