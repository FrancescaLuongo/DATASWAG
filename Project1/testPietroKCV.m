%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità

k_fold = 20;
modelTypes = {'linear','diagquadratic'};
startF = 1;
endF = 150;
stepF = 5;

tic();

% init kcv:
clear('kcv');
kcv = crossValidationClass();
kcv = kcv.setData(trainData,trainLabels);
kcv = kcv.setKFold(k_fold);
kcv = kcv.setModelTypes(modelTypes);
kcv = kcv.setTrainErrors(false);
kcv = kcv.setPriorProbability('empirical');

% se vogliamo un feature selection INSIDE la cv
kcv = kcv.setCV_type('fisher'); % fisher, normal, pca, ...

% se vogliamo variare dei parametri, ex: nFeatures varia il numero di
% features da prendere
kcv.parameterVariation = 'nFeatures'; % nFeatures, altro, ...
kcv = kcv.setStartStop(startF,endF,stepF);

% and run
kcv = kcv.runCV();

toc();

%% plot data:

x = startF:stepF:endF;

figure;
hold on;
res = kcv.getResultsByType('linear','classErrorsMean');
plot(x,res)
res = kcv.getResultsByType('diagquadratic','classErrorsMean');
plot(x,res)

legend(modelTypes);
xlabel({'number of features','(b)'});
ylabel('Class errors mean');


%% PROVA NESTED

% per la riproducibilità
seed = 45;
rng(seed);
startN = 1;
stopN = 1;

k_inner_fold = 10;
k_outer_fold = 5;
modelTypes = {'linear','diagquadratic'};

tic();

cvp = cvpartition(nObservations,'kfold',k_outer_fold);


for outerFold = 1:k_outer_fold
    
    % setting masks
    idxSetInnerFold = cvp.training(outerFold);
    idxSetOuterTest = cvp.test(outerFold);

    % selecting data
    dataSetInnerFold = trainData(idxSetInnerFold,:);
    dataLabelInnerFold = trainLabels(idxSetInnerFold);
    dataSetOuterTest = trainData(idxSetOuterTest,:);
    dataLabelOuterTest = trainLabels(idxSetOuterTest);

    clear('kcv');
    kcv = crossValidationClass();
    kcv = kcv.setData(dataSetInnerFold,dataLabelInnerFold);
    kcv = kcv.setKFold(k_inner_fold);
    kcv = kcv.setModelTypes(modelTypes);
    kcv = kcv.setCV_type('fisher');
    
    for innerFold = 1:k_inner_fold
        
        kcv = kcv.initPartitionByFold(outerFold);
        
        for to = startN:stopN
            kcv = kcv.setRangeFeatures(startN:to);
            kcv = kcv.runCurrentFold();
        end
        
        
        
        kcv = kcv.treatError();
    end

end

toc();



