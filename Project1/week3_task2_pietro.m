%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità
seed = 45;
rng(seed);

orderedFeatures = rankfeat(trainData,trainLabels,'fisher');

% settings
startN = 1;
stopN = 100;
k_outer_fold = 10;
k_inner_fold = 10;
modelTypes = {'linear','diaglinear','diagquadratic'};


% attenzione, non so se è giusto, ma le partition che si creano negli inner
% fold non sono sempre le stesse tra un outer e l'altro..
cvp = cvpartition(nObservations,'kfold',k_outer_fold);
seed = randi(100);
outerCvErrors = [];

tic();
for outerFold = 1:k_outer_fold

    % setting masks
    idxSetInnerFold = cvp.training(outerFold);
    idxSetOuterTest = cvp.test(outerFold);

    % selecting data
    dataSetInnerFold = trainData(idxSetInnerFold,:);
    dataLabelInnerFold = trainLabels(idxSetInnerFold);
    dataSetOuterTest = trainData(idxSetOuterTest,:);
    dataLabelOuterTest = trainLabels(idxSetOuterTest);
    
    errorsResults = [];
    for nDF = startN:stopN %nDF = number of discriminant features
        % selecting features
        selectedFeatures = orderedFeatures(1:nDF);

        % cross validation on the training data
        [cvErrors,modelTypes] = kcvClassifier( ... 
                dataSetInnerFold(:,selectedFeatures), ... 
                dataLabelInnerFold, ...
                'kfold',k_inner_fold, ...
                'modelTypes',modelTypes, ...
                'trainErrors',true, ...
                'seed',seed ...
            );
        errorsResults = [errorsResults,cvErrors];
    end

    outerCvErrors = [outerCvErrors;errorsResults];
    
end
toc();

%% extract stats on class errors:

bestClassErrors     = [];
bestClassErrorsStd  = []; % boh magari serviranno

for outerFold = 1:k_outer_fold
    bestClassErrors = [bestClassErrors; ...
        min([outerCvErrors(outerFold,:).classErrorsMean]')];
    bestClassErrorsStd = [bestClassErrorsStd, ...
        std([outerCvErrors(outerFold,:).classErrorsMean]')];
end

%%

figure;
boxplot(bestClassErrors,'Notch','off','Labels',modelTypes);
figure;
bar(bestClassErrors);
legend(modelTypes);



