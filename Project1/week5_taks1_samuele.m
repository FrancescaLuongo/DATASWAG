%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilit�
seed = 45;
rng(seed);

orderedFeatures = rankfeat(trainData,trainLabels,'fisher');

% settings
startN = 1;
stopN = 10;
k_outer_fold = 10;
k_inner_fold = 10;
modelTypes = {'linear','diaglinear'};


% attenzione, non so se � giusto, ma le partition che si creano negli inner
% fold non sono sempre le stesse tra un outer e l'altro..
%TODO Questo non va bene dovrebbero essere le stesse per la riproducibilit�
%credo (SEM)

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
bestClassErrorsStd  = []; % boh magari serviranno - S� servono! Bravo Pie (Sem)

for outerFold = 1:k_outer_fold
    bestClassErrors = [bestClassErrors; ...
        min([outerCvErrors(outerFold,:).classErrorsMean]')];
    bestClassErrorsStd = [bestClassErrorsStd, ...
        std([outerCvErrors(outerFold,:).classErrorsMean]')];
end

%% t-TEST

[hLinear, pLinear] = ttest(bestClassErrors(1,:), 0.5);
[hDiagLinear, pDiagLinear] = ttest(bestClassErrors(2,:), 0.5);

disp(pLinear);
disp(pDiagLinear);
%We can not reject the null-hypothesis that the distribution comes
%from a distribution with mean 0.5 with a 0.05 probability because 
%the ttest returns us a p value too high.


