%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità
seed = 45;
rng(seed);

% settings
startN = 1;
stopN = 8;
k_outer_fold = 3;
k_inner_fold = 10;
modelTypes = {'linear','diagquadratic'};


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

    % cross validation on the training data
    [cvErrors,modelTypes] = fisher_kcvClassifier( ... 
            dataSetInnerFold, ... 
            dataLabelInnerFold, ...
            'startN',startN,'stopN',stopN,...
            'kfold',k_inner_fold, ...
            'modelTypes',modelTypes, ...
            'trainErrors',true, ...
            'seed',seed ...
        );
    
    
    [~,I] = min(cvErrors.classErrorsMean)
    
    
    
    
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



