%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% SETTA I DATI PER IL TRAINER
% features selection
step = 50; % step = 1 per averle tutte
featuresSelection = 1:step:nFeatures;
nSelectedFeatures = size(featuresSelection,2);
selectedModels = {'linear','quadratic','diaglinear','diagquadratic'};

%% CV TRAINER SENZA PRIOR
% per la riproducibilità
seed = 45;
rng(seed);
k_fold = 10;

classErrors = [];

tic();

[cvErrors,modelTypes] = ...
    kcvClassifier(trainData(:,featuresSelection),trainLabels,...
    'kfold',k_fold,'modelTypes',selectedModels);

classErrors = [classErrors,cvErrors.classErrorsMean];

toc();

%% CV TRAINER PRIOR UNIFORM
% per la riproducibilità
seed = 45;
rng(seed);
k_fold = 10;

classErrorsUniform = [];

tic();

[cvErrors,modelTypes] = ...
    kcvClassifier(trainData(:,featuresSelection),trainLabels,...
    'kfold',k_fold,'modelTypes',selectedModels,'priorProbability','uniform');

classErrorsUniform = [classErrorsUniform,cvErrors.classErrorsMean];

toc();


%% CV TRAINER PRIOR SETTED
% per la riproducibilità
seed = 45;
rng(seed);
k_fold = 10;

classErrorsPrior = [];

tic();

[cvErrors,modelTypes] = ...
    kcvClassifier(trainData(:,featuresSelection),trainLabels,...
    'kfold',k_fold,'modelTypes',selectedModels,'priorProbability',[0.3,0.7]);

classErrorsPrior = [classErrorsPrior,cvErrors.classErrorsMean];

toc();


%% PLOT DIFFERENZE
figure;
bar([classErrors,classErrorsUniform,classErrorsPrior]);
set(gca,'xticklabel',selectedModels);
title('class errors w/ cross validation');
legend('empiric','uniform','defined');




