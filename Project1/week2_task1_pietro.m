%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% features selection
step = 10; % step = 1 per averle tutte
featuresSelection = 1:step:nFeatures;
nSelectedFeatures = size(featuresSelection,2);
featuresSubset = trainData(:,featuresSelection);

%% LDA/QDA classifiers SENZA SPECIFICARE LA PRIOR P

% INIZIALIZZA
mClass = modelClassificationClass();
mClass = mClass.setTrainData(featuresSubset,trainLabels);
mClass = mClass.setModelTypes({'linear','diaglinear','diagquadratic'});

% ALLENA
mClass = mClass.trainAndCalc();

%% PLOT ERRORS

figure;
bar([mClass.getTrainClassErrors',mClass.getTrainClassificationErrors']);
set(gca,'xticklabel',mClass.getModelTypes()');
legend('class error','classification error');

%% LDA/QDA classifiers SPECIFICANDO PRIOR P UNIFORM

% INIZIALIZZA
mClassUniform = modelClassificationClass();
mClassUniform = mClassUniform.setTrainData(featuresSubset,trainLabels);
mClassUniform = mClassUniform.setModelTypes({'linear','diaglinear','diagquadratic'});

% SET PRIOR PROBABILITY
mClassUniform = mClassUniform.setPriorProbability('uniform');
% ALLENA
mClassUniform = mClassUniform.trainAndCalc();

%% PLOT ERRORS

figure;
bar([mClassUniform.getTrainClassErrors',mClassUniform.getTrainClassificationErrors']);
set(gca,'xticklabel',mClassUniform.getModelTypes()');
legend('class error','classification error');


%% LDA/QDA classifiers SPECIFICANDO PRIOR P 0.7 e 0.3

% INIZIALIZZA
mClassPrior = modelClassificationClass();
mClassPrior = mClassPrior.setTrainData(featuresSubset,trainLabels);
mClassPrior = mClassPrior.setModelTypes({'linear','diaglinear','diagquadratic'});

% SET PRIOR PROBABILITY
mClassPrior = mClassPrior.setPriorProbability([0.3,0.7]);
% ALLENA
mClassPrior = mClassPrior.trainAndCalc();

%% PLOT ERRORS

figure;
bar([mClassPrior.getTrainClassErrors',mClassPrior.getTrainClassificationErrors']);
set(gca,'xticklabel',mClassPrior.getModelTypes()');
legend('class error','classification error');


%% PLOT DIFFERENZE

figure;
bar([mClass.getTrainClassErrors',mClassUniform.getTrainClassErrors',mClassPrior.getTrainClassErrors']);
set(gca,'xticklabel',mClassPrior.getModelTypes()');
title('class errors');
legend('empiric','uniform','defined');

figure;
bar([mClass.getTrainClassificationErrors',mClassUniform.getTrainClassificationErrors',mClassPrior.getTrainClassificationErrors']);
set(gca,'xticklabel',mClassPrior.getModelTypes()');
title('classification errors');
legend('empiric','uniform','defined');


