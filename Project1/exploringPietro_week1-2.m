%% LOADING DATA & VARS

loadingInVars_pietro;

%% ricerca dei più bassi class error [validation test]

% elegant code per filtrare a cazzo 
maskSelection = randi([0 1],nObservations,1);

trainingSet = trainData(maskSelection == 1,:);
trainingSetLabels = trainLabels(maskSelection == 1);
validationSet = trainData(~maskSelection == 1,:);
validationSetLabels = trainLabels(~maskSelection == 1);

rangeFeatures = 500:1200;
tresholds = [];

for feature = rangeFeatures % trovo i migliori treshold per ogni feature
    [~,~,~,classError,minError,minIndex] = tresholdPlotter(trainingSet(:,feature),trainingSetLabels);
    foundedTreshold = classError(minIndex,1);
    tresholds = [tresholds;feature,foundedTreshold,minError];
end

% mo riscorro le features ma testo sul validationSet e prendo salvo i 
% ClassErros
classErrors = [];
for n = 1:numel(rangeFeatures) % trovo i migliori treshold per ogni feature
    feature = rangeFeatures(n);
    classError = findCEByTreshold(validationSet(:,feature),validationSetLabels,tresholds(n,2));
    classErrors = [classErrors;feature,tresholds(n,2),classError];
end

%%
erroriNorm = classErrors(:,3);
inverted = (1-(erroriNorm>0.5).*erroriNorm).*(erroriNorm>0.5);
erroriNorm = (erroriNorm<=0.5).*erroriNorm + inverted;

figure;
scatter(rangeFeatures,erroriNorm,'k*');

xlabel({'features'})
ylabel('Min class error')


%% trovati i più bassi, vai di probabilità semplici

selectedData = trainData(:,feature);

%selectedFeatures = [669,965,1093,1149,1608,1661,1736];
%selectedTresholds = [0.428976685655704,...
%    0.779063676301258,0.620938714550238,0.481, ...
%    0.552680646475706, 0.41, 0.598970439687342];

% random features!
n = 1800;
nRandomFeatures = 200;
selectedFeatures = randperm(n,nRandomFeatures);

selectedTresholds = tresholds(selectedFeatures,2);


idxSumm = zeros(length(selectedData),1);

for n = 1:numel(selectedFeatures)
    feature = selectedFeatures(n);
    idxSumm = idxSumm+(selectedData>selectedTresholds(n));
end

idx = idxSumm;
idx(idx<=100) = 0;
idx(idx>100) = 1;

sum(idx)

%%

treshold = 0.489247970397037;
bestFeature = 964;

selectedTestData = testData(:,bestFeature);


l = length(trainData(:,bestFeature));
figure;
scatter(1:l,trainData(:,bestFeature),[],trainLabels);
hline(treshold);

%%

idxTestData = selectedTestData > treshold;

labelToCSV(~idxTestData,'prediction_f1609_t0_603.csv','./predictions/')





%% plotmatrix per capire

% /!\ ATTENZIONE: cicla e genera un sacco di plotmatrix 

fromTo = 70:73; % fattore 10 di moltiplicazione (0:80) = (0:800)

for i = fromTo
    lRange = i*10; % low range
    tRange = (i+1)*10;
    setGlobalRanges(lRange,tRange);
    figure;
    gplotmatrix(trainData(:,usefulFeaturesRange),[],trainLabels,'br');
    title(strcat('Matrix da ',num2str(lRange),' a ',num2str(tRange)));
end
