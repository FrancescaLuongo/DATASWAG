%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% PCA
setGlobalRanges(688,816);

[wcoeff, score, latent, ~, explained] = pca(trainData(:,usefulFeaturesRange));

%% Covariance comparison 
% We compare covariance of trainData and PCAscore Data

trainDataCov = cov(trainData(:,usefulFeaturesRange));
scoreCov = cov(score);

figure;
imagesc(trainDataCov);
figure;
imagesc(scoreCov);

%%
%We compare the variances (diagonal elements) as ratios
figure;
hold on;
plot(cumsum(var(score))./sum(var(score)));
hline(0.9);
hold off;

%%
%We compare the variances (diagonal elements) as ratios
bar(explained);


%% CROSS VALIDATION
% per la riproducibilità
seed = 45;
rng(seed);

startN = 1;
stopN = 120;
k_fold = 10;
cvp = cvpartition(nObservations,'kfold',k_fold);

classErrors = [];

tic();

[~, scoredData] = pca(trainData);

for nDF = startN:stopN %nDF = number of discriminant features
    selectedFeatures = 1:nDF;
    
    [cvErrors,modelTypes] = ...
        kcvClassifier(scoredData(:,selectedFeatures),trainLabels,'kfold',k_fold,'modelTypes',{'linear'});
        
    classErrors = [classErrors,cvErrors.classErrorsMean];
end
toc();

%%
x = startN:stopN;

figure;
hold on;

for i = 1:length(modelTypes)
    plot(x,classErrors(i,:));
end
legend(modelTypes);



