%% LOADING DATA & VARS

loadingInVars_pietro;

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


%% CROSS VALIDATION PCA
% per la riproducibilità
% seed = 45; fanculo la riproducibilità

startN = 1; 
stopN = 500; % fino a che numero di features si vuole provare
k_fold = 10;

classErrors = [];

tic();
[cvErrors,modelTypes] = ...
    pca_kcvClassifier(trainData,trainLabels,'kfold',k_fold, ...
    'startN',startN,'stopN',stopN, ...
    'modelTypes',{'diagquadratic'});

classErrors = [classErrors,cvErrors.classErrorsMean];
toc();

%%
plot(classErrors);
xlabel('# principal components');
ylabel('Class error');
