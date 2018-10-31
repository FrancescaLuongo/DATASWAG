%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% PCA

%We have to evaluate PCA on the transpose of trainData
%because our principal components are the events and the
%features are the variables which have more or less influence

[coeff, score, variance] = pca(trainData.');

%% Covariance comparison 
% We compare covariance of trainData and PCAscore Data

%Centering trainData
trainData = trainData-repmat(mean(trainData),size(trainData,1),1);


trainDataCov = cov(trainData);
figure(1);
image(trainDataCov,'CDataMapping','scaled');

PCAScoreDataCov = cov(score.');
figure(2);
image(PCAScoreDataCov,'CDataMapping','scaled');

%%
%We compare the variances (diagonal elements) as ratios
VariancesRatios = diag(PCAScoreDataCov)./diag(trainDataCov);
figure(3);
plot(VariancesRatios);

CovariancesRatios = PCAScoreDataCov./trainDataCov;
image(CovariancesRatios,'CDataMapping','scaled');

%% Choosing the number of PCs
CumulativeVariance = cumsum((variance)/sum(variance));
image(CovariancesRatios,'CDataMapping','scaled');
