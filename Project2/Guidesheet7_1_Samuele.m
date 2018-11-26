clear all;
close all;
clc;

%% Loading data

Data = load('./Data/Data.mat');
addpath('functions');

[mData, nData] = size(Data.Data);


trainData = Data.Data(1:round(mData*0.7),:);
trainX = Data.PosX(1:round(mData*0.7),:);
trainY = Data.PosY(1:round(mData*0.7),:);

[mTrainData, nTrainData] = size(trainData);

testData  = Data.Data(mTrainData+1:mData,:);
testX = Data.PosX(mTrainData+1:mData,:);
testY = Data.PosY(mTrainData+1:mData,:);

[mTestData, nTestData] = size(testData);

%% Normalization

[normalizedTrainData,mu,sigma] = zscore(trainData);

%We normalize test data with same parameters found for trainData
for index = 1:mTestData
    testData(index,:) = (testData(index,:)-mu)./sigma;
end

%% PCA

pcaCoeff = pca(normalizedTrainData,'Centered','off');

%We put our train data on PC space
projectedTrainingData = normalizedTrainData*pcaCoeff;

%We put our test Data on PC space using PCs found on train data
projectedTestData = testData*pcaCoeff;

scatter3(projectedTrainingData(:,1),projectedTrainingData(:,2),projectedTrainingData(:,3));

%% Regression
I = ones(mTrainData,1);
testI = ones(mTestData,1);
%train Data in PC space, only using two features for speed
FM = projectedTrainingData(:,1:2);
%test Data in PC space, only using two features for speed
testFM = projectedTestData(:,1:2);

%Ordre 1 Regressor
XOrder1 = [ I FM ];
testDataXOrder1 = [ testI testFM ];

%We make regressions for both coordinates
bX = regress(trainX, XOrder1);
bY = regress(trainY, XOrder1);

%evaluate performance of the regression
immseTrainX = immse(trainX, XOrder1*bX);
immseTrainY = immse(trainY, XOrder1*bY);

immseTestX = immse(testX, testDataXOrder1*bX); 
immseTestY = immse(testY, testDataXOrder1*bY); %Y plutot non?

%Ordre 2 Regressor (polynomial)
XOrder2 = [ I FM FM.^2];
testDataXOrder2 = [ testI testFM testFM.^2];

bX = regress(trainX, XOrder2);
bY = regress(trainY, XOrder2);


immseTrainX = immse(trainX, XOrder2*bX);
immseTrainY = immse(trainY, XOrder2*bY);

immseTestX = immse(testX, testDataXOrder2*bX);
immseTestY = immse(testY, testDataXOrder2*bY); 

%% Regression with for loop integration of features (ordre 1)

nstep = 50; %nb de features que veut prendre
[exp,nbFeatures] = size(projectedTrainingData(1,:));%nb de features dans la PCA
extractedFeatures = [];
% nbFeatures=100;
for n = 1:nstep:nbFeatures
    %regression on the train set, ajoute une feature et fait la regression
     % peut prendre features dans l'ordre ou mieux random?
    addedFeatures = projectedTrainingData(:,1:n);
    testaddedFeatures = projectedTestData(:,1:n);
    
    [bX1,perfTrainX1] = TrainRegression(trainX,addedFeatures,1);
    [bY1,perfTrainY1] = TrainRegression(trainY,addedFeatures,1);
    
    perfTestX1= TestRegressionPerformance(testX, testaddedFeatures,bX1,1); 
    perfTestY1= TestRegressionPerformance(testY, testaddedFeatures,bY1,1);
   
    [bX2,perfTrainX2]= TrainRegression(trainX,addedFeatures,2);
    [bY2,perfTrainY2] = TrainRegression(trainY,addedFeatures,2);
    
    perfTestX2= TestRegressionPerformance(testX, testaddedFeatures,bX2,2);
    perfTestY2= TestRegressionPerformance(testY, testaddedFeatures,bY2,2);
    
    
    plot(n,perfTestX1,'cx',n,perfTestY1,'rx',n,perfTestX2,'gx',n,perfTestY2,'bx')
%   plot(bX1,bY1,'o')
    hold on;
    plot(n,perfTrainX1,'co-', n, perfTrainY1, 'ro-',n,perfTrainX2,'go-', n, perfTrainY2, 'bo-')
    
    legend('Performance test X regression degré 1','Performance test Y regression degré 1', ...
       'Performance test X regression degré 2' ,'Performance test Y regression degré 2','p.train x regr degr 1', ...
       'P.train y regr deg 1','p.train x regr degr 2','P.train y regr deg 2')
    hold on;
end 