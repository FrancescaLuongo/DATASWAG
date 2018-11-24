clear all;
close all;
clc;

%% Loading data

Data = load('./Data/Data.mat');

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
testFM = testData(:,1:2);

XOrder1 = [ I FM ];
testDataXOrder1 = [ testI testFM ];

%We make regressions for both coordinates
bX = regress(trainX, XOrder1);
bY = regress(trainY, XOrder1);


immseTrainX = immse(trainX, XOrder1*bX);
immseTrainY = immse(trainY, XOrder1*bY);

immseTestX = immse(testX, testDataXOrder1*bX);
immseTestY = immse(testY, testDataXOrder1*bX);

%Same with degree2 polynomial
XOrder2 = [ I FM FM.^2];
testDataXOrder2 = [ testI testFM testFM.^2];

bX = regress(trainX, XOrder2);
bY = regress(trainY, XOrder2);


immseTrainX = immse(trainX, XOrder2*bX);
immseTrainY = immse(trainY, XOrder2*bY);

immseTestX = immse(testX, testDataXOrder2*bX);
immseTestY = immse(testY, testDataXOrder2*bX);




