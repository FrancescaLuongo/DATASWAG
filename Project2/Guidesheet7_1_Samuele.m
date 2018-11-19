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

for index = 1:mTestData
    testData(index,:) = (testData(index,:)-mu)./sigma;
end

%% PCA

pcaCoeff = pca(normalizedTrainData,'Centered','off');

projectedTrainingData = normalizedTrainData*pcaCoeff;

scatter3(projectedTrainingData(:,1),projectedTrainingData(:,2),projectedTrainingData(:,3));

%% Regression
I = ones(mTrainData,1);
FM = projectedTrainingData(:,1:2);

X = [ I FM ];