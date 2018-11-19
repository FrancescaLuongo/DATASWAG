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


%% Normalization

trainDataMeans = mean(trainData);
trainDataSTD = std(trainData);

%Need to normalize data here

%% PCA

pcaCoeff = pca(trainData);

transformedTrainingData = trainData*pcaCoeff;

scatter(transformedTrainingData(:,1),transformedTrainingData(:,2));