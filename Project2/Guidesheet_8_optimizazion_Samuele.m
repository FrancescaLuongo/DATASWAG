clear all;
close all;
clc;
%% Loading data

Data = load('./Data/Data.mat');

[mData, nData] = size(Data.Data);


trainData = Data.Data(1:round(mData*0.05),:);
trainX = Data.PosX(1:round(mData*0.05),:);
trainY = Data.PosY(1:round(mData*0.05),:);

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

%%
I = ones(mTrainData,1);
testI = ones(mTestData,1);
%train Data in PC space, only using two features for speed
FM = trainData(:,1:960);
%test Data in PC space, only using two features for speed
testFM = testData(:,1:960);

%% Optimization on LASSO

nalpha= 1;
lambdas = logspace(-10,20,0);
MSEmatrixX = zeros(nalpha/0.01, 76);
MSEmatrixY = zeros(nalpha/0.01, 76);

iteration = 1;

for na=0.01:0.01:nalpha
    [bX,FitInfoX] = lasso(trainData, trainX,'Alpha',na,'CV',10, ...
        'Lambda', lambdas);
    MSEmatrixX(iteration,:)=FitInfoX.MSE;
    
    [bY,FitInfoY] = lasso(trainData, trainY,'Alpha',na,'CV',10, ...
        'Lambda', lambdas);
    MSEmatrixY(iteration,:)=FitInfoY.MSE;
    iteration = iteration+1;
end

%% 3D PLOTTING RESULTS

coordinateAlpha = 0.01:0.01:1;
coordinateLambda = logspace(-10,20,0);

figure(1);
surf(coordinateAlpha, coordinateLambda, MSEmatrixX(:,:), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosX');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');


figure(2);
surf(coordinateAlpha, coordinateLambda, MSEmatrixY(:,:), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosX');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');