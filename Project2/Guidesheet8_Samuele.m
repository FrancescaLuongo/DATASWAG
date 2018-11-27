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

%% Regression
I = ones(mTrainData,1);
testI = ones(mTestData,1);
%train Data in PC space, only using two features for speed
FM = trainData(:,1:960);
%test Data in PC space, only using two features for speed
testFM = testData(:,1:960);

%regression for both coordinates
[BX1, perfX1] = TrainRegression(trainX, FM, 1);
[BY1, perfY1] = TrainRegression(trainY, FM, 1);
%performance of the test with the trained regression
[perfTestX1] = TestRegressionPerformance(testX, testFM,BX1, 1);
[perfTestY1] = TestRegressionPerformance(testY, testFM,BY1, 1);


%% LASSO
%fitinto est structure qui contient les meansqerr etc
[bXLasso, FitInfoLasso] = lasso(trainData, trainX, 'CV', 10,...
  'Lambda', logspace(-10, 0, 15));

lambd = FitInfoLasso.Lambda; %gives vector of lambda
MSQE = FitInfoLasso.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeff = FitInfoLasso.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

%for the plot of the mean squared error
figure(1)
semilogx(lambd,MSQE)
hold on;
figure(2)
plot (lambd,MSQE)

%use the beta (vecteur B) and intercept to regress test data POSx et POSy, plot the data and compute the test MSE 
%test*B pour eliminer les coeffs avec poids 0 et fait une regression avec les intercept
