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

XOrder1 = [ I FM ];
testDataXOrder1 = [ testI testFM ];

%We make regressions for both coordinates
bX = regress(trainX, XOrder1);                       %pour comparer si la fonction faite et le code normal 
                                                        %donnent pareil et ouip c la meme chose
bY = regress(trainY, XOrder1);
immseTrainX = immse(trainX, XOrder1*bX);
immseTrainY = immse(trainY, XOrder1*bY);
% comme �a peut voir si la fct fait pareil
[BX1, perfX1] = TrainRegression(trainX, FM, 1);
[BY1, perfY1] = TrainRegression(trainY, FM, 1);

[perfTestX1] = TestRegressionPerformance(testX, testFM,BX1, 1);
[perfTestY1] = TestRegressionPerformance(testY, testFM,BY1, 1);

immseTestX = immse(testX, testDataXOrder1*bX);
immseTestY = immse(testY, testDataXOrder1*bY);

[perfTestX] = TestRegressionPerformance(testX, testFM,bX, 1); % juste pour comparer et voir si obtient m�me resultat
[perfTestY] = TestRegressionPerformance(testY, testFM,bY, 1);

%% LASSO

%fitinto est structure qui contient les meansqerr etc
[bXLasso, FitInfoLasso] = lasso(trainData, trainX, 'CV', 10,...
  'Lambda', logspace(-10, 0, 15), 'UseParallel', true)
%UseParallel � Set to true to compute in parallel. The default is false,
%pour faire aller plus vite l algorithme 
%lambda is the hyperparameter and its values are logspace(-10, 0, 15)


%for the plot of the mean squared error
semilogx(lambda,FitInfoLasso.MSE)  
