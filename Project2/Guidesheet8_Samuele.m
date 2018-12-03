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
[bXLasso, FitInfoLassoX] = lasso(trainData, trainX, 'CV', 10,...
  'Lambda', logspace(-10, 0, 15));
%bXlasso lignes est pour chaque lambda, cest les beta, intercept est le
%beta0

lambdX = FitInfoLassoX.Lambda; %gives vector of lambda
MSQX = FitInfoLassoX.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffX = FitInfoLassoX.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

%for the plot of the mean squared error
figure(1)
semilogx(lambdX,MSQX,'bx-')
hold on;
figure(2)
plot(lambdX,MSQX,'bo-')
hold on;
figure(3)
plot(lambdX,NonZeroCoeffX, 'bx-') % faut un plot qui montre mieux les petits lambda
hold on;
figure(4)
semilogx(lambdX,NonZeroCoeffX, 'bo-') %see that increasing lambda decreases the number of non zero coeffs, donc plus de 0
hold on;
lassoPlot(bXLasso,FitInfoLassoX,'PlotType','CV');
legend('show') % Show legend

%use the beta (vecteur B) and intercept to regress test data POSx et POSy,
%plot the data and compute the best MSE 

%lambda corresponding to best MSE value (the minimal):

BestLambdaX = FitInfoLassoX.LambdaMinMSE;
indexBestMSQX = FitInfoLassoX.IndexMinMSE; %c'est 10
BestMSQX = MSQX(1,indexBestMSQX);
BestInterceptX = FitInfoLassoX.Intercept(1,indexBestMSQX);%le beta0 (ordonée à l'origine)
BestBetaX = bXLasso(:,indexBestMSQX); %toues les coeffs beta

%regression
POStestX = BestInterceptX + testFM * BestBetaX ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perfX = immse(testX, POStestX);

% plot(POStestX)

%% Elastic net
%it's adding a weight to the two constraints, this weight is alpha, it's an hyperparameter
%when alpha =1, the elastic net corresponds to lasso

[bXelnet,FitInfelnetX] = lasso(trainData, trainX,'Alpha',0.5,'CV',10, ...
    'Lambda', logspace(-10, 0, 15));


lambdelastnetX = FitInfelnetX.Lambda; %gives vector of lambda
MSQelnetX = FitInfelnetX.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffelnetX = FitInfelnetX.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

%for the plot of the mean squared error
figure(1)
semilogx(lambdelastnetX,MSQelnetX,'ro-')
hold on;
figure(2)
plot(lambdelastnetX,MSQelnetX,'rx-')
hold on;
figure(3)
plot(lambdelastnetX,NonZeroCoeffelnetX, 'ro-') % faut un plot qui montre mieux les petits lambda
hold on;
figure(4)
semilogx(lambdelastnetX,NonZeroCoeffelnetX, 'rx-') %see that increasing lambda decreases the number of non zero coeffs, donc plus de 0
hold on;
lassoPlot(bXelnet,FitInfelnetX,'PlotType','CV');
legend('show') % Show legend

%use the beta (vecteur B) and intercept to regress test data POSx et POSy,
%plot the data and compute the best MSE 

%lambda corresponding to best MSE value (the minimal):

BestLambdaelnetX = FitInfelnetX.LambdaMinMSE;
indexBestMSQelnetX = FitInfelnetX.IndexMinMSE; %c'est 10
BestMSQelnetX = MSQelnetX(1,indexBestMSQelnetX);
BestInterceptelnetX = FitInfelnetX.Intercept(1,indexBestMSQelnetX);%le beta0 (ordonée à l'origine)
BestBetaelnetX = bXelnet(:,indexBestMSQelnetX); %toues les coeffs beta

%regression
POStestelnetX = BestInterceptelnetX + testFM * BestBetaelnetX ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perfelnetX = immse(testX, POStestelnetX);
