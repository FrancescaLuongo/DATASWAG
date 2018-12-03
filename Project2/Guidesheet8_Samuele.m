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

figure(1)
plot(perfTestX1,perfTestY1, 'g*')
title('performance of regress(g), lasso(b) and elastic net (alpha 0.5)(r)')
hold on;
%% LASSO
%fitinto est structure qui contient les meansqerr etc
[bXLasso, FitInfoLassoX] = lasso(trainData, trainX, 'CV', 10,...
  'Lambda', logspace(-10, 0, 15));
%bXlasso lignes est pour chaque lambda, cest les beta, intercept est le
%beta0
[bYLasso, FitInfoLassoY] = lasso(trainData, trainY, 'CV', 10,...
  'Lambda', logspace(-10, 0, 15));

lambdX = FitInfoLassoX.Lambda; %gives vector of lambda
MSQX = FitInfoLassoX.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffX = FitInfoLassoX.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

lambdY = FitInfoLassoY.Lambda; %gives vector of lambda
MSQY = FitInfoLassoY.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffY = FitInfoLassoY.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda
%for the plot of the mean squared error
figure(2)
semilogx(lambdX,MSQX,'bx-',lambdY,MSQY,'bo-')
title('semilog scale, lambda in function of the respective MSQ, lasso (b),elastic net (alpha 0.5)(r)')
hold on;
figure(3)
plot(lambdX,MSQX,'bx-',lambdY,MSQY,'bo-')
title('lambda in function of the respective MSQ, lasso (b),elastic net (alpha 0.5)(r)')
hold on;
figure(4)
plot(lambdX,NonZeroCoeffX, 'bx-',lambdY,NonZeroCoeffY, 'bo-') % faut un plot qui montre mieux les petits lambda
title('lambda in function of the number of non zero coeff, lasso (b),elastic net (alpha 0.5)(r)')
hold on;
figure(5)
semilogx(lambdX,NonZeroCoeffX, 'bx-',lambdY,NonZeroCoeffY, 'bo-') %see that increasing lambda decreases the number of non zero coeffs, donc plus de 0
title('semilog scale, lambda in function of the number of non zero coeff, lasso (b),elastic net (alpha 0.5)(r)')
hold on;
lassoPlot(bXLasso,FitInfoLassoX,'PlotType','CV');
legend('show') % Show legend

lassoPlot(bYLasso,FitInfoLassoY,'PlotType','CV');
legend('show') % Show legend
%use the beta (vecteur B) and intercept to regress test data POSx et POSy,
%plot the data and compute the best MSE 

%lambda corresponding to best MSE value (the minimal):

BestLambdaX = FitInfoLassoX.LambdaMinMSE;
indexBestMSQX = FitInfoLassoX.IndexMinMSE; %c'est 10
BestMSQX = MSQX(1,indexBestMSQX);
BestInterceptX = FitInfoLassoX.Intercept(1,indexBestMSQX);%le beta0 (ordonée à l'origine)
BestBetaX = bXLasso(:,indexBestMSQX); %toues les coeffs beta

BestLambdaY = FitInfoLassoY.LambdaMinMSE;
indexBestMSQY = FitInfoLassoY.IndexMinMSE; %c'est 10
BestMSQY = MSQY(1,indexBestMSQY);
BestInterceptY = FitInfoLassoY.Intercept(1,indexBestMSQY);%le beta0 (ordonée à l'origine)
BestBetaY = bYLasso(:,indexBestMSQY); %toues les coeffs beta

%regression
POStestX = BestInterceptX + testFM * BestBetaX ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perflassoX = immse(testX, POStestX);

POStestY = BestInterceptY + testFM * BestBetaY ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perflassoY = immse(testY, POStestY);

% plot(POStestX)
figure(1)
plot(perflassoX,perflassoY ,'b*')
hold on;
%% Elastic net
%it's adding a weight to the two constraints, this weight is alpha, it's an hyperparameter
%when alpha =1, the elastic net corresponds to lasso

[bXelnet,FitInfelnetX] = lasso(trainData, trainX,'Alpha',0.5,'CV',10, ...
    'Lambda', logspace(-10, 0, 15));

[bYelnet,FitInfelnetY] = lasso(trainData, trainY,'Alpha',0.5,'CV',10, ...
    'Lambda', logspace(-10, 0, 15));

lambdelastnetX = FitInfelnetX.Lambda; %gives vector of lambda
MSQelnetX = FitInfelnetX.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffelnetX = FitInfelnetX.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

lambdelastnetY = FitInfelnetY.Lambda; %gives vector of lambda
MSQelnetY = FitInfelnetY.MSE; %gives vector of MSE corresponding to each lambda
NonZeroCoeffelnetY = FitInfelnetY.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

%for the plot of the mean squared error
figure(2)
semilogx(lambdelastnetX,MSQelnetX,'rx-',lambdelastnetY,MSQelnetY,'ro-')
hold on;
figure(3)
plot(lambdelastnetX,MSQelnetX,'rx-',lambdelastnetY,MSQelnetY,'ro-')
hold on;
figure(4)
plot(lambdelastnetX,NonZeroCoeffelnetX, 'rx-',lambdelastnetY,NonZeroCoeffelnetY, 'ro-') % faut un plot qui montre mieux les petits lambda
hold on;
figure(5)
semilogx(lambdelastnetX,NonZeroCoeffelnetX, 'rx-',lambdelastnetY,NonZeroCoeffelnetY, 'ro-') %see that increasing lambda decreases the number of non zero coeffs, donc plus de 0
hold on;
lassoPlot(bXelnet,FitInfelnetX,'PlotType','CV');
legend('show') % Show legend

lassoPlot(bYelnet,FitInfelnetY,'PlotType','CV');
legend('show') % Show legend

%use the beta (vecteur B) and intercept to regress test data POSx et POSy,
%plot the data and compute the best MSE 

%lambda corresponding to best MSE value (the minimal):

BestLambdaelnetX = FitInfelnetX.LambdaMinMSE;
indexBestMSQelnetX = FitInfelnetX.IndexMinMSE; %c'est 10
BestMSQelnetX = MSQelnetX(1,indexBestMSQelnetX);
BestInterceptelnetX = FitInfelnetX.Intercept(1,indexBestMSQelnetX);%le beta0 (ordonée à l'origine)
BestBetaelnetX = bXelnet(:,indexBestMSQelnetX); %toues les coeffs beta

BestLambdaelnetY = FitInfelnetY.LambdaMinMSE;
indexBestMSQelnetY = FitInfelnetY.IndexMinMSE; %c'est 10
BestMSQelnetY = MSQelnetY(1,indexBestMSQelnetY);
BestInterceptelnetY = FitInfelnetY.Intercept(1,indexBestMSQelnetY);%le beta0 (ordonée à l'origine)
BestBetaelnetY = bYelnet(:,indexBestMSQelnetY); %toues les coeffs beta

%regression
POStestelnetX = BestInterceptelnetX + testFM * BestBetaelnetX ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perfelnetX = immse(testX, POStestelnetX);

POStestelnetY = BestInterceptelnetY + testFM * BestBetaelnetY ;  %donne la regression pour chaque event
%ou??? j sais pas trop
%POStestX = [BestIntercept  testFM * BestBeta]; 
perfelnetY = immse(testY, POStestelnetY);

figure(1)
plot(perfelnetX,perfelnetY, 'r*')
hold on;