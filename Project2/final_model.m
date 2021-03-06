clear all;
close all;
clc;
%% Loading data

Data = load('./Data/Data.mat');

[mData, nData] = size(Data.Data);

AllData = Data.Data;
AllX =  Data.PosX;
AllY = Data.PosY;

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

for index = 1:mTestData
    AllData(index,:) = (AllData(index,:)-mu)./sigma;
end

%% PCA

pcaCoeff = pca(normalizedTrainData,'Centered','off');

%We put our train data on PC space
projectedTrainingData = normalizedTrainData*pcaCoeff;

%We put our test Data on PC space using PCs found on train data
projectedTestData = testData*pcaCoeff;

projectedAllData = AllData*pcaCoeff;
%%
I = ones(mTrainData,1);
testI = ones(mTestData,1);
finalI = ones(mData,1);
%train Data in PC space, only using two features for speed
FM = projectedTrainingData(:,1:500);
%test Data in PC space, only using two features for speed
testFM = projectedTestData(:,1:500);

FMY = projectedTrainingData(:,1:100);
%test Data in PC space, only using two features for speed
testFMY = projectedTestData(:,1:100);

AllFMX = projectedAllData(:,1:500);
AllFMY = projectedAllData(:,1:100);
%% Training final models
%We do 100 features order 2 and 3 regression and alpha=0.05, lambda=0.005


%Order 2 Regressor (polynomial)
XOrder2 = [ I FM FM.^2];
testDataXOrder2 = [ testI testFM testFM.^2];

YOrder2 = [ I FMY FMY.^2];
testDataYOrder2 = [ testI testFMY testFMY.^2];

bX2 = regress(trainX, XOrder2);
bY2 = regress(trainY, YOrder2);


immseTrainX2 = immse(trainX, XOrder2*bX2);
immseTrainY2 = immse(trainY, YOrder2*bY2);

immseTestX2 = immse(testX, testDataXOrder2*bX2);
immseTestY2 = immse(testY, testDataYOrder2*bY2); 


%Lasso
[bXLasso,FitInfoX] = lasso(trainData, trainX,'Alpha',0.05, ...
        'Lambda', 0.005);
    
immseLassoX = immse(testX, (testData*bXLasso)+FitInfoX.Intercept);

[bYLasso,FitInfoY] = lasso(trainData, trainY,'Alpha',0.05, ...
        'Lambda', 0.005);
    
immseLassoY = immse(testY, (testData*bYLasso)+FitInfoY.Intercept);

%% Scatter Plot 

figure(1); %PosX MSE
range = 1:1:2;
MSEs = [immseTestX2 immseLassoX];

x = 1:2;
b = num2str(MSEs'); c = cellstr(b);
dx = 0.1; dy = 0.0001; % displacement so the text does not overlay the data points

scatter(range, MSEs, 40, 'filled', 'r');
axis([0 3 0 0.0025])
text(x+dx, MSEs+dy, c);

figure(2); %PosY MSE
range = 1:1:2;
MSEs = [immseTestY2 immseLassoY];

x = 1:2;
b = num2str(MSEs'); c = cellstr(b);
dx = 0.1; dy = 0.0001; % displacement so the text does not overlay the data points

scatter(range, MSEs, 40, 'filled', 'r');
axis([0 3 0 0.0025])
text(x+dx, MSEs+dy, c);

%% Plotting PosX and PosY for order2
index = 1:1:mTestData;
predictedPosX = zeros(1, mTestData);
predictedPosY = zeros(1, mTestData);
substractionX = zeros(1, mTestData);

for i=1:1:mTestData
    predictedPosX(1,i)= bX2(1) + dot(bX2(2:501),projectedTestData(i,1:500)) + ...
        dot(bX2(502:1001),(projectedTestData(i,1:500).^2));

    predictedPosY(1,i)= bY2(1) + dot(bY2(2:101),projectedTestData(i,1:100)) + ...
        dot(bY2(102:201),(projectedTestData(i,1:100).^2));
    
    substractionX(1,i) = abs(testX(i,1)-predictedPosX(1,i));
    
end



figure(3);
hold on;

plot(index, predictedPosX, 'b');
plot(index, testX, 'r');
legend('Predicted PosX','PosX');
axis([1300 1600 -0.1 0.25])
xlabel('Events');
ylabel('Position');

hold off;

figure(4);
hold on;

plot(index, predictedPosY, 'b');
plot(index, testY, 'r');
legend('Predicted PosY','PosY');
axis([1300 1600 0.12 0.32])
xlabel('Events');
ylabel('Position');


hold off;

figure(5);
hold on;

plot(index, substractionX, 'b');
legend('Substraction of PosX minus predicted PosX');



hold off;


%% TRAIN ON EVERYTHING

finalXOrder2 = [ finalI AllFMX AllFMX.^2];

finalYOrder2 = [ finalI AllFMY AllFMY.^2];

finalbX2 = regress(trainX, XOrder2);
finalbY2 = regress(trainY, YOrder2);
