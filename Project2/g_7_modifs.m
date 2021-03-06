clear all;
close all;
clc;

%% Loading data

Data = load('./Data/Data.mat');
addpath('functions');

[mData, nData] = size(Data.Data);


trainData = Data.Data(1:round(mData*0.3),:);
trainX = Data.PosX(1:round(mData*0.3),:);
trainY = Data.PosY(1:round(mData*0.3),:);

[mTrainData, nTrainData] = size(trainData);

testData = Data.Data(mTrainData+1:round(mData*0.7),:);
testX = Data.PosX(mTrainData+1:round(mData*0.7),:);
testY = Data.PosY(mTrainData+1:round(mData*0.7),:);

valData = Data.Data(round(mData*0.7)+1:mData,:);
valX = Data.PosX(round(mData*0.7)+1:mData,:);
valY = Data.PosY(round(mData*0.7)+1:mData,:);

[mTestData, nTestData] = size(testData);


%% Normalization

[normalizedTrainData,mu,sigma] = zscore(trainData);

%We normalize test data with same parameters found for trainData
for index = 1:mTestData
    testData(index,:) = (testData(index,:)-mu)./sigma;
end

%% PCA

[pcaCoeff, score, variance,~,explained] = pca(normalizedTrainData,'Centered','off');

%We put our train data on PC space
projectedTrainingData = normalizedTrainData*pcaCoeff;

%We put our test Data on PC space using PCs found on train data
projectedTestData = testData*pcaCoeff;
%to show the percet of variance
cov_projected = cov(score);
cov_original = cov(normalizedTrainData);


% TODO: labels
figure(99)
imshow(cov_projected);
hold on;

% TODO: labels
figure(98)
imshow(cov_original);
hold on;

[t_max_cov_orig,t_ind_max_cov_orig] = max(cov_original.*(1-eye(size(cov_original))));
[max_cov_orig,ind_max_cov_orig] = max(t_max_cov_orig);

rel_cum_variance = cumsum(variance)/sum(variance);
figure(97)
plot(rel_cum_variance)
grid on
hline(0.95,'r', '95%')
xlabel('Number of principal components')
ylabel('Relative cumulative variance')
hold on;
figure(96)
scatter3(projectedTrainingData(:,1),projectedTrainingData(:,2),projectedTrainingData(:,3));

%% Regression
I = ones(mTrainData,1);
testI = ones(mTestData,1);
%train Data in PC space, only using two features for speed
FM = projectedTrainingData(:,1:100);
%test Data in PC space, only using two features for speed
testFM = projectedTestData(:,1:100);

%Order 1 Regressor
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

%% Increasing iteratively order of regressor

I = ones(mTrainData,1);
testI = ones(mTestData,1);

FM = projectedTrainingData(:,1:250);
testFM = projectedTestData(:,1:250);

nIterations = 2;
immseIterationTrainX = zeros(1,nIterations);
immseIterationTrainY = zeros(1,nIterations);
immseIterationTestX = zeros(1,nIterations);
immseIterationTestY = zeros(1,nIterations);

X = [I];
XTest = [testI];

for order = 1:1:nIterations
    X = [X FM.^order];
    XTest = [XTest testFM.^order];
    
    bX = regress(trainX, X);
    bY = regress(trainY, X);
    
    immseIterationTrainX(1,order) = immse(trainX, X*bX);
    immseIterationTrainY(1,order) = immse(trainY, X*bY);

    immseIterationTestX(1,order) = immse(testX, XTest*bX); 
    immseIterationTestY(1,order) = immse(testY, XTest*bY);
    
end

%% Plot result of regression
range = 1:nIterations;
figure;
plot(range,immseIterationTrainX,'rx', range,immseIterationTrainY,...
    'gx', range,immseIterationTestX,'ro', range,immseIterationTestY,'go');

axis([0 15 0 0.005]);

legend('TrainX MSE','TrainY MSE','TestX MSE', 'TestY MSE');
hold on


%% Regression with for loop integration of features (ordre 1)

nstep = 50; %nb de features que veut prendre
[exp,nbFeatures] = size(projectedTrainingData(1,:));%nb de features dans la PCA
extractedFeatures = [];
% nbFeatures=100;
for n = 1:nstep:nbFeatures
    %regression on the train set, ajoute une feature et fait la regression
     % peut prendre features dans l'ordre ou mieux random? DOIT prendre en
     % ordre!
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
    
    figure;
    plot(n,perfTestX1,'cx',n,perfTestY1,'rx',n,perfTestX2,'gx',n,perfTestY2,'bx')
%   plot(bX1,bY1,'o')
    hold on;
    plot(n,perfTrainX1,'co-', n, perfTrainY1, 'ro-',n,perfTrainX2,'go-', n, perfTrainY2, 'bo-')
    
    legend('Performance test X regression degr� 1','Performance test Y regression degr� 1', ...
       'Performance test X regression degr� 2' ,'Performance test Y regression degr� 2','p.train x regr degr 1', ...
       'P.train y regr deg 1','p.train x regr degr 2','P.train y regr deg 2')
    hold on;
end 

%% Final regressor

nstep = 10; %nb de features que veut prendre
[exp,nbFeatures] = size(projectedTrainingData(1,:));%nb de features dans la PCA


I = ones(mTrainData,1);
testI = ones(mTestData,1);

nIterations = 1; 
immseIterationTrainX = zeros(19,nIterations);
immseIterationTrainY = zeros(19,nIterations);
immseIterationTestX = zeros(19,nIterations);
immseIterationTestY = zeros(19,nIterations);

X = [I];
XTest = [testI];
step = 1;

for n = 1:nstep:851
    %regression on the train set, ajoute une feature et fait la regression
     % peut prendre features dans l'ordre ou mieux random? DOIT prendre en
     % ordre!
    FM = projectedTrainingData(:,1:n);
    testFM = projectedTestData(:,1:n);
    
    disp(n);
    
    for order = 1:1:nIterations
        X = [X FM.^order];
        XTest = [XTest testFM.^order];
        
        %bX = regress(trainX, X);
        bY = regress(trainY, X);

        %immseIterationTrainX(step,order) = immse(trainX, X*bX);
        %immseIterationTrainY(step,order) = immse(trainY, X*bY);

        %immseIterationTestX(step,order) = immse(testX, XTest*bX); 
        immseIterationTestY(step,order) = immse(testY, XTest*bY);
    end
    X = [I];
    XTest = [testI];
    
    step = step+1;

    
end

%% 3D PLOTTING RESULTS

coordinateOrder = 1:1:6;
coordinateFeatures = 1:50:500;

figure(1);
surf(coordinateOrder, coordinateFeatures, immseIterationTestX(1:10,1:6), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosX');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');

figure(2);
surf(coordinateOrder, coordinateFeatures, immseIterationTestY(1:10,1:6), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosY');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');

figure(3);
surf(coordinateOrder, coordinateFeatures, immseIterationTrainX(1:10,1:6), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosY');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');

figure(4);
surf(coordinateOrder, coordinateFeatures, immseIterationTrainY(1:10,1:6), ...
    'FaceAlpha',0.7);
colorbar
title ('MSE on Test partition for PosY');
xlabel ('Regression order');
ylabel('Number of features');
zlabel('MSE');
%% PLOTTING 2D

coordinateFeatures = 1:50:800;

figure(1);
hold on;

for i = 1:1:6
    plot(coordinateFeatures, immseIterationTestX(1:16,i));
    plot(coordinateFeatures, immseIterationTrainX(1:16,i));
end

legend('Order 1','Order 2','Order 3','Order 4','Order 5','Order 6');
xlabel('Number of features');
ylabel('Mean-Squared Error');
title('MSE on Train and Test partition for PosX');
axis([0 750 0 0.01]);


coordinateFeatures = 1:50:500;

figure(2);
hold on;

for i = 1:1:6
    plot(coordinateFeatures, immseIterationTestY(1:10,i));
    plot(coordinateFeatures, immseIterationTrainY(1:10,i));
end

legend('Order 1','Order 2','Order 3','Order 4','Order 5','Order 6');
xlabel('Number of features');
ylabel('Mean-Squared Error');
title('MSE on Train and Test partition for PosY');
axis([0 450 0 0.008]);


hold off;

coordinateFeatures = 1:50:800;

figure(3);
hold on;

for i = 1:1:6
    plot(coordinateFeatures, immseIterationTrainX(1:16,i));
end

legend('Order 1','Order 2','Order 3','Order 4','Order 5','Order 6');
xlabel('Number of features');
ylabel('Mean-Squared Error');
title('MSE on Train partition for PosX');
axis([0 750 0 0.001]);


coordinateFeatures = 1:50:500;

figure(4);
hold on;

for i = 1:1:6
    plot(coordinateFeatures, immseIterationTrainY(1:10,i));
end

legend('Order 1','Order 2','Order 3','Order 4','Order 5','Order 6');
xlabel('Number of features');
ylabel('Mean-Squared Error');
title('MSE on Train partition for PosY');
axis([0 450 0 0.001]);


hold off;