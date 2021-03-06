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

%%
I = ones(mTrainData,1);
testI = ones(mTestData,1);
%train Data in PC space, only using two features for speed
FM = trainData(:,1:960);
%test Data in PC space, only using two features for speed
testFM = testData(:,1:960);

%% optimization of hyperparameters alpha et lambda
nlambda = 1 ;
nalpha= 1 ;
bXgrid = [];
bYgrid =[]; %size est 960 100
FitInfoXgrid =[]; %taille 1 100 (car 10 lambda et 10 alpha (donc les 10 premiers du tableau cest pour le 1e lamdba etc))
FitInfoYgrid =[];
alpha =[0.1:0.1:nalpha];
lambda = [1:1:nlambda];
Yperf=[];
Xperf=[];
for nl=0.01:0.01:nlambda% si regarde les autres graphs cest trop grand 1 � 10 pour lambda et les nn zero coefs sont nuls plus devient grand
    for na=0.1:0.1:nalpha
        [bX,FitInfoX] = lasso(trainData, trainX,'Alpha',na,'CV',10, ...
        'Lambda', nl);
        bXgrid= [bXgrid bX];
        FitInfoXgrid =[FitInfoXgrid FitInfoX];


        [bY, FitInfoY] = lasso(trainData, trainY,'Alpha',na, 'CV', 10,...
        'Lambda', nl);
        bYgrid = [bYgrid bY];
        FitInfoYgrid =[FitInfoYgrid FitInfoY];
    end
end 

for n = 1:1:size(FitInfoXgrid')
        FitInfXg = FitInfoXgrid(1,n);
        FitInfYg = FitInfoYgrid(1,n);
        bXb = bXgrid(:,n);
        bYb= bYgrid(:,n);
        
        lambdelastnetX = FitInfXg.Lambda; %gives vector of lambda
        MSQelnetX = FitInfXg.MSE; %gives vector of MSE corresponding to each lambda
        NonZeroCoeffelnetX = FitInfXg.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

        lambdelastnetY = FitInfYg.Lambda; %gives vector of lambda
        MSQelnetY = FitInfYg.MSE; %gives vector of MSE corresponding to each lambda
        NonZeroCoeffelnetY = FitInfYg.DF; % gives vector, each column correspnds to the nb of non zero coeffs corresponding to a lambda

  %plot 1%  

        %for the plot of the mean squared error
        figure(2)
        semilogx(lambdelastnetX,MSQelnetX,'rx-',lambdelastnetY,MSQelnetY,'bo-')
        title('semilog scale, lambda in function of the respective MSQ, lasso (b),elastic net (alpha 0.5)(r)elastic net (alpha de 0 � 1 )(x=r)(y=b)')
        hold on;
        figure(3)
        plot(lambdelastnetX,MSQelnetX,'rx-',lambdelastnetY,MSQelnetY,'bo-')
        title('lambda in function of the respective MSQ, elastic net (alpha de 0 � 1 )(x=r)(y=b)')
        hold on;
        figure(4)
        plot(lambdelastnetX,NonZeroCoeffelnetX, 'rx-',lambdelastnetY,NonZeroCoeffelnetY, 'bo-') % faut un plot qui montre mieux les petits lambda
        title('lambda in function of the number of non zero coeff,elastic net (alpha de 0 � 1 )(x=r)(y=b)')
        hold on;
        figure(5)
        semilogx(lambdelastnetX,NonZeroCoeffelnetX, 'rx-',lambdelastnetY,NonZeroCoeffelnetY, 'bo-') %see that increasing lambda decreases the number of non zero coeffs, donc plus de 0
        title('semilog scale, lambda in function of the number of non zero coeff,elastic net (alpha de 0 � 1 )(x=r)(y=b)')
        hold on;
%         lassoPlot(bXb,FitInfXg,'PlotType','CV');
%         legend('show') % Show legend
% 
%         lassoPlot(bYb,FitInfYg,'PlotType','CV');
%         legend('show') % Show legend

        %use the beta (vecteur B) and intercept to regress test data POSx et POSy,
        %plot the data and compute the best MSE 

        %lambda corresponding to best MSE value (the minimal):

        BestLambdaelnetX = FitInfXg.LambdaMinMSE;
        indexBestMSQelnetX = FitInfXg.IndexMinMSE; %c'est 10
        BestMSQelnetX = MSQelnetX(1,indexBestMSQelnetX);
        BestInterceptelnetX = FitInfXg.Intercept(1,indexBestMSQelnetX);%le beta0 (ordon�e � l'origine)
        BestBetaelnetX = bXb(:,indexBestMSQelnetX); %toues les coeffs beta

        BestLambdaelnetY = FitInfYg.LambdaMinMSE;
        indexBestMSQelnetY = FitInfYg.IndexMinMSE; %c'est 10
        BestMSQelnetY = MSQelnetY(1,indexBestMSQelnetY);
        BestInterceptelnetY = FitInfYg.Intercept(1,indexBestMSQelnetY);%le beta0 (ordon�e � l'origine)
        BestBetaelnetY = bYb(:,indexBestMSQelnetY); %toues les coeffs beta

        %regression
        POStestelnetX = BestInterceptelnetX + testFM * BestBetaelnetX ;  %donne la regression pour chaque event
        %ou??? j sais pas trop
        %POStestX = [BestIntercept  testFM * BestBeta]; 
        perfelnetX = immse(testX, POStestelnetX);

        POStestelnetY = BestInterceptelnetY + testFM * BestBetaelnetY ;  %donne la regression pour chaque event
        %ou??? j sais pas trop
        %POStestX = [BestIntercept  testFM * BestBeta]; 
        perfelnetY = immse(testY, POStestelnetY);
        
Yperf=[Yperf perfelnetY];
Xperf=[Xperf perfelnetX];
        
%plot 2 

% 
%  figure(1)
%         plot(perfelnetX,perfelnetY, '*')
%         title('performance of test elastic net (alpha de 0 � 1)')
%         hold on;
end

for n=1:1:length(Yperf)
 figure(1)
        plot(Xperf(n),Yperf(n), '*')
        title('performance of test elastic net (alpha de 0 � 1)')
        hold on;
end
surface(Xperf,alpha, lambda) % on a un x perf pour chaque combinaison de alpha et lambda
surface(Yperf,alpha, lambda)%pas afficher en 3d mais en 2d dixit assistants
view(3)