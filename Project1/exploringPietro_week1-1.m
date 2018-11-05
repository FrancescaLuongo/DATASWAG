clear all; close all; clc;

%% IMPORTING DATA

cd('C:\Users\Pietro\Documents\MATLAB\Data Analysis\Project1');
addpath('usefulFunctions');

testData = load('.\data\testSet.mat');
trainLabels = load('.\data\trainLabels.mat');
trainData = load('.\data\trainSet.mat');

testData = testData.testData;
trainLabels = trainLabels.trainLabels;
trainData = trainData.trainData;


sizeTest = size(testData);
sizeLabels = size(trainLabels);
sizeTrain = size(trainData);

nFeatures = sizeTrain(2);
nObservations = sizeTrain(1);

global nUsefulFeatures;
global usefulFeaturesRange;

setGlobalRanges(700,731);

%% ORDER DATA

trainErrorData = trainData(trainLabels==1,:);
trainCorrectData = trainData(trainLabels==0,:);

nErrorObservations = size(trainErrorData);x = 2*randn(5000,1) + 5;
nErrorObservations = nErrorObservations(1);
nCorrectObservations = size(trainCorrectData);
nCorrectObservations = nCorrectObservations(1);



%% EXPLORING DATA: 3D plot

% 3d plot
figure;
hold on;

for sampleN = 1:nObservations
    plot3(usefulFeaturesRange,ones(1,nUsefulFeatures)*sampleN,trainData(sampleN,usefulFeaturesRange));
end
%%
% 3d plot
setGlobalRanges(659,730);

figure;
hold on;

for sampleN = 1:nCorrectObservations
    plot3(usefulFeaturesRange,ones(1,nUsefulFeatures)*sampleN,trainCorrectData(sampleN,usefulFeaturesRange),'g');
end

for sampleN = 1:(nErrorObservations)
    plot3(usefulFeaturesRange,ones(1,nUsefulFeatures)*(nCorrectObservations+sampleN),trainErrorData(sampleN,usefulFeaturesRange),'r');
end

%%
% 3d plot only alcuni selected
figure;
hold on;

alcuniSelected0 = [3 8 100];
alcuniSelected1 = [1 8 4];

for sampleN = 1:length(alcuniSelected0)
    plot3(usefulFeaturesRange,ones(1,nUsefulFeatures)*sampleN,trainCorrectData(alcuniSelected0(sampleN),usefulFeaturesRange),'g');
end

for sampleN = 1:length(alcuniSelected1)
    plot3(usefulFeaturesRange,ones(1,nUsefulFeatures)*(nCorrectObservations+sampleN),trainErrorData(alcuniSelected0(sampleN),usefulFeaturesRange),'r');
end

%%
% 2d plot 
figure;
hold on;

setGlobalRanges(1,2048);

plot(usefulFeaturesRange,trainCorrectData(1,usefulFeaturesRange)','g');
% plot(usefulFeaturesRange,trainErrorData(:,usefulFeaturesRange)','r');

%%
% 2d plot diff
figure;
hold on;

setGlobalRanges(1,2048);

plot(usefulFeaturesRange,diff(trainCorrectData(:,usefulFeaturesRange))','g');
plot(usefulFeaturesRange,diff(trainErrorData(:,usefulFeaturesRange))','r');

%% histogram
figure;
hold on;

setGlobalRanges(600,750);

histogram(trainErrorData(:,usefulFeaturesRange),'FaceColor','r');
histogram(trainCorrectData(:,usefulFeaturesRange),'FaceColor','g');


%% min e max

peaksMax = max(trainData(:,usefulFeaturesRange),[],2);
peaksMin = min(trainData(:,usefulFeaturesRange),[],2);

figure;
histogram(peaksMax);

figure;
histogram(peaksMin);

%% min e max error & not

setGlobalRanges(700,750);
peaksMaxCorrect = max(trainCorrectData(:,usefulFeaturesRange),[],2);
peaksMaxError   = max(trainErrorData(:,usefulFeaturesRange),[],2);
peaksMinCorrect = min(trainCorrectData(:,usefulFeaturesRange),[],2);
peaksMinError   = min(trainErrorData(:,usefulFeaturesRange),[],2);

n_bins = 20;

figure;
hold on;

histogram(peaksMaxCorrect,n_bins);
histogram(peaksMaxError,n_bins);

figure;
hold on;
histogram(peaksMinCorrect,n_bins);
histogram(peaksMinError,n_bins);

%% AVG in range

setGlobalRanges(700,720);
sortedBySmallest = sort(trainData(:,usefulFeaturesRange)');
sortedBySmallest = mean(sortedBySmallest(1:15,:));
sizeSorted = size(sortedBySmallest);

scatter(1:sizeSorted(2),sortedBySmallest,[],trainLabels);


tresholdAccuracy = tresholdPlotter(sortedBySmallest',trainLabels);

figure;
scatter(tresholdAccuracy(:,1),tresholdAccuracy(:,2));

%% mooving AVG in range finder

bestMoving = [];
bestRange = [];

r = 15; % questo l'ho trovato con un ciclo di prova che ora ho eliminato
toMean = 11;

for i = 500:2000 % trovato 1506 come migliore
    setGlobalRanges(i,i+r);
    sortedBySmallest = sort(trainData(:,usefulFeaturesRange)');

    if toMean == 1
        sortedBySmallest = min(sortedBySmallest);
    else
        sortedBySmallest = mean(sortedBySmallest(1:toMean,:));
    end

    [~,maxVal] = tresholdPlotter(sortedBySmallest',trainLabels);
    bestMoving = [bestMoving;i,maxVal];
end

figure;
title(strcat('range lenght: ',num2str(r)));
scatter(bestMoving(:,1),bestMoving(:,2));



%% AVG in range max /!\ ATTENZIONE: notare il 'descend'

bestMoving = [];
bestRange = [];

r = 15; 
toMean = 11; 

for i = 500:2000 % trovato il 1505 come migliore
    setGlobalRanges(i,i+r);
    sortedBySmallest = sort(trainData(:,usefulFeaturesRange)','descend');

    if toMean == 1
        sortedBySmallest = min(sortedBySmallest);
    else
        sortedBySmallest = mean(sortedBySmallest(1:toMean,:));
    end

    [~,maxVal] = tresholdPlotter(sortedBySmallest',trainLabels);
    bestMoving = [bestMoving;i,maxVal];
end

figure;
title(strcat('range lenght: ',num2str(r)));
scatter(bestMoving(:,1),bestMoving(:,2));

%% AVG in range ricerca con SOLO UNA FEATURE

bestMoving = [];

for i = 600:1800 % trovato 1514 e 1769 come migliore
    [~,maxVal] = tresholdPlotter(trainData(:,i),trainLabels);
    bestMoving = [bestMoving;i,maxVal];
end

figure;
scatter(bestMoving(:,1),bestMoving(:,2));

%% AVG in range ricerca con SOLO UNA FEATURE classError

bestMoving = [];

for i = 600:1800 
    [~,~,~,classError,minVal,minIndex] = tresholdPlotter(trainData(:,i),trainLabels);
    bestTreshold = classError(minIndex,2);
    bestMoving = [bestMoving;i,minVal,bestTreshold];
end

figure;
scatter(bestMoving(:,1),bestMoving(:,2));

[M,I] = min(bestMoving(:,2));
bestFeature = bestMoving(I,1);
treshold = bestMoving(I,3);

l = length(trainData(:,bestFeature));

figure;
scatter(1:l,trainData(:,bestFeature),[],trainLabels);
hline(treshold);


%% AVG in range ricerca con PIU FEATURES classError (non completo, ma secondo me non funzionerà prendendo i minimi (overfitting))

piuFeatures = [1734,1609,1485,1093,966];
bestMoving = [];

for currentFeature = piuFeatures 
    [~,~,~,classError,minVal,minIndex] = tresholdPlotter(trainData(:,currentFeature),trainLabels);
    bestTreshold = classError(minIndex,2);
    bestMoving = [bestMoving;i,minVal,bestTreshold];
end

[M,I] = min(bestMoving(:,2));
bestFeature = bestMoving(I,1);
treshold = bestMoving(I,3);

l = length(trainData(:,bestFeature));

figure;
scatter(1:l,trainData(:,bestFeature),[],trainLabels);
hline(treshold);


%% AVG range finder in range ricerca con SOLO UNA FEATURE classError

bestMoving = [];
bestRange = [];

r = 15; 
toMean = 5; 

for i = 500:2000
    setGlobalRanges(i,i+r);
    sortedBySmallest = sort(trainData(:,usefulFeaturesRange)','descend');

    if toMean == 1
        sortedBySmallest = min(sortedBySmallest);
    else
        sortedBySmallest = mean(sortedBySmallest(1:toMean,:));
    end

    [~,~,~,classError,minVal,minIndex] = tresholdPlotter(sortedBySmallest',trainLabels);
    bestMoving = [bestMoving;i,minVal];
end

figure;
scatter(bestMoving(:,1),bestMoving(:,2));


%%

selectedTestData = testData(:,bestFeature);

idxTestData = selectedTestData > treshold;

labelToCSV(idxTestData,'predictionData1.csv','./predictions/')

%% GAIA TEST

idxTestGaiaData = zeros(199,1);

labelToCSV(idxTestGaiaData,'predictionDataGaia.csv','./predictions/')


%% CICLO A MANO ACCURACY:

close all;
clc;

datiScelti = [1:5,501:505];
featureScelta = 700;

l = length(trainData(datiScelti,700));


[tresholdAccuracy,maxVal,maxIndex,classError,minVal,minIndex] = tresholdPlotter(trainData(datiScelti,featureScelta),trainLabels(datiScelti));

figure;
scatter(1:l,trainData(datiScelti,featureScelta),[],trainLabels(datiScelti));
hline(tresholdAccuracy(maxIndex,1));

figure;
plot(tresholdAccuracy(:,1),tresholdAccuracy(:,2))



%% CICLO A MANO CLASS ERROR:

close all;
clc;

datiScelti = [1:597];
featureScelta = 700;

l = length(trainData(datiScelti,700));


[~,~,~,classError,minVal,minIndex] = tresholdPlotter(trainData(datiScelti,featureScelta),trainLabels(datiScelti));

figure;
scatter(1:l,trainData(datiScelti,featureScelta),[],trainLabels(datiScelti));
hline(tresholdAccuracy(minIndex,1));

figure;
plot(classError(:,1),classError(:,2))



%% treshold con una feature

bestFeature = 1514;

[tresholdAccuracy,maxValue,maxIndex] = tresholdPlotter(trainData(:,bestFeature),trainLabels);

% figure;
% scatter(tresholdAccuracy(:,1),tresholdAccuracy(:,2));

foundedTreshold = tresholdAccuracy(maxIndex,1);

figure;
scatter(1:nObservations,trainData(:,bestFeature),[],trainLabels);




%% histogram per capire

% /!\ ATTENZIONE: cicla e genera un sacco di histograms 

% carini: 480 - 490, 440 - 450, 710 - 720, 580-590
fromTo = 680:700;
n_beans = 20;

for i = fromTo
    figure;
    hold on;
    histogram(trainCorrectData(:,i),n_beans,'FaceColor','g');
    histogram(trainErrorData(:,i),n_beans,'FaceColor','r');
    title(strcat('Histogram alla feature: ',num2str(i)));
end


%% plotmatrix per capire

% /!\ ATTENZIONE: cicla e genera un sacco di plotmatrix 

fromTo = 70:73; % fattore 10 di moltiplicazione (0:80) = (0:800)

for i = fromTo
    lRange = i*10; % low range
    tRange = (i+1)*10;
    setGlobalRanges(lRange,tRange);
    figure;
    gplotmatrix(trainData(:,usefulFeaturesRange),[],trainLabels,'br');
    title(strcat('Matrix da ',num2str(lRange),' a ',num2str(tRange)));
end


%% k - means in 3d

trainDataFeaturesMultidimension = [trainData(:,590),trainData(:,712),trainData(:,445)]

k = 2;
idx = kmeans(trainDataFeaturesMultidimension, k, 'distance','sqEuclidean', 'start','sample');

%%

idx = trainData(:,bestFeature) > foundedTreshold;

res = idx == trainLabels;

totGiusti = sum(res)

percGiusti = totGiusti / nObservations
% non abbastanza XD


%% media somma e std

setGlobalRanges(500,800);

meanErrorData = mean(trainErrorData(:,usefulFeaturesRange)')';
meanCorrectData = mean(trainCorrectData(:,usefulFeaturesRange)')';

distanzaMediaCorrect = (trainCorrectData(:,usefulFeaturesRange)-meanCorrectData);
distanzaMediaError   = (trainErrorData(:,usefulFeaturesRange)-meanErrorData);

sommaCorrectData =  sum(trainCorrectData(:,usefulFeaturesRange)')';
sommaErrorData   =  sum(trainErrorData(:,usefulFeaturesRange)')';


%%
plot(usefulFeaturesRange,distanzaMediaCorrect,'g');
hold on;
plot(usefulFeaturesRange,distanzaMediaError,'r');

%%
figure;
plot(usefulFeaturesRange,trainCorrectData(3,usefulFeaturesRange),'b');
figure;
plot(usefulFeaturesRange,trainErrorData(3,usefulFeaturesRange),'r');




