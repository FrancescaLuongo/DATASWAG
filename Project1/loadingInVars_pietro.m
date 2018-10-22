clear all; close all; clc;

%% IMPORTING DATA

addpath('usefulFunctions');

testData = load('./data/testSet.mat');
testLabels = load('./data/testLabelEstimation.mat');
trainLabels = load('./data/trainLabels.mat');
trainData = load('./data/trainSet.mat');

testData = testData.testData;
testLabels = testLabels.testLabelEstimation;
trainLabels = trainLabels.trainLabels;
trainData = trainData.trainData;


sizeTest = size(testData);
sizeLabels = size(trainLabels);

nFeatures = size(trainData,2);
nObservations = size(trainData,1);

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
