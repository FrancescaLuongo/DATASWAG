%% LOADING DATA & VARS

loadingInVars_pietro;

%% prova particolare XD
stepWidth = 8;
nEEG_Electrodes = 16;
nSingleEEGFeatures = nFeatures/nEEG_Electrodes;
selectedFeatures = [];

currentPoint = 1;
for EEG_electrode = 0:(nEEG_Electrodes-1)
    currentStart = (EEG_electrode*nSingleEEGFeatures+currentPoint);
    currentEnd = currentStart+stepWidth;
    currentRange = currentStart:currentEnd;
    selectedFeatures = [selectedFeatures,currentRange];
end

%%

