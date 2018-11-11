function classError = findCEByTreshold(selectedData,correctData,treshold)
%TRESHOLDPLOTTER Summary of this function goes here
%   Detailed explanation goes here

% find CE By treshold


idx = selectedData>treshold;

nGroup0 = sum(~correctData);
nGroup1 = sum(correctData);

nGiusti0 = sum(idx.*(~correctData));
nGiusti1 = sum(~idx.*(correctData));

classError = 1/2*(nGroup0-nGiusti0)/nGroup0 ...
    +  1/2*(nGroup1-nGiusti1)/nGroup1;


end

