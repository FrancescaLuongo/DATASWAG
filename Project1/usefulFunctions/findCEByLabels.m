function classError = findCEByLabels(labels,predictions)
%findCEByLabels Summary of this function goes here
%   Detailed explanation goes here

nGroup0 = sum(~labels);
nGroup1 = sum(labels);

nGiusti0 = sum(predictions.*(~labels));
nGiusti1 = sum(~predictions.*(labels));

classError = 1/2*(nGroup0-nGiusti0)/nGroup0 ...
    +  1/2*(nGroup1-nGiusti1)/nGroup1;

end

