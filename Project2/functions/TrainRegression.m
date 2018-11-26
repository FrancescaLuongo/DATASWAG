function [regression, performance] = TrainRegression(Data, FM, order)
[m,n] = size(Data);
I = ones(m,1); %train X, Y et trainData ont le même nb de colonnes
if order == 1
    XOrder = [ I FM ];
end
if order == 2
   XOrder = [ I FM FM.^2];
end

   regression = regress(Data, XOrder);
   performance = immse(Data, XOrder*regression);
end