function [performance] = TestRegressionPerformance(Data, FM,comparedRegression, order)
[m,n] = size(Data);
I = ones(m,1); %train X, Y et trainData ont le même nb de colonnes
if order == 1
    XtestOrder = [ I FM ];
end
if order == 2
   XtestOrder = [ I FM FM.^2];
end

   performance = immse(Data, XtestOrder*comparedRegression);
end