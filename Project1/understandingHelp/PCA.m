%%
load hald
[coeff,score,latent,~,explained] = pca(ingredients,'VariableWeights','variance');
Xcentered = score*coeff';

%%
biplot(coeff(:,1:3),'scores',score(:,1:3),'varlabels',{'v_1','v_2','v_3','v_4'});