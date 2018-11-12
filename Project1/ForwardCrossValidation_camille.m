function [train_class_errors, test_class_errors] = ForwardCrossValidation(k_fold, features,labels) 
test_class_errors = [];
train_class_errors = [];
partition = cvpartition(length(labels),'kfold',k_fold);
fun = @(xT,yT,xt,yt) length(yt)*(ComputeClassError(0.5, yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear'), xt)));
opt = statset('Display','iter','MaxIter',100);
[sel,hst] = sequentialfs(fun, features, labels,'cv',partition,'options',opt);
ordrerd_ind_preselected = find(sel==1);

for k = 1:partition.NumTestSets 
    idx = partition.training(k);
    train_idx = find(idx==1);
    test_idx = find(idx==0);
    train_data = features(train_idx,:);
    test_data = features(test_idx,:);
    
    for i = 1:length(ordrerd_ind_preselected)
        preselected_features_train = train_data(:,ordrerd_ind_preselected);
        preselected_features_test = test_data(:,ordrerd_ind_preselected);
        selected_features_train = preselected_features_train(:,1:i);
        selected_features_test = preselected_features_test(:,1:i);
        
        classifier_test = fitcdiscr(selected_features_train, labels(train_idx), 'discrimtype', 'linear', 'Prior' , 'uniform');
    
        y_prediction_train = predict(classifier_test, selected_features_train);
        train_class_errors(i,k) = ComputeClassError(0.5, labels(train_idx), y_prediction_train);
   
        y_prediction_test = predict(classifier_test, selected_features_test);
        test_class_errors(i,k) = ComputeClassError(0.5, labels(test_idx), y_prediction_test);
    end
end
end
