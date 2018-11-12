function[outer_test_class_errors, outer_train_class_errors] = ForwardNestedCrossValidation(k_fold_outer, ...
    k_fold_inner, features, labels)
    N = length(labels);
    outer_cv_partition = cvpartition(N,'kfold',k_fold_outer);
    outer_partition = repartition(outer_cv_partition);

    for k_out = 1:outer_partition.NumTestSets 
        outer_idx = outer_partition.training(k_out);
        outer_train_idx = find(outer_idx==1);
        outer_test_idx = find(outer_idx==0);
        outer_train_data = features(outer_train_idx,:);
        outer_test_data = features(outer_test_idx,:);
        outer_labels_train = labels(outer_train_idx);
        outer_labels_test = labels(outer_test_idx);

        inner_cv_partition = cvpartition(length(outer_labels_train),'kfold',k_fold_inner);
        inner_partition = repartition(inner_cv_partition);

        fun = @(xT,yT,xt,yt) length(yt)*(ComputeClassError(0.5, yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear'), xt)));
        opt = statset('Display','iter','MaxIter',100);
        [sel,hst] = sequentialfs(fun, preselected_features, outer_labels_train,'cv',inner_cv_partition,'options',opt);
        optimal_validation_error = hst.Crit(end)
        ordrerd_ind_preselected = find(sel==1);
        selected_features_train = preselected_features(:,ordrerd_ind_preselected);
        selected_features_test = outer_test_data(:,ordrerd_ind_preselected);

        classifier_test = fitcdiscr(selected_features_train, outer_labels_train, 'discrimtype', 'linear', 'Prior' , 'uniform');

        y_prediction_train = predict(classifier_test, selected_features_train);
        outer_train_class_errors(k_out) = ComputeClassError(0.5, outer_labels_train, y_prediction_train);

        y_prediction_test = predict(classifier_test, selected_features_test);
        outer_test_class_errors(k_out) = ComputeClassError(0.5, outer_labels_test, y_prediction_test);
    end
end