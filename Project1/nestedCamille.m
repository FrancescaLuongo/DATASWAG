%% LOADING DATA & VARS

loadingInVars_pietro;

%% yo
k_fold_outer = 10;
k_fold_inner = 10;

[outer_test_class_errors, outer_train_class_errors] = ForwardNestedCrossValidation(k_fold_outer, ...
    k_fold_inner, trainData, trainLabels);