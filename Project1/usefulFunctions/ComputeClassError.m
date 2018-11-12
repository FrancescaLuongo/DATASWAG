function[class_error] = ComputeClassError(ratio, true_y, predicted_y)
miss_class_A = GetMissClass(true_y, predicted_y, 0 );
miss_class_B = GetMissClass(true_y, predicted_y, 1 );
class_error = (ratio * miss_class_A / length(find(true_y==0))) + ((1-ratio) * miss_class_B / length(find(true_y==1)));
end