function [number_miss_class] = GetMissClass(compare_with, compared, ident)
number_good_classified_ident = length(intersect(find(compared == ident), find(compare_with == ident)));
number_miss_class = length(find(compare_with == ident)) - number_good_classified_ident;
end