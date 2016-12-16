function [max_index, max_value] = maximum_in_array(array)
    max_index =1;
    max_value = array(max_index);
    for index = 2:length(max_index);
        if array(index) > max_value
            max_value = array(index);
            max_index = index;
        end
    end
end