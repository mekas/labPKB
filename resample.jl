using Statistics

function better_split(X, sample_rate=0.01)
    # assume X of size num_instance, num_feature+class
    # sample rate is floating16 [0,1]
    (num_instance, num_col) = size(X)
    class = unique(X[:,num_col])
    num_class = length(class)
    # prepare storage to store the resulting sample
    num_sample_instance = Int64(floor(sample_rate*num_instance/num_class))
    X_sample = zeros(Float16, num_sample_instance, num_col, num_class)
    # loop data per class 
    for c=1:num_class
        # take data for the current_class 
        current_data_index = X[:,num_col].==class[c] 
        current_x = X[current_data_index, :]
        current_x_size = size(current_x)[1]
        # shuffle the data
        shuffle_indices = shuffle(1:current_x_size)
        cut_index = Int64(floor(sample_rate * current_x_size))
        x_shuffled = current_x[shuffle_indices[1:cut_index],:]
        # this_instance_size = size(x_shuffled)[1] 
        # store into x_sample
        X_sample[:,:,c] = x_shuffled
        #display(x_shuffled[1:10,:])
    end
    # before return reshape X_sample to original dims
    Xc = custom_reshape(X_sample)
    #display(Xc[end-10:end,:])
    return Xc
end

function custom_reshape(X)
    (num_instance, num_col, num_class) = size(X)
    Xc = zeros(num_instance*num_class, num_col)
    for c=1:num_class
        low = 1 + num_instance*(c-1)
        high = c*num_instance
        Xc[low:high,:] = X[:,:,c]
    end
    
    return Xc
end

function split_data(X)
    dims = size(X)
    nrow = dims[1]
    
    shuffle_indices = shuffle(1:nrow)
    midcut_index = convert(Int64, floor(nrow/2))
    lower_shuffle = shuffle_indices[1:midcut_index]
    upper_shuffle = shuffle_indices[midcut_index+1:end]
    train = X[lower_shuffle,:]
    test =  X[upper_shuffle,:]
    # final checking to assert split data contain the similar amount of class
    if(are_split_allowed(data, train, test))
        return (train, test)
    else
        # recursively call split_data until balanced
        return split_data(X)
    end
end

# this function expect a vector of 1 dim
function check_class_count(X)
    nclass = size(unique(X[:,end]))[1]
    return nclass 
end

function are_split_allowed(X, Y, Z)
    n0 = check_class_count(X)
    n1 = check_class_count(Y)
    n2 = check_class_count(Z)
    if(n0 == n1 == n2)
        return true
    end

    return false
end
