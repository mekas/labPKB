using Random, DataFrames, Statistics

function euclid_distance(X, X2)
    #=
    assume both parameters are vectors
    assume the column signify a feature
    assume the row signify the average of a 
    feature for a certain class 
    =#
    
    #assert both X and X2 have same size
    subtracted_vector = X - X2
    power_vector = subtracted_vector .^ 2
    sum_val = sum(power_vector)
    return sqrt(sum_val)
end

#=
Function to split dataframe into two distinct set 
with equal proportion
=#
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

function generate_row(df, num_feature, i, step)
    new_df = DataFrame()
    for k in 1:num_feature
        # now we need to seriously think how to mod dataframe
        # push!(data[i,k])  
        # now move the data randomly by 0.1 points
        rand_value = rand(Float64) * step
        random_step = df[i,k] + (rand_value - 1/2 * rand_value)
        new_df = insertcols(new_df, size(new_df)[2]+1, features[k] => random_step)
    end
    new_df = insertcols(new_df, size(new_df)[2]+1, features[5] => df[i, 5])
    return new_df
end

function generate_more_dataset(df, max_iter, step)
    # we need an outer loop to generate until satisfaction achieved
    for _ in 1:max_iter
        num_instance = size(df)[1]
        for i in 1:num_instance
            # next in order to balanced the class generate based on class proprotion
            # assume we do full sampling instead of half sampling
            # we move for each instance move the data randomly to 0.01 points
            # for that we need to loop features
            
            new_df = faster_generate_row(df, num_feature, i, step)
            insert!(df, size(df)[1] + 1 , new_df[1, :], promote=true)
            
            #end
        end
        # next merge the data to original one
    end
    return df;
end

# after this function was called, the class is stripped from the dataset
# but the dataset is arranged into vector where the indices identify the class
function split_dataframe_by_class(df, class, num_feature)
    # loop class
    V = []
    for i in eachindex(class)
        # filter dataframe by class
        class_index = df[:,end] .== class[i]
        # we discard the class column
        ndf = df[class_index, 1:num_feature]
        push!(V, Matrix{Float16}(ndf))
    end
    return V
end

function faster_generate_row(df, num_instance, num_feature, step)
    # generate vector 
    half = Float16(0.5)
    rand_value = rand(Float16, (num_instance, num_feature)) .* step
    new_vector = df .+ (rand_value .- (half .* step))
    
    return new_vector
end

# this function supposedly run only once
function faster_generate_dataset(df_list, class, step, max_iter)
    # now let's introduce sampling rate, we are doing sampling rate sample
    # sampling rate must be of value [0.1, 1, step=0.1]
    m = size(df_list[1])[2]

    # loop for each class
    for c=1:length(class)
        # keep track of current size
        num_instance = size(df_list[c])[1]

        new_df = zeros(Float16, num_instance*max_iter, m)
        # println(size(new_df))
        # initially fill with matrix of df_list 
        df = df_list[c]
        # get current df (vector)
        for iter = 0:max_iter-1
            low_index = num_instance * iter + 1
            up_index = num_instance * (iter + 1)
            df = faster_generate_row(df, num_instance, num_feature, step)    
            new_df[low_index:up_index, :] = df
        end
        #print(typeof(new_df))
        df_list[c] = vcat(df_list[c], new_df)
        # next append class info as integer as new column to df
        class_info = ones(Float16, size(df_list[c])[1]) * c
        df_list[c] = hcat(df_list[c], class_info)
    end
    df = to_matrix(df_list, class)
    return df
end

# unused
function generate_data(df_list, class, step, max_iter)
    for _ in 1:max_iter
        df_list = faster_generate_dataset(df_list, class, step, max_iter)
    end
end

function to_matrix(df_list, class)
    # prepare empty matrix of size equal to df_list 
    pmat = zeros(Float16,0,size(df_list[1])[2]) 

    for i=1:length(class)
        (num_instance, num_feature) = size(df_list[i])
        mat = zeros(num_instance, num_feature)
        mat = df_list[i]
        pmat = vcat(pmat, mat)
    end
    return pmat
end
