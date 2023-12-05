using Statistics

include("distance.jl")
include("average.jl")

function classify_by_distance(X, X2)
    dist_vec = general_euclid_distance(X, X2)
    min_index = argmin(dist_vec)[1]
    return min_index
end

# iterate over all instance then run classify_by_distance
function classify_all(X, mu_vec, feature_size)
    # prepare vector to store result
    result = zeros(Float16, size(X)[1])
    for i=1:size(X)[1]
        vec = X[i, 1:feature_size]
        result[i] = classify_by_distance(vec, mu_vec)
    end
    return result
end

function classify_by_distance_v2(X, mu)
    # X is now passed as vector of [nrow, nfeature]
    # mu is vector of [1, nfeature, nclass] but we will transform it as [nrow, nfeature, nclass]
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer = [num_instance, 1, 1])
    dist_vec = vectorized_euclid_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @.get_min_index(min_vector)
    return min_index
end

function get_min_index(A)
    return A[3]
end

# iterate over all instance then run classify_by_distance
function classify_all_v2(X, mu_vec, feature_size, batch_size=100)
    # prepare vector to store result
    result = zeros(Float16, size(X)[1])
    rounding_limit = Int(floor(size(X)[1]/batch_size))
    for chunk=1:batch_size:rounding_limit*batch_size
        vec = X[chunk:chunk + batch_size - 1, 1:feature_size]
        pred = classify_by_distance_v2(vec, mu_vec)
        result[chunk: chunk + batch_size - 1] = pred
    end
    # post processing
    prev_index = rounding_limit*batch_size
    vec = X[prev_index:end, 1:feature_size]
    result[prev_index:end] = classify_by_distance_v2(vec, mu_vec)

    return result
end

function classify(df)
    class = unique(df[:,end])
    class_index = size(df)[2]

    col_size = size(df)[2]
    feature_size = col_size-1
    class_size = length(class)

    mu_vec = compute_mu(df, class_size, feature_size, class_index)
    preds = classify_all(df, mu_vec, feature_size)
    return hcat(df, preds)
end

function classify_v2(df, batch_size)
    class = unique(df[:,end])
    class_index = size(df)[2]

    col_size = size(df)[2]
    feature_size = col_size-1
    
    # df_sample = better_split(df, 0.00001) 
    mu_vec = @time compute_mu_v2(df, class, feature_size, class_index)

    # mu_vec = @time fast_compute_mu(df, class, feature_size, class_index, batch_size)
    preds = @time classify_all_v2(df, mu_vec, feature_size, batch_size)
    df = hcat(df[:,class_index], preds)
    return df
end