using Statistics
using Serialization

include("average.jl")
include("evaluate.jl")
include("classify.jl")
include("distance.jl")

function cascade_classify(df, batch_size)
    class = unique(df[:,end])
    class_index = size(df)[2]

    col_size = size(df)[2]
    feature_size = col_size-1

    # df_sample = better_split(df, 0.00001) 
    mu_vec = compute_mu_v2(df, class, feature_size, class_index)
    # 2nd phase loop over column vector once more to estimate feature distinctiveness 
    preds = classify_all_d1(df, mu_vec, feature_size, batch_size)
    truth = df[:, class_index]
    return (truth, preds)
end

# iterate over all instance then run classify_by_distance
function classify_all_d1(X, mu_vec, feature_size, batch_size=100)
    # prepare vector to store result
    result = zeros(Float16, size(X)[1], feature_size)
    rounding_limit = Int(floor(size(X)[1]/batch_size))
    for chunk=1:batch_size:rounding_limit*batch_size
        vec = X[chunk:chunk + batch_size - 1, 1:feature_size]
        pred = classify_by_distance_features(vec, mu_vec)
        result[chunk: chunk + batch_size - 1, :] = pred
    end

    # post processing
    prev_index = rounding_limit*batch_size
    vec = X[prev_index:end, 1:feature_size]
    result[prev_index:end,:] = classify_by_distance_features(vec, mu_vec)
    return result
end

function classify_by_distance_features(X, mu)
    # X is now passed as vector of [nrow, nfeature]
    # mu is vector of [1, nfeature, nclass] but we will transform it as [nrow, nfeature, nclass]
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer = [num_instance, 1, 1])
    dist_vec = vectorized_d1_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @.get_min_index(min_vector)
    return min_index
end

function get_min_index(X)
    return X[3]
end

path = "data_9m.mat"

df = deserialize(path)

class = unique(df[:,end])
class_index = size(df)[2]

col_size = size(df)[2]
feature_size = col_size-1
class_size = length(class)
num_instance = size(df)[1]
batch_size = Int(floor(num_instance / 2000))

truths, preds = @time cascade_classify(df, batch_size)
gpreds = @time classify_v2(df)


