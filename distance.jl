using Statistics

function general_euclid_distance(X, X2)
    # replicate X according to mu_vec size
    num_class = size(X2)[1]
    X = repeat(X', outer = num_class)
    subtracted_vector = X .- X2
    power_vector = subtracted_vector .^ 2
    sum_val = sum(power_vector, dims=2)
    dist_vec = sum_val .^ (1/2)
    return dist_vec
end

function vectorized_euclid_distance(X, mu)
    # make X has depth channel of depth num_class
    numclass = size(mu)[3]
    # repeat data vector to channel depth
    X = repeat(X, outer = [1,1,numclass])
    # compute the distance
    subtracted_vector = X .- mu
    power_vector = subtracted_vector .^ 2
    sum_val = sum(power_vector, dims=2)
    dist_vec = sum_val .^ (1/2)
    return dist_vec
end

function vectorized_d1_distance(X, mu)
    # make X has depth channel of depth num_class
    numclass = size(mu)[3]
    # repeat data vector to channel depth
    X = repeat(X, outer = [1,1,numclass])
    # compute the distance
    subtracted_vector = abs.(X .- mu)
    return subtracted_vector
end
