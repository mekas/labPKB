using Statistics

function compute_mu(df, class, feature_size, class_index)
    # find the mu for each class
    mu_vec = zeros(Float16, class_size, feature_size)
    class_size = length(class)
    for i = 1:class_size
        c = class[i]
        current_class_pos = (df[:, class_index] .- c) .< Float16(0.1)
        current_df = df[current_class_pos,1:class_index-1]
        mu = zeros(feature_size)
        # mu = mean(current_df, dims=1)
        for j = 1:feature_size
            av = average(current_df[:, j])
            mu[j] = av
        end
        #print(current_df)
        # print(size(mu))
        mu_vec[i,:] = mu
    end
    return mu_vec
end

function compute_mu_v2(df, class, feature_size, class_index)
    class_size = length(class)
    mu_vec = zeros(Float16, 1, feature_size, class_size)
    for i = 1:class_size
        c = class[i]
        current_class_pos = (df[:, class_index] .- c) .< Float16(0.1)
        current_df = df[current_class_pos,1:class_index-1]
        current_df = Float32.(current_df)
        mu = mean(current_df, dims = 1)
        mu_vec[1,:,i] = mu
    end
    return mu_vec
end

function fast_compute_mu(df, class, feature_size, class_index, batch_size)
    class_size = length(class)
    mu_vec = zeros(Float16, 1, feature_size, class_size)

    for i = 1:class_size
        c = Int16(class[i])
        current_class_pos = (df[:, class_index] .- c) .< Float16(0.1)
        current_df = df[current_class_pos,1:class_index-1]
        # loop over all data in batch_size
        for j = 1:feature_size
            av = fast_average(current_df[:, j], batch_size)
            mu_vec[1, j, c] = av
        end
        
    end
    return mu_vec
end

function fast_average(vec, batch_size)
    rounding_limit = Int(floor(length(vec)/batch_size))
    chunks = zeros(rounding_limit)
    i=1
    for j=1:batch_size:rounding_limit*batch_size
        chunk = vec[j:j + batch_size - 1]
        chunks[i] = mean(chunk)
        i+=1
    end
    av = mean(chunks)
    return av
end

function compute_mu_manual(df, class, feature_size, class_index)
    class_size = length(class)
    mu_vec = zeros(Float16, 1, feature_size, class_size)
    for i = 1:class_size
        c = class[i]
        current_class_pos = (df[:, class_index] .- c) .< Float16(0.1)
        current_df = df[current_class_pos,1:class_index-1]
        mu = zeros(feature_size)
        for j = 1:feature_size
            av = average(current_df[:, j])
            mu[j] = av
        end
        mu_vec[1,:,i] = mu
    end
    return mu_vec
end

# given current vector, compute average by recurse before overflow
function average(vec)
    vec_len = length(vec)
    mu = mean(vec)
    # print(mu)
    # print(mu !== NaN)
    if (!isnan(mu) && !isinf(mu))
        return mu
    else
        # divide vector by 2
        mid = Int(round(vec_len/2))
        v1 = average(vec[1:mid])
        v2 = average(vec[mid+1:end])
        # println(vec_len)
        return mean([v1, v2])
    end
end

function closer_average(vec)
    vec_len = length(vec)
    total = sum(vec)
    #if (!isnan(total) && !isinf(total))
    if(vec_len < 10000)
        return (total/vec_len, vec_len)
    else
        # divide vector by 2
        mid = Int(round(vec_len/2))
        (av1, len1) = closer_average(vec[1:mid])
        (av2, len2) = closer_average(vec[mid+1:end])
        slen = len1 + len2
        av = ((len1/slen)*av1) + ((len2/slen)*av2)
        return (av, slen)
    end
end