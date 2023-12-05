function confusion_matrix(Y,P)
    num_class = length(unique(Y))
    # cast all data to int 
    Y = Int8.(Y)
    P = Int8.(P)
    # create empty matrix of size (num_class, num_class)
    cnf = zeros(Int32, num_class, num_class)
    num_instance = length(Y)
    # loop over all label
    @simd for i=1:num_instance
        @inbounds @fastmath cnf[Y[i], P[i]] += 1
    end
    return cnf
end

function confusion_matrix_v2(Y,P)
    num_class = length(unique(Y))
    # cast all data to int 
    Y = Int8.(Y)
    P = Int8.(P)
    # create empty matrix of size (num_class, num_class)
    cnf = zeros(Int32, num_class, num_class)

    # prepare cache to store search space for Y & passed
    y_vec = Vector()
    p_vec = Vector()

    # build the cache
    for i=1:num_class
        push!(y_vec,Y.==i)
        push!(p_vec,P.==i)
    end

    #loop over all class
    @simd for i=1:num_class
        @simd for j=1:num_class
            bitvec = y_vec[i] .& p_vec[i]
            cnf[i,j]= sum(bitvec[1])
        end
    end
    return cnf
end

function boolean_and(x,y)
    return x && y
end

function true_correctness(conf_mat)
    correctness = zeros(Float32, size(conf_mat)[1])
    for i=1:size(conf_mat)[1]
        correctness[i] = conf_mat[i,i] / sum(conf_mat[:,i])
    end
    return correctness
end

function measure_corretness(truths, gpreds, preds)
    # compute global correctness for gpreds 
    valuation = confusion_matrix(truths, gpreds)
    gcorrectness_vector = true_correctness(valuation)
    
    # compute correctness for each feature in preds
    _, nc = size(preds)
    for i=1:nc
        valuation = confusion_matrix(truths, preds[:,i])
        current_correctness = true_correctness(valuation)
    end
end