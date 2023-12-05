using StatsPlots, RDatasets
using CSV, DataFrames
using Statistics
using Random

include("lib.jl")

iris = dataset("datasets", "iris")
path = "iris.csv"
data = CSV.read(path, DataFrame)
train, test = split_data(data)

# step 0: we'll split the data into two

# step 1: phase out row based on class of interest
# now we want to detect all class
class = unique(train.class)
# fdata = filter(row -> row[end]=="Iris-virginica" || row[end]=="Iris-setosa", data)

# step 2: iterate for each row 
col_size = size(train)[2]
feature_size = col_size-1
groups = unique(train[:,end])
class_size = size(groups)[1]
# get unique class

#=mu = zeros(class_size, feature_size)
for i=1:col_size-1
    for j=1:class_size
        mu[j,i] = mean(filter(row -> row[end]==groups[j], train)[:,i])
    end
end
=#
indices = data[:,5] .== "Iris-versicolor"
data = data[indices, :]
dist = zeros(feature_size)
for i=1:feature_size
    dist[i] = abs(mu[3,i] - mu[2,i])
end

display(dist)

# best two max columns
idx = sortperm(dist)
max_idx = idx[end-1:end]

# X=data[:, max_idx]
# Y=data[:,end]
# pdata = hcat(X,Y)

scatter(
    data[:,max_idx[1]],
    data[:,max_idx[2]],
    group = data[:,end],
    m = (0.5, [:+ :h :star7], 12),
    bg = RGB(0.2, 0.2, 0.2)
)

