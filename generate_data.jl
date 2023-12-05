using StatsPlots, RDatasets
using CSV
using Statistics
using Profile
using DataFrames
using Serialization

include("lib.jl")

#iris = dataset("datasets", "iris")
const path = "iris.csv"
df = CSV.read(path, DataFrame)

# split the dataframe into df for each class

# our task is to loop all data by each class
# read unique class from data 
const class = unique(df.class)
# we need to store placeholder, assume we store in placeholder
const features = names(df)

const num_feature = size(df)[2] - 1

# better clone to another dataframe to concat it with the original later
step = Float16(0.1);
num_iter = 2^20 - 1;

# let's split the data per each class
df_collection = @time split_dataframe_by_class(df, class, num_feature)
# print(size(df_collection[1]))

# next we do processing for each data set class to keep balance
df = @time faster_generate_dataset(df_collection, class, step, num_iter)
print(size(df))
# we need an outer loop to generate until satisfaction achieved
# @profview generate_more_dataset(df, 0, 0)
# generate_more_dataset(df, 0,0)
#df = @time generate_more_dataset(df, num_iter, step)
#print(size(df))

#scatter(
#    df[:,1],
#    df[:,3],
#    group = df[:,5],
#    m = (0.5, [:+ :h :star7], 12),
#    bg = RGB(0.2, 0.2, 0.2)
#)  

# serialize("data.mat", df)
 