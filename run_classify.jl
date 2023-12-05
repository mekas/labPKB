using Serialization
using Statistics

include("lib.jl")
include("classify.jl")
include("evaluate.jl")
include("resample.jl")

path = "data_78m.mat"

df = deserialize(path)

class = unique(df[:,end])
class_index = size(df)[2]

col_size = size(df)[2]
feature_size = col_size-1
class_size = length(class)

num_instance = size(df)[1]
batch_size = Int(floor(num_instance / 2000))

preds = classify_v2(df, batch_size)
# println(size(preds))
# display(preds[end-10:end,:])
# println()


valuation = confusion_matrix(preds[:,1], preds[:,2])
display(valuation)
println()
correctness = true_correctness(valuation)
println(correctness)

