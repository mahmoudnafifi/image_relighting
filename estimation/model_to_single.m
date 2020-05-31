function model = model_to_single(model)
model.features =  single(model.features);
model.encoder.weights = single(model.encoder.weights);
model.encoder.bias = single(model.encoder.bias);
end