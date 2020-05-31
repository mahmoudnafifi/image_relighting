
classdef PCAFeature
    properties
        weights
        bias
    end
    methods
        function feature = encode(obj,hist)
            feature = (reshape(hist,1,[]) - obj.bias') *obj.weights;
        end
    end
end