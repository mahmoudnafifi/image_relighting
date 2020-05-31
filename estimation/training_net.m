% prepare training/validation data
if exist('deep_learning_models','dir') == 0
    mkdir('deep_learning_models');
end

if exist('results_validation','dir') == 0
    mkdir('results_validation');
end

imgs = dir(fullfile('..','training','*.png'));
imgs = {imgs(:).name};
labels = zeros(length(imgs),1);
inputSize = [227, 227, 3];
for i = 1 : length(imgs)
    parts = strsplit(imgs{i},'_');
    gt = strrep(parts{3},'.png','');
    switch gt
        case 'N'
            fi = 0;
        case 'NE'
            fi = 45;
        case 'E'
            fi = 90;
        case 'SE'
            fi = 135;
        case 'S'
            fi = 180;
        case 'SW'
            fi = 225;
        case 'W'
            fi=270;
        case 'NW'
            fi=315;
            
    end
    labels(i) = fi;
end

labels = categorical(labels);

load('folds.mat');

for f = 1 : 3
    training_folds = setdiff([1:3],f);
    testing_fold = f;
    
    %generate a model for current training folds
    training_inds  = [];
    for i = 1 : 2
        training_inds = [training_inds; ...
            find(cvIndices==training_folds(i))];
    end
    testing_inds = find(cvIndices == testing_fold);
    
    trImgs = fullfile('..','training',imgs(training_inds));
    valImgs = fullfile('..','training',imgs(testing_inds));
    
    valLabels = labels(testing_inds);
    trLabels = labels(training_inds);
    
    
    imdsTrain = imageDatastore(trImgs);
    imdsTrain.Labels = trLabels;
    
    imdsValidation = imageDatastore(valImgs);
    imdsValidation.Labels = valLabels;
    
    net = squeezenet;
    net = layerGraph(net);
    conv10 = convolution2dLayer(1,8,'Stride',1,'Name','conv10', ...
        'WeightLearnRateFactor', 10,'BiasLearnRateFactor', 10);
    output = classificationLayer('Name','ClassificationLayer_predictions');
    net = replaceLayer(net, 'conv10', conv10);
    net = replaceLayer(net, 'ClassificationLayer_predictions', output);
    
    
    
    
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    
    
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',14, ...
        'MaxEpochs',30, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    net = trainNetwork(augimdsTrain,net,options);
    
%     [YPred,scores] = classify(net,augimdsValidation);
%     
%     YPred = str2double(char((YPred)));
%     valLabels = str2double(char((valLabels)));
%     
%     results.MAE = mean(abs(YPred(:) - valLabels(:)));
%     results.MAE = mean((YPred(:) - valLabels(:)).^2);
    
    save(fullfile('deep_learning_models',sprintf('deep_model_%d.mat',f)),'net','-v7.3');
%     save(fullfile('results_validation',sprintf('fi_fold_%d.mat',f)),'results','-v7.3');
end