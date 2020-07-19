%Author: Mahmoud Afifi

mode = 'classification';

resnet_depth = 18;

training_imgs_dir = fullfile('..','input_aug');

if strcmpi(mode,'regression') == 1
    if exist(fullfile('..','input_224'),'dir') == 0
        mkdir(fullfile('..','input_224'))
    end
    imgs = dir(fullfile(training_imgs_dir,'*.png'));
    imgs = {imgs(:).name};
    for i = 1 : length(imgs)
        imwrite(imresize(imread(fullfile(training_imgs_dir,imgs{i})),...
            [224, 224]), fullfile('..','input_224',imgs{i}));
    end
end


% prepare training/validation data
if strcmpi(mode,'regression') == 1
    if exist('deep_learning_models_r','dir') == 0
        mkdir('deep_learning_models_r');
    end
elseif strcmpi(mode,'classification') == 1
    if exist('deep_learning_models','dir') == 0
        mkdir('deep_learning_models');
    end
end

if strcmpi(mode,'classifiation') == 1
    
    imgs = dir(fullfile(training_imgs_dir,'*.png'));
    
else
    
    imgs = dir(fullfile('..','input_224','*.png'));
    
end

imgs = {imgs(:).name};

if aug == 1
    if exist('folds_aug.mat','file') == 0
        cvIndices = crossvalind('Kfold',length(imgs),10);
        save(fullfile('folds_aug.mat'),'cvIndices');
    else
        load('folds_aug.mat');
    end
else
    if exist('folds.mat','file') == 0
        cvIndices = crossvalind('Kfold',length(imgs),10);
        save(fullfile('folds.mat'),'cvIndices');
    else
        load('folds.mat');
    end
end

labels = zeros(length(imgs),1);
inputSize = [224, 224, 3];
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

if aug == 1
    load('folds_aug.mat');
else
    load('folds.mat');
end

if strcmpi(mode,'classification')
    labels = categorical(labels);
    
    
    for f = 1 : 10
        training_folds = setdiff([1:3],f);
        testing_fold = f;
        
        %generate a model for current training folds
        training_inds  = [];
        for i = 1 : 2
            training_inds = [training_inds; ...
                find(cvIndices==training_folds(i))];
        end
        testing_inds = find(cvIndices == testing_fold);
        
        trImgs = fullfile(training_imgs_dir,imgs(training_inds));
        valImgs = fullfile(training_imgs_dir,imgs(testing_inds));
        
        
        valLabels = labels(testing_inds);
        trLabels = labels(training_inds);
        
        if aug == 1
            vinds = randperm(length(testing_inds));
            valLabels = valLabels(vinds(1:max(length(testing_inds),300)));
            valImgs = valImgs(vinds(1:max(length(testing_inds),300)));
        end
        imdsTrain = imageDatastore(trImgs);
        imdsTrain.Labels = trLabels;
        
        
        
        
        imdsValidation = imageDatastore(valImgs);
        imdsValidation.Labels = valLabels;
        
        if resnet_depth == 18
            net = resnet18;
            net = layerGraph(net);
            fc8 = fullyConnectedLayer(8,'Name', 'f8');
            output = classificationLayer('Name','ClassificationLayer_predictions');
            net = replaceLayer(net, 'fc1000', fc8);
            net = replaceLayer(net, 'ClassificationLayer_predictions', output);
        elseif resnet_depth == 50
            net = resnet50;
            net = layerGraph(net);
            fc8 = fullyConnectedLayer(8,'Name', 'f8');
            output = classificationLayer('Name','ClassificationLayer_predictions');
            net = replaceLayer(net, 'fc1000', fc8);
            net = replaceLayer(net, 'ClassificationLayer_fc1000', output);
        end
        
        
        pixelRange = [-50 50];
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
            'ValidationFrequency',150, ...
            'Verbose',false, ...
            'Plots','training-progress');
        
        net = trainNetwork(augimdsTrain,net,options);
        
        save(fullfile('deep_learning_models',sprintf('deep_resnet_model_%d_%d.mat',resnet_depth, f+5)),'net','-v7.3');
        
    end
elseif strcmpi(mode,'regression')
    
    
    for f = 1 : 10
        training_folds = setdiff([1:3],f);
        testing_fold = f;
        
        %generate a model for current training folds
        training_inds  = [];
        for i = 1 : 2
            training_inds = [training_inds; ...
                find(cvIndices==training_folds(i))];
        end
        testing_inds = find(cvIndices == testing_fold);
        
        trImgs = fullfile('..','input_224',imgs(training_inds));
        valImgs = fullfile('..','input_224',imgs(testing_inds));
        
        valLabels = labels(testing_inds);
        trLabels = labels(training_inds);
        
        if aug == 1
            vinds = randperm(length(testing_inds));
            valLabels = valLabels(vinds(1:max(length(testing_inds),300)));
            valImgs = valImgs(vinds(1:max(length(testing_inds),300)));
        end
        
        
        if resnet_depth == 18
            net = resnet18;
            net = layerGraph(net);
            fc1 = fullyConnectedLayer(1,'Name', 'f1');
            output = regressionLayer('Name','regression_Layer');
            net = replaceLayer(net, 'fc1000', fc1);
            net = removeLayers(net, 'prob');
            net = removeLayers(net, 'ClassificationLayer_predictions');
            net = addLayers(net, output);
            net = connectLayers(net, 'f1', 'regression_Layer');
        elseif resnet_depth == 50
            net = resnet50;
            net = layerGraph(net);
            fc1 = fullyConnectedLayer(1,'Name', 'f1');
            output = regressionLayer('Name','regression_Layer');
            net = replaceLayer(net, 'fc1000', fc1);
            net = removeLayers(net, 'fc1000_softmax');
            net = removeLayers(net, 'ClassificationLayer_fc1000');
            net = addLayers(net, output);
            net = connectLayers(net, 'f1', 'regression_Layer');
        end
        
        
        options = trainingOptions('sgdm', ...
            'MiniBatchSize',14, ...
            'MaxEpochs',30, ...
            'InitialLearnRate',1e-4, ...
            'Shuffle','every-epoch', ...
            'ValidationFrequency',150, ...
            'Verbose',false, ...
            'ValidationData',table(valImgs',valLabels),...
            'Plots','training-progress');
        
        net = trainNetwork(table(trImgs',trLabels),net,options);
        
        save(fullfile('deep_learning_models',sprintf('deep_resnet_model_%d_%d_r.mat',resnet_depth, f)),'net','-v7.3');
        
    end
end