%Author: Mahmoud Afifi

clear all
clc
training_imgs_dir = fullfile('..','train_t2');
hist_bin = 224;

if exist('training_data_hist_classification.mat','file') == 0
    
    names = dir(fullfile(training_imgs_dir,'*.png'));
    names = {names(:).name};
    n = length(names);
    training_hists = zeros(hist_bin,hist_bin,3,n);
    training_gts = zeros(n,1);
    fprintf('preparing training data...\n');
    for ind = 1 : n
        fprintf('processing iamge %d/%d\n',ind,n);
        I = im2double(imread(fullfile(training_imgs_dir,names{ind})));
        temp = get_RGB_uv_hist(I,hist_bin);
        temp(isnan(temp)) = 0;
        training_hists(:,:,:,ind) = temp;
        parts = strsplit(names{ind},'_');
        ti = str2num(parts{2});
        training_gts(ind,:) = ti;
    end
    data.training_hists = training_hists;
    data.training_gts = training_gts;
    clear training_hists training_gts
    save(fullfile('training_data_hist_classification.mat'),'data','-v7.3');
end

resnet_depth = 18;

if exist('deep_learning_models','dir') == 0
    mkdir('deep_learning_models');
end

load('training_data_hist_classification.mat');
training_hists = data.training_hists;
training_gts= data.training_gts;
    
if exist('folds.mat','file') == 0
    cvIndices = crossvalind('Kfold',size(training_hists,4),10);
    save(fullfile('folds.mat'),'cvIndices');
else
    load('folds.mat');
end

inputSize = [224, 224, 3];


data.training_gts = categorical(data.training_gts');

data.training_hists = data.training_hists * 100;

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
    
    valLabels = data.training_gts(testing_inds)';
    trLabels = data.training_gts(training_inds)';
    
    tr_hists = data.training_hists(:,:,:,training_inds);
    val_hists = data.training_hists(:,:,:,testing_inds);
    
    if resnet_depth == 18
        net = resnet18;
        net = layerGraph(net);
        fc5 = fullyConnectedLayer(5,'Name', 'f5');
        output = classificationLayer('Name','ClassificationLayer_predictions');
        net = replaceLayer(net, 'fc1000', fc5);
        net = replaceLayer(net, 'ClassificationLayer_predictions', output);
    elseif resnet_depth == 50
        net = resnet50;
        net = layerGraph(net);
        fc5 = fullyConnectedLayer(5,'Name', 'f5');
        output = classificationLayer('Name','ClassificationLayer_predictions');
        net = replaceLayer(net, 'fc1000', fc5);
        net = replaceLayer(net, 'ClassificationLayer_fc1000', output);
    end
    
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',14, ...
        'MaxEpochs',30, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',150, ...
        'Verbose',false, ...
        'ValidationData',{val_hists, valLabels},...
        'Plots','training-progress');
    
    net = trainNetwork(tr_hists, trLabels ,net,options);
    
    save(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_%d.mat',resnet_depth, f)),...
        'net','-v7.3');
    
end

