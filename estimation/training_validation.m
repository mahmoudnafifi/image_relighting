clear all
clc
testing_only = 1;

hist_bin = 91;
feature_length = 191;
K = 5;
if exist('data_driven_models','dir') == 0
    mkdir('data_driven_models');
end

if exist('results_validation','dir') == 0
    mkdir('results_validation');
end

image_base = fullfile('..','training');
names = dir(fullfile(image_base,'*.png'));
names = {names(:).name};

if exist('folds.mat','file') == 0
    cvIndices = crossvalind('Kfold',length(names),3);
    save(fullfile('folds.mat'),'cvIndices');
else
    load('folds.mat');
end
for f = 3 : 3
    %% training data
    disp('Preparing training data ...');
    fprintf('Current testing fold is %d.\n',f);
    
    training_folds = setdiff([1:3],f);
    testing_fold = f;
    
    %generate a model for current training folds
    training_inds  = [];
    for i = 1 : 2
        training_inds = [training_inds; ...
            find(cvIndices==training_folds(i))];
    end
    testing_inds = find(cvIndices == testing_fold);
    if testing_only == 0
        n = length(training_inds);
        training_hists = zeros(hist_bin*hist_bin*3,n);
        
        training_gts = zeros(n,1);
        for i = 1 : n
            ind = training_inds(i);
            I = im2double(imread(fullfile('..','training',names{ind})));
            temp = get_RGB_uv_hist(I,hist_bin);
            temp = temp(:);
            temp(isnan(temp)) = 0;
            training_hists(:,i) = temp;
            parts = strsplit(names{ind},'_');
            Ti = str2num(parts{2});
            switch Ti
                case 2500
                    ti = 0;
                case 3500
                    ti = 0.25;
                case 4500
                    ti = 0.5;
                case 5500
                    ti = 0.75;
                case 6500
                    ti = 1;
            end
            training_gts(i,:) = ti;
        end
        
        b = mean(training_hists,2);
        training_hists_=training_hists-repmat(b,[1,n]); %subtract mean
        training_hists_=training_hists_/sqrt(n); %divide by the sqrt(number of points)
        disp('calculating the singular value decomposition...');
        %calculate the singular value decomposition
        [W,~,~] = svd(training_hists_,'econ');
        feature_length = min(feature_length,size(W,2));
        W = W(:,1:feature_length);
        PCAencoder = PCAFeature;
        PCAencoder.weights = W;
        PCAencoder.bias = b;
        
        model = ill_est_model;
        model.encoder = PCAencoder;
        
        model.K = K;
        model.temps = training_gts;
        
        features = zeros(n,feature_length);
        for j = 1 : n
            features(j,:) = PCAencoder.encode(training_hists(:,j));
        end
        
        
        model.features = features;
        model = model_to_single(model);
        
        save(fullfile('data_driven_models',sprintf('model_%d.mat', ...
            f)),'model','-v7.3');
    else
        load(fullfile('data_driven_models',sprintf('model_%d.mat', f)));
        model.K = 5;
    end
    %% validation
    
    disp('Testing...')
    
    
    testing_images = names(testing_inds);
    responses = zeros(length(testing_images),1);
    gts = zeros(length(testing_images),1);
    nn = length(testing_images);
    for t = 1 : nn
        I_testing = im2double(imread(...
            fullfile('..','training',testing_images{t})));
        responses(t) = model.estimate_ill(I_testing);
        parts = strsplit(testing_images{t},'_');
        Ti = str2num(parts{2});
        switch Ti
            case 2500
                ti = 0;
            case 3500
                ti = 0.25;
            case 4500
                ti = 0.5;
            case 5500
                ti = 0.75;
            case 6500
                ti = 1;
        end
        gts(t) = ti;
       
    end
    
    error.MAE = mean(abs(gts(1:length(responses)) - responses(:)));
    error.MSE = mean((gts(1:length(responses))-responses(:)).^2);
    save(fullfile('results_validation',sprintf('ti_fold_%d_error.mat',...
        f)),'error','-v7.3');
    
    
end
