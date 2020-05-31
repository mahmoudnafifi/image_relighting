images = dir(fullfile('..','validation_original','*.png'));

images = {images(:).name};

fold = 0; %use fold = 0 for ensembling

round = 0;

if round == 1
         tis = [0, 0.25, 0.5, 0.75, 1];
end

if fold ~= 0 
    
    est_model = sprintf('model_%d.mat',fold);
    deep_model = sprintf('deep_model_%d.mat',fold);
    load(fullfile('data_driven_models',est_model));
    load(fullfile('deep_learning_models',deep_model));
else
    
    load(fullfile('data_driven_models','model_1.mat'));
    load(fullfile('deep_learning_models','deep_model_1.mat'));
    model_1 = model;
    net_1 = net;
    
    %load(fullfile('data_driven_models','model_2.mat'));
    load(fullfile('deep_learning_models','deep_model_2.mat'));
    %model_2 = model;
    net_2 = net;
    
    %load(fullfile('data_driven_models','model_3.mat'));
    load(fullfile('deep_learning_models','deep_model_3.mat'));
    %model_3 = model;
    net_3 = net;
    
end
results = zeros(length(images), 2);
for i = 1 : length(images)
    I = im2double(imread(fullfile('..','validation_original',images{i})));
    I = imresize(I, [227, 227]);
    if fold ~=0
        ti = model.estimate_ill(I);
        [YPred,~] = classify(net,I*255);
        fi = str2double(char(YPred));
    else
        ti_1 = model_1.estimate_ill(I);
        [YPred,~] = classify(net_1,I*255);
        fi_1 = str2double(char((YPred)));
        %ti_2 = model_2.estimate_ill(I);
        [YPred,~] = classify(net_2,I*255);
        fi_2 = str2double(char((YPred)));
        %ti_3 = model_3.estimate_ill(I);
        [YPred,~] = classify(net_3,I*255);
        fi_3 = str2double(char((YPred)));
        ti = ti_1;
        %ti = median([ti_1, ti_2, ti_3]);
        fi = median([fi_1, fi_2, fi_3]);
        if round == 1
            [~,id] = min(abs(tis - ti));
            ti = tis(id);
        end
    end
    results(i, :) = [ti, fi];
    fprintf('image %s, ti = %f, fi = %f\n',images{i}, ti, fi);
end

writematrix(results,'results.csv')