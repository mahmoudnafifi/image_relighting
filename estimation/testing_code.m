%Author: Mahmoud Afifi

imgs_dir = fullfile('..','..','track2_test');

images = dir(fullfile(imgs_dir,'*.png'));

images = {images(:).name};

fold = 0; %use fold = 0 for ensembling

round = 1;

resnet_depth = 18;

if fold ~= 0
    est_model = sprintf('deep_hist_resnet_model_%d_%d.mat',resnet_depth, fold);
    deep_model = sprintf('deep_resnet_model_%d_%d.mat',resnet_depth, fold);
    load(fullfile('deep_learning_models',est_model));
    net_t = net;
    load(fullfile('deep_learning_models',deep_model));
else
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_1.mat',resnet_depth)));
    net_t_1 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_1.mat',resnet_depth)));
    net_1 = net;
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_2.mat',resnet_depth)));
    net_t_2 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_2.mat',resnet_depth)));
    net_2 = net;
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_3.mat',resnet_depth)));
    net_t_3 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_3.mat',resnet_depth)));
    net_3 = net;
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_4.mat',resnet_depth)));
    net_t_4 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_4.mat',resnet_depth)));
    net_4 = net;
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_5.mat',resnet_depth)));
    net_t_5 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_5.mat',resnet_depth)));
    net_5 = net;
    
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_6.mat',resnet_depth)));
    net_t_6 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_6.mat',resnet_depth)));
    net_6 = net;
    
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_7.mat',resnet_depth)));
    net_t_7 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_7.mat',resnet_depth)));
    net_7 = net;
    
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_8.mat',resnet_depth)));
    net_t_8 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_8.mat',resnet_depth)));
    net_8 = net;
    
    
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_9.mat',resnet_depth)));
    net_t_9 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_9.mat',resnet_depth)));
    net_9 = net;
    
    
    
    load(fullfile('deep_learning_models',...
        sprintf('deep_hist_resnet_model_%d_10.mat',resnet_depth)));
    net_t_10 = net;
    load(fullfile('deep_learning_models',...
        sprintf('deep_resnet_model_%d_10.mat',resnet_depth)));
    net_10 = net;
    
end
results = zeros(length(images), 2);
for i = 1 : length(images)
    tic
    I = im2double(imread(fullfile(imgs_dir,images{i})));
    I = imresize(I, [224, 224]);
    in_h = get_RGB_uv_hist(I,224);
    if fold ~=0
        
        [YPred,~] = classify(net_t,in_h*100);
        ti = str2double(char(YPred));
        [YPred,~] = classify(net,I*255);
        fi = str2double(char(YPred));
        
        
        switch ti
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
    else
        
        [YPred,~] = classify(net_t_1,in_h*100);
        ti_1 = str2double(char(YPred));
        
        [YPred,~] = classify(net_1,I*255);
        fi_1 = str2double(char((YPred)));
        
        
        
        
        
        switch ti_1
            case 2500
                ti_1 = 0;
            case 3500
                ti_1 = 0.25;
            case 4500
                ti_1 = 0.5;
            case 5500
                ti_1 = 0.75;
            case 6500
                ti_1 = 1;
        end
        
        
        [YPred,~] = classify(net_t_2,in_h*100);
        ti_2 = str2double(char(YPred));
        [YPred,~] = classify(net_2,I*255);
        fi_2 = str2double(char((YPred)));
        
        
        switch ti_2
            case 2500
                ti_2 = 0;
            case 3500
                ti_2 = 0.25;
            case 4500
                ti_2 = 0.5;
            case 5500
                ti_2 = 0.75;
            case 6500
                ti_2 = 1;
        end
        
        [YPred,~] = classify(net_t_3,in_h*100);
        ti_3 = str2double(char(YPred));
        [YPred,~] = classify(net_3,I*255);
        fi_3 = str2double(char((YPred)));
        
        
        
        switch ti_3
            case 2500
                ti_3 = 0;
            case 3500
                ti_3 = 0.25;
            case 4500
                ti_3 = 0.5;
            case 5500
                ti_3 = 0.75;
            case 6500
                ti_3 = 1;
        end
        
        [YPred,~] = classify(net_t_4,in_h*100);
        ti_4 = str2double(char(YPred));
        [YPred,~] = classify(net_4,I*255);
        fi_4 = str2double(char((YPred)));
        
        
        
        switch ti_4
            case 2500
                ti_4 = 0;
            case 3500
                ti_4 = 0.25;
            case 4500
                ti_4 = 0.5;
            case 5500
                ti_4 = 0.75;
            case 6500
                ti_4 = 1;
        end
        [YPred,~] = classify(net_t_5,in_h*100);
        ti_5 = str2double(char(YPred));
        [YPred,~] = classify(net_5,I*255);
        fi_5 = str2double(char((YPred)));
        
        
        switch ti_5
            case 2500
                ti_5 = 0;
            case 3500
                ti_5 = 0.25;
            case 4500
                ti_5 = 0.5;
            case 5500
                ti_5 = 0.75;
            case 6500
                ti_5 = 1;
        end
        
        [YPred,~] = classify(net_t_6,in_h*100);
        ti_6 = str2double(char(YPred));
                [YPred,~] = classify(net_6,I*255);
        fi_6 = str2double(char((YPred)));
       
        switch ti_6
            case 2500
                ti_6 = 0;
            case 3500
                ti_6 = 0.25;
            case 4500
                ti_6 = 0.5;
            case 5500
                ti_6 = 0.75;
            case 6500
                ti_6 = 1;
        end
        
        [YPred,~] = classify(net_t_7,in_h*100);
        ti_7 = str2double(char(YPred));
                [YPred,~] = classify(net_7,I*255);
        fi_7 = str2double(char((YPred)));
       
        
        switch ti_7
            case 2500
                ti_7 = 0;
            case 3500
                ti_7 = 0.25;
            case 4500
                ti_7 = 0.5;
            case 5500
                ti_7 = 0.75;
            case 6500
                ti_7 = 1;
        end
        
        [YPred,~] = classify(net_t_8,in_h*100);
        ti_8 = str2double(char(YPred));
                [YPred,~] = classify(net_8,I*255);
        fi_8 = str2double(char((YPred)));
        
        switch ti_8
            case 2500
                ti_8 = 0;
            case 3500
                ti_8 = 0.25;
            case 4500
                ti_8 = 0.5;
            case 5500
                ti_8 = 0.75;
            case 6500
                ti_8 = 1;
        end
        
        [YPred,~] = classify(net_t_9,in_h*100);
        ti_9 = str2double(char(YPred));
                [YPred,~] = classify(net_9,I*255);
        fi_9 = str2double(char((YPred)));
      
        switch ti_9
            case 2500
                ti_9 = 0;
            case 3500
                ti_9 = 0.25;
            case 4500
                ti_9 = 0.5;
            case 5500
                ti_9 = 0.75;
            case 6500
                ti_9 = 1;
        end
        
        
        [YPred,~] = classify(net_t_10,in_h*100);
        ti_10 = str2double(char(YPred));
                [YPred,~] = classify(net_10,I*255);
        fi_10 = str2double(char((YPred)));
    
        switch ti_10
            case 2500
                ti_10 = 0;
            case 3500
                ti_10 = 0.25;
            case 4500
                ti_10 = 0.5;
            case 5500
                ti_10 = 0.75;
            case 6500
                ti_10 = 1;
        end
        
        ti = mean([ti_1, ti_2, ti_3, ti_4, ti_5, ti_6, ti_7, ti_8, ti_9, ti_10]);
        
        fi = mode([fi_1, fi_2, fi_3, fi_4, fi_5, fi_6, fi_7, fi_8, fi_9, fi_10]);
        
    end
    results(i, :) = [ti, fi];
    fprintf('image %s, ti = %f, fi = %f (%f seconds)\n',images{i}, ti, fi, toc);
end

writematrix(results,'results.csv')