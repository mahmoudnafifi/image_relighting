training_dir_o = '../training_original';
validation_dir_o = '../validation_original';

training_dir = '../training';
validation_dir = '../validation';

if exist(training_dir, 'dir') == 0
    mkdir(training_dir)
end

if exist(validation_dir, 'dir') == 0
    mkdir(validation_dir)
end

imgs_tr = dir(fullfile(training_dir_o,'*.png')); imgs_tr = {imgs_tr(:).name};
imgs_vl = dir(fullfile(training_dir_o,'*.png')); imgs_vl = {imgs_vl(:).name};


for i =1 : length(imgs_tr)
    imwrite(imresize(imread(fullfile(training_dir_o,imgs_tr{i})),...
        [227, 227]),fullfile(training_dir,imgs_tr{i}));
end
for i =1 : length(imgs_vl)
    imwrite(imresize(imread(fullfile(validation_dir_o,imgs_vl{i})),...
        [227, 227]),fullfile(validation_dir,imgs_vl{i}));
end