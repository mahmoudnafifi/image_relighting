%Author: Mahmoud Afifi

dir_in = 'gt';
dir_out = 'gt_images';
sz = [512,512,3];
if exist(dir_out,'dir') == 0
    mkdir(dir_out);
end
images = dir(fullfile(dir_in,'*.png'));
images = {images(:).name};
for i = 1 : length(images)
    parts = strsplit(images{i}, '_');
    base_name = [parts{1} '_*'];
    t_images = dir(fullfile(dir_in, base_name));
    t_images = {t_images(:).name};
    gt_img = zeros(sz);
    found = 0;
    for j = 1 : length(t_images)
        if exist(fullfile(dir_out, t_images{j}),'file') ~=0
            found = 1;
            break;
        end
        gt_img = gt_img + im2double(imread(fullfile(dir_in, t_images{j})));
    end
    if found == 1
        continue;
    end
    gt_img = gt_img/length(t_images);
    for j = 1 : length(t_images)
        imwrite(gt_img,fullfile(dir_out, t_images{j}));
    end
end

for i = 1 : length(images)
    delete(fullfile(dir_in,images{i}));
end
rmdir(dir_in);


    