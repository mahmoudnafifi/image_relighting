addpath('./bgu-master/src/matlab');

dirs = {'results'};

for d = 1 : length(dirs)
    mkdir([dirs{d}, '_final']);
    images = dir(fullfile(dirs{d},'*.png'));
    images = {images(:).name};
    for i = 1 : length(images)
        input_fs = im2double(imread(fullfile('input',images{i})));
        zero_mask = ones(size(input_fs,1), size(input_fs,2));
        zero_mask(input_fs(:,:,1)==0 & input_fs(:,:,2)==0 & input_fs(:,:,3)==0)= 0;
        edge_fs = rgb2luminance(input_fs); % Used to slice at full resolution.
        output_ds = im2double(imread(fullfile(dirs{d},images{i})));
        output_ds = imresize(output_ds,[128, 128]);
        output_fs_gt = zeros(size(input_fs));
        input_ds = imresize(input_fs,[size(output_ds,1),size(output_ds,2)]);
        edge_ds = rgb2luminance(input_ds); % Determines grid z at low resolution.
        results = testBGU(input_ds, edge_ds, output_ds, [], input_fs, edge_fs, output_fs_gt);
        output = results.result_fs;
        output = output .* zero_mask;
        imwrite(output, fullfile([dirs{d}, '_final'],images{i}));
    end
end

