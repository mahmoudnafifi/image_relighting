%Author: Mahmoud Afifi


function output = post_processing(input_fs_file, upsampling, output_ds_file, remove_black_pixels, w_color_transfer, target_file)

if nargin == 3
    remove_black_pixels= "0";
    w_color_transfer = "0";
    target_file = [];
elseif nargin == 3
    remove_black_pixels= "0";
    w_color_transfer = "0";
    target_file = [];
elseif nargin == 4
    w_color_transfer = "0";
    target_file = [];
elseif nargin == 5
    error('no target is given');
end

upsampling = str2double(upsampling);
w_color_transfer = str2double(w_color_transfer);
remove_black_pixels = str2double(remove_black_pixels);

%if w_color_transfer == 1
%    addpath('color_transfer');
%end

input_fs = im2double(imread(input_fs_file));

if remove_black_pixels == 1
    zero_mask = ones(size(input_fs,1), size(input_fs,2));
    zero_mask(input_fs(:,:,1)==0 & input_fs(:,:,2)==0 & input_fs(:,:,3)==0)= 0;
    %se = strel('disk', 7);
end

output_ds = im2double(imread(output_ds_file));

if upsampling == 1
    addpath('./bgu-master/src/matlab');

    edge_fs = rgb2luminance(input_fs); % Used to slice at full resolution.
    output_ds = imresize(output_ds,[128, 128]);
    output_fs_gt = zeros(size(input_fs));
    input_ds = imresize(input_fs,[size(output_ds,1),size(output_ds,2)]);
    edge_ds = rgb2luminance(input_ds); % Determines grid z at low resolution.
    results = testBGU(input_ds, edge_ds, output_ds, [], input_fs, edge_fs, output_fs_gt);
    output = results.result_fs;
else
    output = output_ds;
    if sum(size(output) == size(input_fs)) ~= 3
        output = imresize(output, [size(input_fs, 1), size(input_fs,2)]);
    end
end

if remove_black_pixels == 1
    for c = 1 : 3
        output(:,:,c) = output(:,:,c) .* zero_mask;
    end
end

if w_color_transfer == 1
    target = im2double(imread(target_file));
    output_2 = colour_transfer_MKL(output, target);
    output = 0.75 * output  + 0.25 * output_2;
end

imwrite(output,'result.png');


