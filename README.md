# image_relighting

## Normalization Data:

We used training images of track 2 to train the noramlization net. Let assume that the input images of track 2 are located in `input` directory. Please follow the following steps:

1. Download WB correction and WB augmenter from the following links:


   *  WB correction: https://github.com/mahmoudnafifi/WB_sRGB

   *  WB augmenter: https://github.com/mahmoudnafifi/WB_color_augmenter


2. Apply WB correction to images in training directory of track 2 -- do not 
overwrite original images. Name the result directory as `gt`

3. After white balancing, run `generate_gt_normalization.m` file. This code
 will generate `gt_images` directory that contains the final ground-truth 
images. Also, the code will delete the temporary `gt` directory created by 
the `WB_sRGB_Matlab/demo_images.m` code. Now, you can use the `input` and 
`gt_images` directories to start training the normalization net. 

4. Run the WB augmenter (with its default 2 augmented version in the given 
code) with the given `demo_WB_color_augmentation.m` file. 
Then, run `demo_WB_color_augmentation.m`. Change the directory to make 
input directory variable = 'input'. The code will generate for you 
augmented sets (name the result directory as 'input_aug' and the new gt 
directory as 'gt_images_aug'). Use those directories to train the 
normalization net. 


## Data Augmentation for Track 1 & 2:

Augment input images with WB augmenter to generate 10 augmented images for 
each original image. We assume the input images (after augmentation) are located in `train_t1`. For `train_t1`, the augmented data and new 
ground truth should be located in `train_t1/input_aug` and `train_t1/target_aug`. We have used the modified `demo_WB_color_augmentation.m` (given) to process input  and target images for track 1.


## Citation:

Please cite the following paper if you use our training/testing code:

1. ...

If you use the [RGB-uv histogram](https://github.com/mahmoudnafifi/image_relighting/blob/master/estimation/get_RGB_uv_hist.m), please cite the following papers:

1. Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When color constancy goes wrong: Correcting improperly white-balanced images. In CVPR, 2019.

2. Mahmoud Afifi and Michael S. Brown. Sensor-independent illumination estimation for DNN models. In BMVC, 2019.
