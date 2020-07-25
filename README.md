# AIM 2020 Relighting Challenge

This is YorkU team's source code for the three tracks of [AIM 2020 relighting challenge](https://data.vision.ee.ethz.ch/cvl/aim20/).


<p align="center">
  <img width = 30% src="https://user-images.githubusercontent.com/37669469/87887746-dcfd9c00-c9f5-11ea-83a1-c57b4b0e11f3.gif">
</p>


As we explained in the challenge report, our solution is based on a normalization network (for track 1 & 3). We'll explain the details of the data used to train our normalizataion network and also how we augment the training data using the [WB augmenter](https://github.com/mahmoudnafifi/WB_color_augmenter). 

![models](https://user-images.githubusercontent.com/37669469/88464846-f6866400-ce8b-11ea-8487-8fa275a150a1.jpg)

### Normalization Data:

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


### Data Augmentation for Track 1 & 2:

Augment input images with WB augmenter to generate 10 augmented images for 
each original image. We assume the input images (after augmentation) are located in `train_t1`. For `train_t1`, the augmented data and new 
ground truth should be located in `train_t1/input_aug` and `train_t1/target_aug`. We have used the modified `demo_WB_color_augmentation.m` (given) to process input  and target images for track 1.


## Image Relighting (task 1 & task 3)

This code requires PyTorch. We process downscaled images followed by a guided upsampling and post-processing steps. The code for the post-processing steps is given in `relighting/post_processing.m`. It depends on [Bilateral Guided Upsampling](https://github.com/google/bgu). If you use Windows operating system, we built an exe program for the upsampling and post-processing steps. However, if this does not work, you need to rebuild the `post_processing.exe` program by downloading the [Bilateral Guided Upsampling](https://github.com/google/bgu) and build the `relighting/post_processing.m` code using Matlab. 

To begin, please make sure that you prepared the normalization training data and image augmentation as described above. 


Run `relighting/train.py` to train the normalization net, the one-to-one image relighting net, and the one-to-any image relighting net. Please adjust the training image directories accordingly. 

To test trained models, run `relighting/train.py`. You can select the target task using the 'task' argument. If you selected `relighting` for the `task` and the `input_dir_t` was `None`, the code will run the one-to-one mapping. 


## Illuminant estimation (task 2)
This code requires Matlab 2019b or higher. To begin, please make sure that your data was augmented as described above. 

Run `estimation/training_resnet_hist.m` to train the color temperature estimation network and `estimation/training_resnet.m` to train the angle estimation network. Please adjust the training image directory accordingly. 

To test the trained models, run `testing_code.m`. 

## Trained models
Currently, we are not going to release our trained models, but they may be available soon.


## Citation

Please cite the following paper if you used our method:

1. Majed El Helou, et al., AIM 2020: Scene Relighting and Illumination Estimation. In ECCV workshops, 2020.

If you used the [RGB-uv histogram](https://github.com/mahmoudnafifi/image_relighting/blob/master/estimation/get_RGB_uv_hist.m), please cite the following papers:

1. Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When color constancy goes wrong: Correcting improperly white-balanced images. In CVPR, 2019.

2. Mahmoud Afifi and Michael S. Brown. Sensor-independent illumination estimation for DNN models. In BMVC, 2019.

If you used the [WB augmenter](https://github.com/mahmoudnafifi/WB_color_augmenter), please cite the following paper:

1. Mahmoud Afifi and Michael S. Brown. What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance. In ICCV, 2019.
