This repository holds the codes and models for my paper at IEEE GCCE 2018 Conference.
>
**Fast Recognition of Human Actions Using Autocorrelation Sequence**; Anh H. Nguyen, Huyen T. T. Tran, Truong Cong Thang, Yong Man Ro ([link](https://ieeexplore.ieee.org/document/8574820))

## 1. Compute video autocorrelation
Run this MatLab script to compute autocorrelation sequences of input videos.
```
examples/LRCN_activity_recognition/human_action_recognition/compute_autocorr.m
```
This script calculates autocorrelation sequences of all videos and writes all autocorrelation sequences into ```video_corr.txt```.

## 2. Find global minimum and its adjacent maximum
Run the following script to compute global maxima and adjacent maxima of autocorrelation sequences using the file ```video_corr.txt``` obtained from the last step.
```
examples/LRCN_activity_recognition/human_action_recognition/process_corr.py
```
Output of this script is a new text file containing all videos for testing in the form 
```
video_name true_label minimum_position maximum_position
```

## 3. Clasify videos
Run the following script to classify action videos using the new test list file.
```
examples/LRCN_activity_recognition/human_action_recognition/classify_video_argument.py
```
There are few parameters controlling the inference process.
```
Params 		  Meaning                       Values	
 -test    file listing test videos          xxx.txt				
 -o       name of output file               xxx.txt				
 -c       clip length                       integer				
 -m       type of input                     'rgb', 'flow'		
 -l       number of overlaping frames       integer				
 -s       starting frame of input segment   '0', 'gmin', 'lmax'	
 -e       end frame of input segment        '0', 'gmin', 'lmax', 'vlength'	
```
For example, this is the command to run the simple model (non-onverlapping clips) using proposed strategy 3 (from the first frame to the maximum immediately following the global minimum) with RGB modality
```
python examples/LRCN_activity_recognition/human_action_recognition/classify_video_argument.py 
-test ucf101_split1_minima_maxima.txt -o simple_rgb_s3.txt -c 16 -m rgb -l 0 -s 0 -e lmax
```

## 4. Fuse scores
To fuse prediction scores of two modalities (RGB and Flow), run ```fuse_rgb_flow.py``` script as follows:
```
python examples/LRCN_activity_recognition/human_action_recognition/fuse_rgb_flow.py 
-rgb rgb_prediction.txt -flow flow_prediction.txt -w 0.333333
```
Here, ```0.33333``` is fusion weight of RGB modality, which means fusion weight of Flow modality is ```0.666667 (2/3)```.

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
