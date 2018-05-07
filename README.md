# CS766-final-project
## Project Title
Multi-view stereo estimation with pixel-wise view selection network <br>  
## Author
Zhengyang Lou, Xucheng Wan, Xiao Wang <br>  

## Introduction
This project is aiming at: 
 - Integrate state-of-art binocular disparity estimation algorithms to multiview disparity estimation.
 - Implement neural networks to determine the views to use for every pixel.

## Why disparity estimation is important
Stereo estimation is a fundamental computer vision: <br>
Given two images for the same scene from different views, compute the disparity for each pixel and then generate depth map, from which we can form a 3D scene. <br>
Following is a figure showing the left-right view of a 3D scene and two depth map they generated. <br>
<img src='/imgs/disparity.png' position="center" width=450>

## Binocular vs. Multi-view disparity
Introduction and details about to Binocur and Multi-view disparity can be found in file "/PPT_slides/Intro_disparity.pdf" <br> <br>
Binocular disparity is just 1-D estimation which may ignore some vertical information.  <br>
Multi-view disparity is a nature extension of binocular method at 2-D estimation, which uses more than two images and thus usually would get better performance. <br>

## Multi-view Selection
However, multi-view disparity estimating method often yields noisy, spurious disparity maps due to occlusions, scene discontinuity, imperfect light balance and other disturbance. <br>
The following picture is the performance using different number of views to generate disparity map for a specific 3D scene. <br>
As can be seen from the curve, when number of views increases, the performance first get better and reach a min point on the curve. When number of views keeps increasing, the performance however, drops a little afterwards. <br>
So，we only need to select some of the views to reach best performance -- this is the problem to solve in this project.
<img src='/imgs/viewselect.png' position="center" width=400>

## Existing Algorithms
Binocular-disparity estimation algorithms can be found in file "/PPT_slides/existing_binocular.pdf" <br>
Multiview-disparity estimation algorithms can be found in file "/PPT_slides/existing_multiview.pdf" <br><br>

## Our proposed algorithm
The proposed algorithm is based on our previous research: "Shiwei Zhou, Zhengyang Lou et, al.(2018) Improving disparity map estimation for multi-view noisy images. ICASSP 2018 conference." <br>
The paper can be obtained here: [[multi-view]](https://sigport.org/documents/improving-disparity-map-estimation-multi-view-noisy-images) <br>
In this research, A disparity estimation method for multi-view images with noise is investigated by constructing multi-focus image and view selection. The basic steps for this multi-view disparity estimation can be seen






