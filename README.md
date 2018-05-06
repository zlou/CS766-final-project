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
Introduction and details about to Binocur and Multi-view disparity can be found in folder "/PPT_slides/disparity.pdf" <br>
Binocular disparity is just 1-D estimation which may ignore some vertical information.  <br>
Multi-view disparity is the extension of binocular method at 2-D estimation, which uses more than two images. <br>
