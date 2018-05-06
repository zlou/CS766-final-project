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
Stereo estimation is a fundamental computer vision: 
Given two images for the same scene from different views, compute the disparity for each pixel and then generate depth map, from which we can form a 3D scene. Following is a figure showing the left-right view of a 3D scene and two depth map they generated.
<img src='imgs/disparity.png.' width=600>
