# CS766-final-project
## Project Title
Multi-view stereo estimation with pixel-wise view selection network <br> <br>

Project proposal: "CS766_Project Proposal.pdf"<br>
Midterm report: "CS766_Midterm Report.pdf"<br>
Project proposal: "CS766_Project Report.pdf"<br>
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
### Previous research
The proposed algorithm is based on our previous research: "Shiwei Zhou, Zhengyang Lou et, al.(2018) Improving disparity map estimation for multi-view noisy images. ICASSP 2018 conference."[[multi-view]](https://sigport.org/documents/improving-disparity-map-estimation-multi-view-noisy-images) <br><br>

In this research, A disparity estimation method for multi-view images with noise is investigated by constructing multi-focus image(MFI) and view selection. <br>
Some details of the MFI and optimizing process are given in file "/PPT_slides/zhou2018.pdf"<br>
In order to introduce our method, we use one page of our PPT to restate the problem:
<img src='/imgs/matchingcost.png' position="center" width=500><br>
Here the 'h' is defined manually to achieve good performance, which may not be reasonable when the geometric structure of the 3D scene changes. Instead we want to design a CNN to learn how to do multi-view selection -- to learn value for 'h'.

### New approach
So our intuition of the project is:
 - We designed and trained a neural network to adaptively learn how to select the propriate views for disparity estimation.
 - The input of the neural network is the reference image and multi-view images; 
 - the output is an image where the value of each pixel is the number of views that can achieve the best disparity estimation.
 - Then we use the number of views to continue doing multi-view disparity estimation using method from previous research.

The basic steps of the algorithm for each point to determine the number of views that achieve the best performance are shown in the following picture:<br><br>
<img src='/cnnaim.png' position="center" width=700><br><br>
Detailed steps of the predicting process:
 - Select a point at reference view (RED point) and form a 15 **\times** 15 window around it.
 - Window the same size at other images to form a 25 images stack as an input volume **x**.
 - Order the views with a specific standard so the network only needs to choose the number of views.
 - Using from 5 views to 25 views to generate 21 disparity maps in total: **d_5** to **d_25**.
 - Compare the ground truth from 5 to 25 and find **d_k** that matches the best.
 - Output '**y** = **k**' as the number of views to select for multi-view disparity estimation.

Since we have the clear definition of **x** and **y**, we can train our image data on a CNN structure based on LeNet(shown in following picture).<br>
<img src='/imgs/lenet.png' position="center" width=400><br>

## Experiment and evaluation
### Training the model
Due to limitation of training images, we trained our network on the same group of images. We use half of the image as training data  and use the other half as test data. The image is Tsukuba from the middlebury dataset. [[middlebury2001]](http://vision.middlebury.edu/stereo/data/scenes2001/).<br> 
<br>
### Experiment Evaluation
Here are some results of our proposed algorithm:
Size map are shown as following, as can be seen from the size map result, the overall outlines and structure of objects can be seen clearly. Some surface pixels are smoothed which do not affect too much on the overall performance.<br>
<img src='/imgs/result0.png' position="center" width=400><br>
<br>
Generated disparity map alone with results using other algorithms are shown as following. As can be seen from the visual inspection, our approach achieved relatively better performance than other two methods. Details like the table lamp arm are clearer to see.  <br>
<img src='/imgs/result2.png' position="center" width=400><br>

<br>
Here shows the comparison of the ground truth, image of real object and our result.
<p float="left">
<img src='/imgs/result_true.png' width=270 height = 200>
<img src='/imgs/real_object_image.png'width=270 height = 200>
<img src='/imgs/result01.png' width=270 height = 200>
</p>

Obviously, our image shows more details than the ground truth, especially for those far away from the camera such as bookshelves. However, our algorithm did not handle the details near the camera very well such as the base of the plaster.
<br>

We also compared the per-pixel error rate with the ground truth disparity image, results are shown as following. The error rate decreases by about 8 percent compared to our previous method.<br>
<img src='/imgs/table.png' position="center" width=300><br>
<br>


### Future work
 - Instead of using current method, we will use state-of-the-art method to compute the matching cost.
 - Our method was currently limited to use discrete integer disparity, so we will try to conduct the method with continuous disparities.
<br>
