clear;
clc;
close all;
% read the images in to work space
% the first and second images are disparity map for img2 and img6
% images from 3 - 11 is rgb images from different view: img0 - img8
dirr = 'venus';
subdir = dir(dirr);
img_stk = {};
j = 1;
for i = 1:length(subdir)
    if ~(strcmp( subdir(i).name,'.')||strcmp( subdir(i).name,'..'))
        %fprintf(subdir(i).name)
        %fprintf('\n')
        %img_stk{j}=imread(fullfile('barn1',subdir(i).name));
        img_stk{j}=imread(fullfile(dirr,subdir(i).name));
        [a,b,c] = size(img_stk{j});
        if c == 3
            img_stk{j} = rgb2gray(img_stk{j});
        end
        j = j+1;
    end
end

% choose the img2 as the reference
ind_dis = 1;
ind_ref = 2;
dis_img = img_stk{ind_dis};
ref_img = img_stk{ind_ref+3};

max_dis = max(max(dis_img));
num_img = length(img_stk);
% scale the disparity map or not
ind_scale = 0;
sca_fac = 8;
if ind_scale == 1    
    sca_fac = 1;
    dis_img = dis_img/sca_fac;
end

% create a disparity image stack
[ch1,ch2,ch3] = size(ref_img);
dis_stack = zeros(ch1,ch2,(length(img_stk)-2)*ch3);
num_sft_img = (length(img_stk)-2);
dis_img_list = {};
genFname = @(x)([sprintf('venus_label%d.png', round(x/8))]);
outdir = 'venus_label';

% half disparity range (0,1,2,3,4,5)
dis_r_l = 6;
dis_r_r = 3;

% loop through all possible disparity
for d = 8:8:max_dis 
    % create label image stack to hold the labels
    %label_stack = zeros(ch1,ch2,round(max_dis/8)+1);
    label_stack = zeros(ch1,ch2);
    for i = 1:ch1
        for j = 1:ch2
            % label all pixels 
            if dis_img(i,j)>(d-dis_r_l) && dis_img(i,j)<(d+dis_r_r)
                label_stack(i,j) = 1;
            end
        end
    end
    %label_img = label_stack(:,:,round(d/8)+1);
    %label_stack(:,:,d) = label_img;
    %imshow(label_img);
    imwrite(label_stack,fullfile(outdir,genFname(d)));
end
    




