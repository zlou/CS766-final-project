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
genFname = @(x)([sprintf('venus_gray_dis%d.png', round(x/8))]);
outdir = 'venus_data_gray';
% create label image stack to hold the labels
label_stack = [];

% loop through all possible disparity
for d = 8:8:max_dis
    % shift each image 
    dis_stack = zeros(ch1,ch2,(length(img_stk)-2)*ch3);
    for i = 3:num_img
        % to judge which direction to shift, smaller num shift left, large
        % number shift right
        shift_dic = 0;
        if i ~= ind_ref+3
            shift_dic = (i-3 - ind_ref)/abs(i-3-ind_ref);
        end
        % shift the image towards right
        if shift_dic == 1
            sft_img = imtranslate(img_stk{i},...
                [abs(double(i-3-ind_ref))/4*double(d)/double(sca_fac),0]);
        elseif shift_dic == -1
            sft_img = imtranslate(img_stk{i},...
                [-abs(double(i-3-ind_ref))/4*double(d)/double(sca_fac),0]);
        elseif shift_dic == 0
            sft_img = img_stk{i};
        end
        %figure; imshow(uint8(sft_img));
        j = j+1;
        dis_stack(:,:,i-ind_ref) = sft_img;
    end
    % average the image and save in new stack
    % now we have a ch1 x ch2 x ch3*num_imgs image stack
    % we need to average through the images for each channel
    % loop through pixels
    ave_img = zeros(size(ref_img));
    % create a zero label image
    label_img = zeros(ch1,ch2);
    % loop over the image stack pixel by pixel
    for x = 1:ch1
        for y = 1:ch2
            for z = 1:ch3
                % average 
                ave_img(x,y,z) = mean(dis_stack(x,y,[z:3:num_sft_img*ch3]));
            end
            % label all the pixels 1 within infocus area
            %if std(dis_stack(x,y,[1:3:num_sft_img*ch3])) < 2
            %    label_img(x,y) = 1;
            %end
        end
    end
    dis_img_list{d} = ave_img;
    %label_stack(:,:,d) = label_img;
    %imshow(label_img);
    imwrite(uint8(ave_img),fullfile(outdir,genFname(d)));
end
    




