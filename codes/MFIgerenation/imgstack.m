clear;
clc;
close all;

display('Generating focus image stacks...')
max_disp = 15; % max disparity
s0 = 3; t0 = 3; % target view
s1 = 1; s2 = 5; % start and end view for row
t1 = 1; t2 = 5; % start and end view for column
num_views = (s2 - s1 + 1) * (t2 - t1 + 1);
image_stack = cell(1, max_disp);   % super images that contain stacks of translated images for each disparity
% grad_stack = cell(1, max_disp);
std_map = zeros(m,n,max_disp);
% figure; imshow(uint8(images_noise{s0, t0}))

for z = 1:max_disp
    super_img = zeros(m, n, num_views);
%     super_grad = zeros(m, n, num_views);
    super_img_temp = zeros(m, n, num_views);
    num = 1;
    for s = s1:s2
        for t = t1:t2
            im_trans = imtranslate(images_noise{s, t}, [(t - t0) * (z), (s - s0) * (z)]);
%             grad_trans = imtranslate(images_grad{s, t}, [(t - t0) * (z), (s - s0) * (z)]); 
            super_img(:, :, num) = im_trans;
%             super_grad(:, :, num) = grad_trans;
            im_trans_temp = im_trans;
            %   im_trans_temp(im_trans_temp == 0) = NaN;
               if (t-t0)*z>0
                    im_trans_temp(:,1:(t-t0)*z)=NaN;
               elseif (t-t0)*z<0
              %else
                    im_trans_temp(:,end-(t-t0)*z+1:end)=NaN;
               end
                if (s-s0)*z>0
                   im_trans_temp(1:(s-s0)*z,:)=NaN;
               elseif  (s-s0)*z<0
               % else
                    im_trans_temp(end-(s-s0)*z+1:end,:)=NaN;
                end
            super_img_temp(:, :, num) = im_trans_temp;
            num = num + 1;
        end
    end
    image_stack{z} = super_img;
%     grad_stack{z} = super_grad;
    stds = nanstd(super_img_temp, [], 3);

    std_map(:,:,z) = imfilter(stds, 1/9*ones(3));
end