2% testing modified depth estimation
clear all;
% close all;

% for sigma=20:10:50
s = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(s);

%% load images
display('Loading images...')
image_dir = 'ohta';
images_orig = cell(3, 3);   % original images
images_noise = cell(3, 3);  % noisy images
images_bilateral = cell(3, 3);
ref_vnum = 9;
% images_grad = cell(5, 5);
% sigma = 50;
sigma=50;
num_img = 0;
sigmas = 2;
sigmar = 25;
tol = 0.01; 
% window for spatial Gaussian
w  = round(6*sigmas);
if (mod(w,2) == 0)
    w  = w+1;
end

for row = 1:5
    for col = 1:5
        num_img = num_img + 1;
        images_orig{row, col} = double(rgb2gray(imread(['scene1.row', num2str(row), '.col', num2str(col), '.ppm'])));
        if row == 3 && col == 3
            image_true = images_orig{row, col};
        end
        images_noise{row, col} = images_orig{row, col} + sigma * randn(size(images_orig{row, col}));
        images_noise{row, col}(images_noise{row, col} < 0) = 0;
    end
end
% true_disp = double(imread([image_dir, '/scene1.truedisp.pgm']));
[m, n] = size(images_noise{1,1});
% 
  figure;
  imshow(uint8(images_noise{3,3}))
 tic;



% tic
%% disparity map
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
%                if (t-t0)*z>0
%                     im_trans_temp(:,1:(t-t0)*z)=NaN;
%                elseif (t-t0)*z<0
%               %else
%                     im_trans_temp(:,end-(t-t0)*z+1:end)=NaN;
%                end
%                 if (s-s0)*z>0
%                    im_trans_temp(1:(s-s0)*z,:)=NaN;
%                elseif  (s-s0)*z<0
%                % else
%                     im_trans_temp(end-(s-s0)*z+1:end,:)=NaN;
%                 end
            super_img_temp(:, :, num) = im_trans_temp;
            num = num + 1;
        end
    end
    image_stack{z} = super_img;
%     grad_stack{z} = super_grad;
    stds = nanstd(super_img_temp, [], 3);

    std_map(:,:,z) = imfilter(stds, 1/9*ones(3));
end
% MFI=sum(image_stack{10},3)/25
% figure
% imshow(uint8(MFI),[])

texture_map = mean(std_map, 3);
% figure
% imshow(texture_map,[])
% figure; imagesc(texture_map); colormap(gray);
texture_max = max(texture_map(:));
texture_min = min(texture_map(:));

a_1=0.6;
     b_1=round(25-15*a_1);
range_min = 0.75*sigma +5; % Test for the best
range_max = 0.5*sigma +12;

p_size_map = round(-(range_max - range_min)/10*texture_map + 15 + (range_max - range_min)/10*range_min); 
% may com up with other functions

p_size_map(p_size_map >= 15) = 15;
p_size_map(p_size_map < 5) = 5;
% figure; imagesc(p_size_map); colormap(gray)
views_map = 5*ones(m, n);
%views_map(31:end-30, 31:end-30) = round(1.2*p_size_map(31:end-30, 31:end-30) + 7); %p_size_map: patch size based on texture, views_map:number of views based on patch size (essentially
% the flatness of the area
views_map(31:end-30, 31:end-30)=round(a_1*p_size_map(31:end-30, 31:end-30) +b_1);

 views_map(views_map > 25) = 25;
 views_map(views_map<=0)=1;

h_s = cell(1, 15-5+1);
for s = 5:15
    h_s{s-4} = fspecial('average', [s s]);
end
% views_map=imread('E:\CS 766 computer vision\UW-Madison\project\research\pixelwise view selection using LeNet\result.jpg');
% size(views_map)
% views_map=padarray(views_map,[25 25],'both');
% size(views_map)
win_size=15;
    matchScore_im = zeros(m, n, max_disp);
    for z = 1:max_disp
        super_img = image_stack{z};
        image_diff = abs(super_img - repmat(super_img(:,:,ceil(num_views/2)), 1, 1, num_views));
        if z==10
            temp=image_diff(:,:,1);
        end
        image_diff_s = sort(image_diff, 3);
        image_diff_c = cumsum(image_diff_s, 3);
        [y, x] = meshgrid(1:n, 1:m);
        y = y(:); x = x(:);
        %zt = views_map(:);
        zt=win_size*ones(size(views_map(:)));
        idx = sub2ind(size(image_diff_c), x, y, zt);
        matchScore_im(:,:,z) = reshape(image_diff_c(idx)./views_map(:), m, n);  % using flat indeces flatten the matrix into a vector
    end
    [~, disp_map] = min(matchScore_im, [], 3);
    disp_map = medfilt2(disp_map, [3 3], 'symmetric');
    disp_map = uint8(disp_map);
    figure;
    imshow(uint8(disp_map),[]);
    imwrite(disp_map(19:end-19+1,19:end-19+1),['disp_map_win_size',num2str(1),'.jpg']);


