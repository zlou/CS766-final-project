clear all;
close all;

s = RandStream('mt19937ar', 'Seed', 1);
RandStream.setGlobalStream(s);

%% load images
sigma=0
image_dir = 'ohta';
images_orig = cell(1, 5);   % original images
images_noise = cell(1, 5);  % noisy images
% sigma = 50;
num_img = 0;
for row = 1:5
    for col = 1:5
        num_img = num_img + 1;
        images_orig{row, col} = double(rgb2gray(imread([image_dir, '/scene1.row', num2str(row), '.col', num2str(col), '.ppm'])));
       % images_orig{row, col} = double(rgb2gray(imread([image_dir, '/im', num2str(col-1), '.ppm'])));
        images_noise{row, col} = images_orig{row, col} + sigma * randn(size(images_orig{row, col}));
    end
end
[m, n] = size(images_noise{1,1});
tic;
%% generate multi focus images
max_disp = 15; % max disparity
s0 = 3; t0 = 3; % target view
Iz = zeros(m, n, max_disp); % multi focus images
for z = 1:max_disp
    C_sum = zeros(m, n);
    w_sum = 0;
    for s = 1:5
        for t = 1:5
            C_translate = imtranslate(images_noise{s, t}, [(t - t0) * (z), (s - s0) * (z)]);
            weight = exp(-((s - s0)^2 + (t - t0)^2) / 50);
%             weight = 1;
            C_sum = C_sum + weight * C_translate;
            w_sum = w_sum + weight;
        end
    end
    Iz(:, :, z) = C_sum / w_sum;
end

%% focus measurement for MFI
tic;
Fz = zeros(m, n, max_disp); % focus measure matrix
L = 2;  % half block size
C_ref = images_noise{s0, t0}; % target image
I_est = zeros(m, n);    % estimated image
F_est = zeros(m, n);    % for calculating reliability
img_diff = cell(1, max_disp);
for z = 1:max_disp
    img_diff{z} = abs(Iz(:, :, z) - C_ref);
end
for z = 1:max_disp
    for x = 1:m
        for y = 1:n
            rmin = max(1, x - L);
            rmax = min(x + L, m);
            cmin = max(1, y - L);
            cmax = min(y + L, n);
%             Fz(x, y, z) = sum(sum(abs(Iz(rmin:rmax, cmin:cmax, z) - C_ref(rmin:rmax, cmin:cmax)))); % use SAD for evaluating similarity of blocks
            Fz(x, y, z) = sum(sum(img_diff{z}(rmin:rmax, cmin:cmax)));
        end
    end
end
[val, idx] = min(Fz, [], 3);    % min Fz to get the estimated disparity
disp_map = idx; % estimated disparity map, i.e. z(x,y) in the paper
toc;
for i = 1:n
    Iz_2d = Iz(:, i, :);
    Fz_2d = Fz(:, i, :);
    I_est(:, i) = Iz_2d(sub2ind(size(Iz_2d), [1:length(Iz_2d)]', disp_map(:, i)));
    F_est(:, i) = Fz_2d(sub2ind(size(Fz_2d), [1:length(Fz_2d)]', disp_map(:, i)));
end
save(['disp_Miyata_ohta_sigma_',num2str(sigma),'_.mat'],'disp_map');
%% reliability evaluation
th = sigma;    % threshold
R = F_est / (L * 2 + 1)^2;    % reliability
% th = prctile(R(:), 90);    % threshold

reliable_map = R > th;
% SE = strel('disk', 2);
% reliable_map = imdilate(reliable_map, SE);
% reliable_map = imerode(reliable_map, SE);
idx = find(reliable_map);   % index of unreliable pixels

I_est(idx) = C_ref(idx);
I_est_PS = I_est;   % original PS
toc;
%% for pixels with R > th, replace the pixel with weighted average of its neighbors

K = 5; % half search window size
output=zeros(m+2*L,n+2*L);
output_count=zeros(m+2*L,n+2*L);
% h = sigma; % decay of weighting function
if sigma > 0 && sigma <=30
    h=0.4*sigma^2;
elseif sigma > 30 && sigma <= 75
    h=0.35*sigma^2;
else
    h=0.3*sigma^2;
end
I_est_pad = padarray(C_ref, [L L], 'symmetric');
% make kernel
kernel = make_kernel(L);
kernel = kernel / sum(sum(kernel));
% do NLM only on unreliable pixels (using existing code)
[i, j] = ind2sub([m, n], idx);
for k = 1:length(idx)
    i1 = i(k) + L;
    j1 = j(k) + L;
    W1 = I_est_pad(i1 - L:i1 + L , j1 - L:j1 + L);
    
    wmax = 0;
    average = zeros(2*L+1);
    sweight = 0;
    
    rmin = max(i1 - K,L + 1);
    rmax = min(i1 + K,m + L);
    smin = max(j1 - K,L + 1);
    smax = min(j1 + K,n + L);
    
    for r = rmin:1:rmax
        for s = smin:1:smax
            if (r == i1 && s == j1) continue; end;
            W2 = I_est_pad(r - L:r + L , s - L:s + L);
            d = sum(sum(kernel .* (W1 - W2) .* (W1 - W2)));
            w = exp(-max(d-2*sigma^2, 0) / h);
            if w > wmax
                wmax = w;
            end
            sweight = sweight + w;
            average = average + w * W2;
        end
    end
    
    average = average + wmax * W1;
    sweight = sweight + wmax;
    
    if sweight > 0
        output(i1-L:i1+L, j1-L:j1+L) = output(i1-L:i1+L, j1-L:j1+L) + average / sweight;
%         I_est(i(k), j(k)) = average / sweight;
    else
        output(i1-L:i1+L, j1-L:j1+L) = output(i1-L:i1+L, j1-L:j1+L) + I_est_pad(i1-L:i1+L, j1-L:j1+L);
%         I_est(i(k), j(k)) = I_est(i(k), j(k));
    end
    output_count(i1-L:i1+L, j1-L:j1+L) = output_count(i1-L:i1+L, j1-L:j1+L) + ones(2*L+1);
end
output = output ./ output_count;
output = output(L+1:end-L, L+1:end-L);
for k = 1:length(idx)
    I_est(i(k), j(k)) = output(i(k), j(k));
end

figure; imshow(uint8(disp_map), [])
title('Disparity Map')
figure; imshow(reliable_map)
title('Reliability Map')
figure; imshow(uint8(images_orig{s0, t0}))
title('Original Image')
figure; imshow(uint8(C_ref))
title('Noisy Image')
figure; imshow(uint8(I_est_PS))
title('Denoised Image - PS')
figure; imshow(uint8(I_est))
title('Denoised Image - NEW')


%%
% fima = NLmeansfilter_patch(C_ref, 5, 2, sigma);
% figure; imshow(uint8(fima))
% title('Denoised Image - NLM')

%% NLM denoising
% tic;
% fima = NLmeansfilter(C_ref, 5, 2, sigma);
% figure; imshow(uint8(fima))
% title('Denoised Image - NLM')
% toc;
%% compute PSNR
MAX = 255;
MSE = sum(sum((C_ref - images_orig{s0, t0}).^2)) / m / n;
MSE_PS = sum(sum((I_est_PS - images_orig{s0, t0}).^2)) / m / n;

PSNR_Noise = 10 * log10(MAX^2 / MSE)  % before denoising
PSNR_PS = 10 * log10(MAX^2 / MSE_PS)  % after denoising

% MSE_NLM = sum(sum((fima - images_orig{s0, t0}).^2)) / m / n;
% PSNR_NLM = 10 * log10(MAX^2 / MSE_NLM)

MSE_NEW = sum(sum((I_est - images_orig{s0, t0}).^2)) / m / n;
PSNR_NEW = 10 * log10(MAX^2 / MSE_NEW)
