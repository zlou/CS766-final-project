import pdb
import cv2
import numpy as np
from math import exp


############################# load images #########################################################
image_dir = './ohta'
image_orig = np.full((5,5,288,384), 0)
image_noise = np.full((5,5,288,384), 0)
sigma = 0
allfile = 5
print('1')
for row in range(0,allfile):
    for col in range(0,allfile):
        print(row,col)
        img = cv2.imread(image_dir+ '/scene1.row'+ str(row+1)+ '.col'+ str(col+1)+ '.ppm',-1)
        print(img.shape[:2])
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_orig[row,col,:,:] = np.float32(gray_image)
        image_noise[row,col,:,:] = np.float32(gray_image) + sigma * np.random.random(image_orig[0,0,:,:].shape)
        #print(a.shape)
        #print(a[0:5,1])
        #cv2.imshow('image',gray_image)
        #print(img[0:5,0:5])
        #cv2.waitKey(0)
m, n = image_orig[0,0,:,:].shape
#pdb.set_trace()
#print(m,n)

print(image_noise[0,0,0:5,1])
#cv2.imshow('image',image_noise[0,0,:,:])
#cv2.waitKey(0)


############################# generate multi focus images #########################################
max_disp = 15 # max disparity
s0 = 3; t0 = 3 # target view
Iz = np.float32(np.full((m, n, max_disp),0)) # multi focus images

for z in range(0,max_disp):
    C_sum = np.full((m, n),0)
    w_sum = 0
    for s in range(0,5):
        for t in range(0,5):
            translation_matrix = np.float32([ [1,0,(t + 1 - t0) * (z + 1)], [0,1,(s + 1 - s0) * (z + 1)] ])
            
            #cv2.imwrite('color_img.jpg', b)
            C_translate = cv2.warpAffine(np.uint8(image_noise[s,t,:,:]), translation_matrix, (n,m))
            weight = exp(-((s + 1 - s0)*(s + 1 - s0) + (t + 1 - t0)*(t + 1 - t0)) / 50);
            #print(s,t,s + 1 - s0)
            C_sum = C_sum + weight * np.float32(C_translate);
            #if s== 0 and t == 0:
            #    print('weight:' + str(weight))
            #    print('C_translate')
            #    print(C_translate[0:5,0:5])
            #print(C_sum[0:5,0:5])
            w_sum = w_sum + weight;
    Iz[:, :, z] = np.float32(C_sum)/ (w_sum * 1.0);

#np.set_printoptions(precision=4)
#print(type(C_sum))
#print(type(Iz))
print(Iz[10:15, 10:15, 3])
#print(w_sum)


############################## focus measurement for MFI ##########################################
Fz = np.float32(np.full((m, n, max_disp),0))
L = 2
C_ref = image_noise[s0-1,t0-1,:,:]
I_est = np.float32(np.full((m, n),0))
F_est = np.float32(np.full((m, n),0))

img_diff = np.float32(np.full((m, n, max_disp),0))

for z in range(0,max_disp):
    img_diff[:,:,z] = abs(Iz[:, :, z] - C_ref)

for z in range(0,max_disp):
    for x in range(0,m):
        for y in range(0,n):
            rmin = max(1 - 1, x - L);
            rmax = min(x + L, m - 1);
            cmin = max(1 - 1, y - L);
            cmax = min(y + L, n - 1);
            Fz[x, y, z] = np.sum(img_diff[rmin:rmax + 1, cmin:cmax + 1, z]);
            if x == 0 and y == 0 and z == 0:
                print(img_diff[rmin:rmax, cmin:cmax, z])
                #break

val = np.amin(Fz, axis = 2)
idx = Fz.argmin(axis = 2)
#print(val[0:5,0:5])
#print(idx[0:17,0:17])
disp_map = idx


#def a = sub2ind(shape, colidx)
#    for r in rawidx:
        
for i in range(0,n):
    Iz_2d = Iz[:, i, :]
    Fz_2d = Fz[:, i, :]
    #first = np.array(range(0,m))
    #second = np.fill(((m, 1),1))
    #third = disp_map[:,i]
    firstrange,_,_ = Iz.shape
    for raw_1 in range(0,firstrange):
        #print(disp_map[raw_1, i])
        I_est[raw_1, i] = Iz_2d[raw_1, disp_map[raw_1, i]]
        F_est[raw_1, i] = Fz_2d[raw_1, disp_map[raw_1, i]]

#print(I_est[0:5,0:5])
#print(F_est[0:5,0:5])       

    
    #I_est(:, i) = Iz_2d(sub2ind(size(Iz_2d), [1:length(Iz_2d)]', disp_map(:, i)))
    #F_est(:, i) = Fz_2d(sub2ind(size(Fz_2d), [1:length(Fz_2d)]', disp_map(:, i)))

########################## reliability evaluation ################################################
th = sigma
R = F_est / ((L * 2 + 1)*(L * 2 + 1))
reliable_map = R > th;
idx_x,idx_y = np.where(reliable_map)
I_est[idx_x[:],idx_y[:]] = C_ref[idx_x[:],idx_y[:]]
I_est_PS = I_est

print(I_est_PS[0:10,0:10])

########################## for pixels with R > th, replace the pixel with weighted average of its neighbors #######################################################################################

K = 5
output = np.full((m + 2 * L, n + 2 * L),0)
output_count = np.full((m + 2 * L, n + 2 * L),0)
if sigma > 0 and sigma <= 30:
    h = 0.4*sigma*sigma
elif sigma > 30 and sigma <= 75:
    h = 0.35 * sigma*sigma
else:
    h = 0.3*sigma*sigma


I_est_pad = np.pad(C_ref, [L,L],'symmetric')

#def kernel_1 = make_kernel(f)
#    kernel_1 = np.fill((2 * f + 1, 2 * f + 1), 0)
#    for d in range(0,f)
#        value = 1/( (2 * d + 1) * (2 * d + 1))
#        for i in range
kernel = [  [0.02,0.02,0.02,0.02,0.02],\
            [0.02,68/900,68/900,68/900,0.02],\
            [0.02,68/900,68/900,68/900,0.02],\
            [0.02,68/900,68/900,68/900,0.02],\
            [0.02,0.02,0.02,0.02,0.02]]

i_singleunit = np.array(range(0,m))# ATTENTION!
i = i_singleunit
for loop in range(0,n):
    i = np.concatenate((i, i_singleunit))
#print('ishape'+i.shape)
print(i[0:289])
j = np.repeat(np.array(range(0,n)),m)
print('j')
print(j[0:289])
#for x_cord in range(0,m):
#    for y_cord in range(0,n):
#        a = 1

idx = np.array(range(0,m*n))
print(idx[-1])

for k in idx:
    i1 = i[k] + L
    j1 = j[k] + L
    W1 = I_est_pad[i1 - L:i1 + L + 1, j1 - L:j1 + L + 1]
    #print(i1,j1,W1)
    
    wmax = 0
    average = np.full((2*L+1,2*L+1),0)
    sweight = 0
    #print(average)
    
    rmin = max(i1 - K ,L + 1 - 1);
    rmax = min(i1 + K ,m + L - 1);
    smin = max(j1 - K ,L + 1 - 1);
    smax = min(j1 + K ,n + L - 1);
    #print(rmin,rmax,smin,smax)    

    for r in range(rmin,rmax + 1):
        for s in range(smin,smax + 1):
            if r == i1 and s == j1:
                continue

            W2 = I_est_pad[r - L:r + L + 1, s - L:s + L + 1]
            d = np.sum(np.multiply(np.multiply(np.double(kernel),(W1 - W2)),W1 - W2))
            #print(W1 - W2,d,h)
            w = exp(-1 * max(d - 2 * sigma * sigma, 0) / h)

            if w > wmax:
                wmax = w
            sweight = sweight + w
            average = average + w * W2

    average = average + wmax * W1;
    sweight = sweight + wmax;
    #print(average,sweight)

    #print('start')
    #print(average)
    #print(sweight)

    if sweight > 0:
        output[i1-L:i1+L+1, j1-L:j1+L+1] = output[i1-L:i1+L+1, j1-L:j1+L+1]+ average / sweight
    else:
        output[i1-L:i1+L+1, j1-L:j1+L+1] = output[i1-L:i1+L+1, j1-L:j1+L+1] + \
            I_est_pad[i1-L:i1+L+1, j1-L:j1+L+1]

    #print(output[i1-L:i1+L+1, j1-L:j1+L+1])
    output_count[i1-L:i1+L + 1, j1-L:j1+L + 1] = output_count[i1-L:i1+L + 1, j1-L:j1+L+1] + np.full((2*L+1,2*L+1),1)

    #print(output_count[i1-L:i1+L + 1, j1-L:j1+L + 1])

    if k % 1000 == 0:
        print(k)


print(output.shape)
print(output[0:10, 0:10])
print(output_count[0:10, 0:10])

output  = np.divide(output, output_count)
output = output[2:-2,2:-2]

for k in idx:
    I_est[i[k],j[k]] = output[i[k],j[k]] 

print(I_est[0:10,0:10])

pdb.set_trace()
######################### compute PSNR ###########################################################
MAX = 255

MSE = np.sum(np.multiply(C_ref - image_orig[s0-1,t0-1,:,:],C_ref - image_orig[s0-1,t0-1,:,:]))/m/n
#MSE = sum(sum((C_ref - images_orig{s0, t0}).^2)) / m / n;
MSE_PS = np.sum(np.multiply(I_est_PS - image_orig[s0-1,t0-1,:,:],C_ref - image_orig[s0-1,t0-1,:,:]))/m/n
#MSE_PS = sum(sum((I_est_PS - images_orig{s0, t0}).^2)) / m / n;

PSNR_Noise = 10 * np.log10(MAX * MAX / MSE) 
#PSNR_Noise = 10 * log10(MAX^2 / MSE)  % before denoising
PSNR_PS = 10 * np.log10(MAX * MAX / MSE_PS)
#PSNR_PS = 10 * log10(MAX^2 / MSE_PS)  % after denoising

#% MSE_NLM = sum(sum((fima - images_orig{s0, t0}).^2)) / m / n;
#% PSNR_NLM = 10 * log10(MAX^2 / MSE_NLM)

MSE_NEW = np.sum(np.multiply(I_est - image_orig[s0-1,t0-1,:,:],C_ref - image_orig[s0-1,t0-1,:,:]))/m/n
#MSE_NEW = sum(sum((I_est - images_orig{s0, t0}).^2)) / m / n;
#PSNR_NEW = 10 * log10(MAX^2 / MSE_NEW)
PSNR_NEW = 10 * np.log10(MAX*MAX / MSE_NEW)















