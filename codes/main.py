import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import LeNet_modified as lenet
from sklearn.utils import shuffle
# import disparity image
disp_gt=cv2.imread('scene1.truedisp.pgm',0) # import image, 0 indicates reading the image as a grayscale image
if disp_gt is None: # check if the image has been correctly imported
    print('none')

row,col=disp_gt.shape
num_img=25
# import images
img_stack=np.ndarray(shape=(row,col,25))
for i in range(5):
    for j in range(5):
        k=i*5+j
        im_temp=cv2.imread('scene1.row'+str(i+1)+'.col'+str(j+1)+'.ppm',0)
        img_stack[:, :, k]=im_temp[18:-18,18:-18]
        img_stack[:,:,k]=np.float64(img_stack[:,:,k])
        if img_stack[:,:,k] is None:
            print('image not imported correctly')
disp_img_stack=np.ndarray(shape=(row,col,21),dtype=np.float64)

# plt.imshow(img_stack[:,:,1], cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


for i in range(0,21):
    disp_img_stack[:,:,i]=cv2.imread('disp_map_win_size'+str(i+5)+'.jpg',0)
    disp_img_stack[:,:,i]=np.float64(disp_img_stack[:,:,i])



# initialize the win_size_map, whose values are the number of windows should be used for every pixel
win_size_map=np.zeros(shape=(row,col),dtype=np.int64)
disp_gt_stack=np.ndarray(shape=(row,col,21),dtype=np.float64)
for i in range(21):
    np.copyto(disp_gt_stack[:,:,i],disp_gt,casting='unsafe')

diff_stack=np.abs(disp_gt_stack-disp_img_stack)

np.argmin(diff_stack,axis=2,out=win_size_map)

plt.imshow(win_size_map, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#print((win_size_map.shape[0]-15+1)*(win_size_map.shape[1]-15+1))
""" creating the training and testing set"""
rmin=7
rmax=row-7
colmin=7
colmax=col-7
# training_set={}
# training_set['image']=[]
# training_set['']
X=[]
Y=[]
for i in range(rmin,rmax):
    for j in range(colmin,colmax):
        Y.append(win_size_map[i,j])
        X.append(img_stack[i-7:i+8,j-7:j+8,:])
print(np.max(win_size_map),'  ',np.min(win_size_map))
train_x,x_test,train_y,y_test=train_test_split(X,Y,test_size=0.2)
# #
# x_train,x_validation,y_train,y_validation=train_test_split(train_x,train_y,test_size=0.2)
# #
# lenet.train(x_train=x_train,y_train=y_train,x_validation=x_validation,y_validation=y_validation)
# lenet.test(x_test,y_test,50)




#win_size_map_pre=np.zeros(shape=(rmax-rmin,colmax-colmin),dtype=np.int64)
out=lenet.predict(X,num_channel=25,batch_size=len(X))
out=out+5
# # for i in range(rmin,rmax):
# #     for j in range(colmin,colmax):
# #         #print(np.expand_dims(img_stack[i-7:i+8,j-7:j+8,:],0),np.shape(np.expand_dims(img_stack[i-7:i+8,j-7:j+8,:],0)))
# #         win_size_map_pre[i,j]=lenet.predict(np.expand_dims(img_stack[i-7:i+8,j-7:j+8,:],0))[0]
win_size_map_pre=np.reshape(out,(rmax-rmin,colmax-colmin))
# plt.imshow(win_size_map_pre, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
cv2.imwrite('result.jpg',np.uint8(win_size_map_pre))