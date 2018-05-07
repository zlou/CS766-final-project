disp_gt=imread('scene1.truedisp.pgm'); % ground truth disparity map image

win_size_gt=zeros(size(disp_gt));

[row,col]=size(disp_gt);
img_temp=imread('disp_map_win_size21.jpg');
size(img_temp)
img_stack=zeros(row,col,25-5+1);
row
col
% import images 
for i=1:21
        img_stack(:,:,i)=imread(['disp_map_win_size',num2str(i+4),'.jpg']);
end

