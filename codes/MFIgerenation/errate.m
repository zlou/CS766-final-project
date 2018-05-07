img1 = imread('truedisp.row3.col3.pgm');
img2 = rgb2gray(imread('result0503.png'));
count = 1;
[m,n] = size(img2);
for i = 1:m
    for j = 1:n
        if abs(img1(i,j) - img2(i,j))<7.1
            count = count+1;
        end
    end
end
errorrate = 0;
errorrate = 100*(m*n - count)/(m*n);
disp(errorrate)