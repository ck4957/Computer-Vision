
%Read both images
img1= im2double(rgb2gray(imread('aerial1.jpg')));
img2= im2double(rgb2gray(imread('aerial2.jpg')));


points1 = detectSURFFeatures(img1);
[features1, valid_points1] = extractFeatures(img1, points1);

points2 = detectSURFFeatures(img2);
[features2, valid_points2] = extractFeatures(img2, points2);

figure;ax = axes;
showMatchedFeatures(img1,img2,valid_points1,valid_points1,'montage','Parent',ax);
