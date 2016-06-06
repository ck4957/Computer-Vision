
img1= im2double(rgb2gray(imread('aerial1.jpg')));
img2= im2double(rgb2gray(imread('aerial2.jpg')));

[hog1, hogvisualization] = extractHOGFeatures(img1);

 % figure;
    %imshow(img1); hold on;
    %plot(hogvisualization);

[row,col] = size(img1);
row,col;
gx = imfilter(img1,[-1,0,1]);
gy = imfilter(img1,[-1,0,1]');

magnitude = ((gx.^2) + (gy.^2)).^.5;
angle = atan2d(gy,gx);


keyPoints1 = [402 372; 
      371 230; 
      156 381; 
      419 231; 
      323 322; 
      ]
 
  keyPoints2 = [325 232; 
      300 90; 
      81 230; 
      348 94; 
      249 182; 
      ];  
  

 [features,validPoints] = extractFeatures(img1,keyPoints1,'Method','SURF');
 figure; imshow(img1); hold on;
 plot(validPoints.selectStrongest(10),'showOrientation',true);
 
 
 
 [features1,validPoints1] = extractHOGFeatures(img1,keyPoints1);
 [features2,validPoints2] = extractHOGFeatures(img2,keyPoints2);
 
 features1;
 validPoints1;
features2
validPoints2;
figure;
showMatchedFeatures(img1,img2,validPoints1,validPoints2,'montage','Parent',axes);