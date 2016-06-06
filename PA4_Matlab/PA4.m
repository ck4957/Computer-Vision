
%%%%%%MATLAB STUB%%%%%%%%%
%implement the TODOs then run the PA4 function from 
%the matlab command line simply with 'PA4' (make sure the Current
%Matlab folder on left has this file.
%
function PA4

%Read both images
img1= im2double(rgb2gray(imread('aerial1.jpg')));
img2= im2double(rgb2gray(imread('aerial2.jpg')));

%allocate matrix to hold features1 and features 2
%HOG vecs are 128 length, and we are providing 5 keypoints
%each column will be a feature vector

%Initialize the two feature vectors
features1 = zeros(128,5);
features2 = zeros(128,5);

keyPoints1 = [402 372; 
      371 230; 
      156 381; 
      419 231; 
      323 322; 
      ];
  
keyPoints2 = [325 232; 
      300 90; 
      81 230; 
      348 94; 
      249 182; 
      ];  
  

for i = 1:5
    curr = keyPoints1(i,:);
    features1(:,i) = HOG(img1,curr(1),curr(2))
    curr = keyPoints2(i,:);
    features2(:,i) = HOG(img2,curr(1),curr(2));
end

%TODO: print result in meaningful way,
%      unless using specified format, see matchFeatures description
result = matchFeatures(features1,features2);

%get the two keypoints based on the result values
keyPts1 = keyPoints1(result(:,1),:);
keyPts2 = keyPoints2(result(:,2),:);

%Plot the matching featurs using inbuilt function of matlab
figure;ax = axes;
showMatchedFeatures(img1,img2,keyPts1,keyPts2,'montage','Parent',ax);

end

function out_vec = HOG(img, x,y)
   %Function that calculates a 128 element Histogram of Gradients feature 
   %vector for a given keypoint (x,y) in the provided image.
   %TODO: Run HOG algorithm centered around the point x,y and return the 
   %generated feature vector
   
   %Get at 16*16 subimage from the given keypoint
   img1616 = img(x-7:x+8,y-7:y+8);
   
   %Compute the gradient values in x and y direction
   [gx,gy] = imgradient(img1616);
   
   %Calculate the magnitude and angle of 16*16 sub image
   imgMagnitude = ((gx.^2) + (gy.^2)).^.5;
   imgAngle = atan2d(gy,gx);
   
   %If any angle is negative, make it positive by adding 360 to the value
   for i=1:16
       for j=1:16
            if imgAngle(i,j)<0
            	imgAngle(i,j) = imgAngle(i,j) + 360;
            end
       end
   end
   
   %Divide each angle value into corresponding angle category 
   for i=1:16
     for j=1:16
        angle = imgAngle(i,j);
        if (angle>=0) && (angle<=44)
            imgAngle(i,j) = 0;
        elseif  (angle>=45) && (angle<90)
            imgAngle(i,j) = 45;
        elseif  (angle>=90) && (angle<135)
            imgAngle(i,j) = 90;
        elseif  (angle>=135) && (angle<180)
            imgAngle(i,j) = 135;
        elseif  (angle>=180) && (angle<225)
            imgAngle(i,j) = 180;
        elseif  (angle>=225) && (angle<270)
            imgAngle(i,j) = 225;
        elseif  (angle>=270) && (angle<315)
            imgAngle(i,j) = 270;
        elseif  (angle>=315) && (angle<=360)
            imgAngle(i,j) = 315;
        end
     end
   end
   
   %to store final feature vectors
   result_bins = [];
   
   %Travrse the 16*16 sub image
   for i=1:4:16
     for j=1:4:16
         %Divide the image into 4*4 block and get its magnitude and angle
        img44Magnitude = imgMagnitude(i:i+3,j:j+3);
        img44Angle = imgAngle(i:i+3,j:j+3);
        %initialize the bins arrays
        bins = zeros(1,8);
        for m = 1:4
            for n = 1:4
            %check the angle value and classify them into their bins
            %category
            angle = img44Angle(m,n);
            switch angle
                case 0
                bins(1) = bins(1) + img44Magnitude(m,n);
                case 45
                bins(2) = bins(2) + img44Magnitude(m,n);
                case 90
                bins(3) = bins(3) + img44Magnitude(m,n);
                case 135
                bins(4) = bins(4) + img44Magnitude(m,n);
                case 180
                bins(5) = bins(5) + img44Magnitude(m,n);
                case 225
                bins(6) = bins(6) + img44Magnitude(m,n);
                case 270
                bins(7) = bins(7) + img44Magnitude(m,n);
                case 315
                bins(8) = bins(8) + img44Magnitude(m,n);
            end
            end
        end
        %Concatenate the 8 bins to final bin array
        result_bins = cat(2,result_bins, bins);
     end
   end
   out_vec = result_bins;
end

function out_indicies = matchFeatures(features1,features2)
   %Function that takes in a matrix with feature vectors as columns
   %dim(features1) = 128 by n = dim(features2)
   %where n is the number of feature vectors being compared.
   %Output should indicate which columns (indicies) are the best matches
   %between the features1 and features2. One possibility is 
   %dim(out_indicies) = n by 2, where n is the same as before. 
   %The first column could be the elements 1:n (indicies of columns of
   %features1), and then for each row the element in the second column is
   %the column index of the best match from features2.
   %Your output does not have to be exactly of this format, but should
   %clearly indicate which columns from features1 match with features2, if
   %not points will be deducted.
   %
   %TODO: Calculate the closest match for the vectors in the columns of
   %features1 to the columns of features2 and return the a matrix that
   %indicates the matched indicies.
   
   %Initiliaze the out_indices array
   out_indicies = zeros(5,2);
   %threshold value
   min = 0.03;
   a = sqrt(sum(features1.^2));
   x = features1/a;
   b = sqrt(sum(features2.^2));
   y = features2/b;
   for i=1:5
           out_indicies(i,1)=i;
           if x(i)-y(i)<=min
               out_indicies(i,2) = i;
           end
   end
end