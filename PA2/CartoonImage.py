import cv2
import numpy as np
import math

def bilateralFilter(img, kSize, sigmaColor, sigmaSpace, borderType=cv2.BORDER_DEFAULT):
    """
    DESCRIBE YOUR FUNCTION HERE
    (Param descriptions from the OpenCV Documentation)
    :param img: Source 8-bit or floating-point, 1-channel or 3-channel image.
    :param kSize: Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace.
    :param sigmaColor: Filter sigma in the color space. A larger value of the parameter
        means that farther colors within the pixel neighborhood (see sigmaSpace) will
        be mixed together, resulting in larger areas of semi-equal color.
    :param sigmaSpace: Filter sigma in the coordinate space. A larger value of the
        parameter means that farther pixels will influence each other as long as their
        colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood
        size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    :param borderType: always Reflect_101
    :return: Filtered image of same size and type as img
    """
    #Convert Image to LAB
    img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(img_lab)
    pad_l = np.pad(l,(4,4),mode='constant')
    pad_a = np.pad(a,(4,4),mode='constant')
    pad_b = np.pad(b,(4,4),mode='constant')
    #kernel = cv2.getGaussianKernel(kSize,sigmaColor)
    result = np.zeros(img_lab.shape)
    img_dim = img_lab.shape
    height = img_dim[0]
    widhth = img_dim[1]
    empty_window = np.zeros([kSize,kSize])
    for i in range(4,height-4):
        print "Here i=",i
        for j in range(4,widhth-4):
            sum1=0;sum2=0;
            for m in range(i-4,i+4):
                for n in range(j-4,j+4):
                    X =pad_a[i,j]-pad_a[m,n]
                    Y = pad_b[i,j]-pad_b[m,n]
                    tempA = ((X**X) - (Y**Y))/(2*sigmaColor*sigmaColor)
                    Z = pad_l[i,j]-pad_l[m,n]
                    tempB = (Z*Z)/(sigmaSpace**2)
                    tempC = -tempA-tempB
                    empty_window[m,n] = math.exp(tempC)
                    sum1 += empty_window[m,n]*pad_l[m,n]
                    sum2+= empty_window[m,n]
            result[i,j]=sum1/sum2


def Canny(img, thresh1, thresh2, L2norm):
    """
    DESCRIBE YOUR FUNCTION HERE
    (Param descriptions from the OpenCV Documentation)
    :param img: 8-bit input image.
    :param thresh1: hysteresis threshold 1.
    :param thresh2: hysteresis threshold 2.
    :param L2norm: boolean to choose between L2norm or L1norm
    :return: a single channel 8-bit with the same size as img
    """
    gaussian_blur_img = cv2.GaussianBlur(img,(5,5),0)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,1,0)

    edge_gradient = math.sqrt((sobelx**2)+(sobely**2))
    theta = math.atan2(sobely,sobelx)


def cartoonImage(filtered, edges):
    """
    DESCRIBE YOUR FUNCTION HERE
    :param filtered: a bilateral filtered image
    :param edges: a canny edge image
    :return: a cartoon image
    """
    img_dim = edges.shape
    height = img_dim[0]
    widhth = img_dim[1]
    for i in range(height):
        for j in range(widhth):
            if edges[i,j] == 1:
                filtered[i,j,:]=0
    return filtered

def RMSerror(img1, img2):
    """
    A testing function to see how close your images match expectations
    Try to make sure your error is under 1. Some floating point error will occur.
    :param img1: Image 1
    :param img2: Image 2
    :return: The error between the two images
    """
    diff = np.subtract(img1.astype(np.float64), img2.astype(np.float64))
    squaredErr = np.square(diff)
    meanSE = np.divide(np.sum(squaredErr), squaredErr.size)
    RMSE = np.sqrt(meanSE)
    return RMSE

if __name__ == '__main__':
    img = cv2.imread("Castle.jpg")
    #bilat = bilateralFilter(img, 9, 50, 100)
    cvbilat = cv2.bilateralFilter(img, 9, 50, 100)
    #print "Bilateral Filter RMSE: "+str(RMSerror(bilat, cvbilat))
    #edges = Canny(img, 100, 200, True)
    cvedges = cv2.Canny(img, 100, 200, True)
    #print "Canny Edge RMSE: "+str(RMSerror(edges, cvedges))
    cartoon = cartoonImage(cvbilat, cvedges)
    #cv2.imshow("Bilateral", bilat)
    #cv2.imwrite("BilateralOutput.jpg", bilat)
    #cv2.imshow("Edges", edges)
    #cv2.imwrite("CannyOutput.jpg", edges)
    cv2.imshow("Cartoon", cartoon)
    cv2.imwrite("CartoonOutput.jpg", cartoon)
    cv2.imshow("Original",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()