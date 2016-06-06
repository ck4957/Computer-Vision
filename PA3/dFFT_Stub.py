import cv2
import numpy as np

def addText(img,text,y=0,x=0):
    """
    A helper function that addes text on top of an image
    :param image: Image for text to be added to
    :param text: String to be placed on image
    :param y: The y offset of the anchor
    :param x: The x offset of the anchor
    """
    image2 = np.copy(img)
    cv2.putText(image2,text,(5+x,25+y),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0.0),4)
    cv2.putText(image2,text,(5+x,25+y),cv2.FONT_HERSHEY_TRIPLEX,0.7,(1.0),2)
    return image2

def logMagDisp(inFFT):
    """
    Calcuates the log magnitude of an image for display
    The image should also be shifted so the fundamental frequency
    is in the center of the image
    :param inFFT: the input complex array from the FFT
    :return: output image from 0 to 1
    """
    # TODO: WRITE THIS CODE
    pass

def dFFT2(img):
    """
    Calculated the discrete Fourier
    Transfrom of a 2D image
    :param img: image to be transformed
    :return: complex valued result
    """
    N=len(img)
    if N==1:
        return img

    even=dFFT2([img[k] for k in range(0,N,2)])
    odd= dFFT2([img[k] for k in range(1,N,2)])

    M=N/2
    l=[ even[k] + cv2.exp(-2j*3.14*k/N)*odd[k] for k in range(M) ]
    r=[ even[k] - cv2.exp(-2j*3.14*k/N)*odd[k] for k in range(M) ]

    return l+r

    # TODO: WRITE THIS CODE
    #pass

def idFFT2(img):
    """
    Calculated the inverse discrete Fourier
    Transfrom of a 2D image
    :param img: image to be transformed
    :return: complex valued result
    """
    # TODO: WRITE THIS CODE
    pass

def outImgArray(img,inFFT,usedFFT,deltas,scaledSin,outImg):
    """
    Creates the output image for display
    Merges the six images together into a display
    Addes textual labels to images
    Calls logMagDisp for each of the FFTs
    ALL IMAGES MUST BE THE SAME SHAPE!!!
    :param img: input image
    :param inFFT: dFFT of the input image
    :param usedFFT: components of dFFT used so far
    :param deltas: the pair of points (current component)
    :param scaledSin: normalized idFFT of the current component
    :param outImg: idFFT of the usedFFT
    :return: the output image to be displayed
    """
    imgDisp = addText(img,"Original Image")
    inFFTdisp = logMagDisp(inFFT)
    inFFTdisp = addText(inFFTdisp,"Input dFFT log(Mag)")
    usedFFTdisp = logMagDisp(usedFFT)
    usedFFTdisp = addText(usedFFTdisp,"Used dFFT log(Mag)")
    topRow = np.hstack((imgDisp,inFFTdisp,usedFFTdisp))
    deltaDisp = logMagDisp(deltas)
    deltaDisp = addText(deltaDisp,"Current Component")
    bottomRow = np.hstack((deltaDisp,scaledSin,outImg))
    bottomRow = addText(bottomRow,"Scaled Sine",0,img.shape[1])
    bottomRow = addText(bottomRow,"Inverse dFFT",0,img.shape[1]*2)
    outImg = np.vstack((topRow,bottomRow))
    return outImg

def mainLoop(img,name="Temp"):
    if len(img.shape) != 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
-    img = img.astype(np.float64)/255.0
    paused = True
    lastKey = ord(' ')
    components = 0
    cv2.namedWindow(name+"'s FFT Display",cv2.WINDOW_AUTOSIZE)
    # TODO: Calculate the input dFFT
    while components < img.size//2:
        # TODO: Calculate the current component and other arrays
        out = outImgArray(img,inFFT,usedFFT,comp,scaledSin,outImg)
        cv2.imshow(name+"'s FFT Display",out)
        if paused:
            lastKey = cv2.waitKey(0) & 0xFF
        else:
            lastKey = cv2.waitKey(5) & 0xFF
        if lastKey == ord('q') or lastKey == ord('Q') or lastKey == 27:
            break
        elif lastKey == ord('p') or lastKey == ord('P'):
                paused = not paused
        components += 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread("lena.tif",0)
    name = "YOUR_NAME_HERE"
    mainLoop(img,name)