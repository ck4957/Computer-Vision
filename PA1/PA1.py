"""
description: Program performs the various operation on given input image
file: PA1.py
language: python3
author: Chirag Kular
"""



import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


def read_display_image():
    """
    Function reads the image and display it.
    It also writes the image to another format
    :return:
    """
    img = cv2.imread("flowers.PNG")
    #cv2.namedWindow("Chirag_Kular", cv2.WINDOW_NORMAL)
    cv2.imshow("Chirag_Kular", img)
    cv2.waitKey()
    #cv2.destroyWindow("Chirag Kular")
    cv2.imwrite("flowers_converted_format.jpg",img)


def convert2grayscale():
    """
    Converts the input image to gray scale and displays it
    :return:
    """
    gray_img = cv2.imread("flowers.png",cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Gray Scale",gray_img)
    cv2.imwrite("flowers_grayscale.jpg",gray_img)
    cv2.waitKey()


def subplotRGB():
    """
    Red, Green and Blue channels of the image are displayed using subplot
    :return:
    """
    img = cv2.imread("flowers.png")
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img[:, :, 2])
    plt.title("Red Channel")

    plt.subplot(1, 3, 2)
    plt.imshow(img[:, :, 1])
    plt.title("Green Channel")

    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, 0])
    plt.title("Blue Channel")
    plt.show()


def gray2binary():
    """
    Converts the grayscale image to binary image
    :return:
    """
    thresh = 100
    img = cv2.imread("flowers.png", 0)
    bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Binary', bin_img)
    cv2.waitKey()


def arithmeticOps():
    """
    Performs different arithmetic operations on two input image
    :return:
    """
    day = cv2.imread("mum@day.jpg")
    h1,w1 = day.shape[:2]
    night = cv2.imread("mum@night.jpg")
    night = cv2.resize(night, (w1,h1)) #Resize the image

    plt.subplot(2,4,1)
    plt.imshow(day)
    plt.title("Mumbai@Day")

    plt.subplot(2,4,2)
    plt.imshow(night)
    plt.title("Mumbai@Night")


    plt.subplot(2,4,3)
    added = cv2.add(day, night)
    plt.imshow(added)
    plt.title("Addition")

    plt.subplot(2,4,4)
    subtract = cv2.subtract(day, night)
    plt.imshow(subtract)
    plt.title("Subtract")

    plt.subplot(2,4, 5)
    bitwise_and = cv2.bitwise_and(night,day)
    plt.imshow(bitwise_and)
    plt.title("bitwise_and")

    plt.subplot(2,4, 6)
    bitwise_not = cv2.bitwise_not(night)
    plt.imshow(bitwise_not)
    plt.title("bitwise_not")

    plt.subplot(2,4, 7)
    bitwise_or = cv2.bitwise_or(night,day)
    plt.imshow(bitwise_or)
    plt.title("bitwise_or")

    plt.subplot(2,4, 8)
    bitwise_xor = cv2.bitwise_xor(night,day)
    plt.imshow(bitwise_xor)
    plt.title("bitwise_xor")
    plt.show()


def histogram_of_image():
    """
    Computes the Histogram, PDF and CDF of the input image
    :return:
    """
    img = cv2.imread("flowers.png", 0)
    imgHW = img.shape
    bins_array = np.zeros(255)

   #Calculating Histogram
    for i in range(0,imgHW[0]):
        for j in range(0,imgHW[1]):
            bins_array[img[i][j]] += 1

    # Calculating PDF
    pdf = bins_array/(imgHW[0] * imgHW[1])

    # Calculating CDF
    cdf = []
    previous = 0
    for i in bins_array:
        previous = previous + i
        cdf.append(previous)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Histogram of Image")
    plt.plot(bins_array)

    plt.subplot(1, 3, 2)
    plt.plot(pdf)
    plt.title("Probability Density Function")

    plt.subplot(1, 3, 3)
    plt.plot(cdf)
    plt.title("Cumulative Density Function")
    plt.show()


def histequalisation():
    im = cv2.imread("hazecity.png",0)
    equalized_img = cv2.equalizeHist(im)
    result = np.hstack((im,equalized_img))
    cv2.imshow('hist_equalize.png',result)
    cv2.waitKey()


def main():
    """
    Prompts the user for a number between 1-7 and
    on given input performs corresponding operations
    :return:
    """

    print("Option 1 : Read,Display and Write image to another format: ")
    print("Option 2 : Display gray scale map of image: ")
    print("Option 3 : Subplot Red, Green and Blue channels of image: ")
    print("Option 4 : Convert gray scale image to binary and display it: ")
    print("Option 5 : Arithmetic operations using RGB channels,gray-scale or binary image: ")
    print("Option 6 : Compute Histogram,PDF,CDF of the image")
    print("Option 7 : Compute Histogram Equalisation of the image")
    print("Option 8 : Exit the program")
    while True:
        try:
            option = int(input("Enter option number to perform that operation :"))
            if option is 1:
                read_display_image()
            elif option is 2:
                convert2grayscale()
            elif option is 3:
                subplotRGB()
            elif option is 4:
                gray2binary()
            elif option is 5:
                arithmeticOps()
            elif option is 6:
                histogram_of_image()
            elif option is 7:
                histequalisation()
            elif option is 8:
                sys.exit()
            else:
                print("Not a valid number")
        except Exception:
            print "Not a valid input"

if __name__ == '__main__':
    main()


