__author__ = 'Chirag'


print("Chirag Narendra Kular")
print("ck4957@rit.edu")
print("-----------------------------------------------")
import cv2
print("Open CV version :", cv2.__version__, "\n")

import sys
print("Python version: :", sys.version, "\n")

import matplotlib
print("Matplotlib version :", matplotlib.__version__, "\n")
print("-----------------------------------------------")

class Solution(object):
    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        count_ones = 0
        for i in range(n+1):
            if '1' in str(i):
                count_ones += str(i).count('1')
        return count_ones

s = Solution()
print(s.countDigitOne(13))

'''
    imgHW = im.shape
    bins_array = np.zeros(255)
    #print len(bins_array)
    ## Calculating PDF
    for i in range(0,imgHW[0]):
        for j in range(0,imgHW[1]):
            bins_array[im[i][j]] += 1

    # Calculating CDF
    #print len(bins_array)
    cdf = []
    previous = 0
    for i in bins_array:
        previous = previous + i
        cdf.append(previous)
    d = 1.0/imgHW[0]/imgHW[1]
    h = np.zeros(255)
    for i in range(0,imgHW[0]):
        for j in range(0,imgHW[1]):
            h[im[i][j]] += d
    #print len(h)

    lookup = np.zeros(255)
    sum = 0.0
    for i in range(0,255):
        sum += h[i]
        lookup[i]= (sum * 255 + 0.5)
        #h.append((cdf[-1])/(((imgHW[0]*imgHW[1])-1))*255)
    #print lookup
    print "Here",imgHW[0],imgHW[1]

    #y = [0 for _ in imgHW[0]][0 for _ in imgHW[1]]
    #y = [[0 for i in range(imgHW[0])] for j in range(imgHW[1])]
    y = np.empty((imgHW[0],imgHW[1]))
    print len(y)
    for i in range(0,imgHW[0]):
        for j in range(0,imgHW[1]):
            im[i][j] = lookup[im[i][j]]
    #cv2.imshow("Result",im)
    #cv2.waitKey()

    normalized = 255 * cdf / cdf[-1]
    print normalized
    integral = 0
    for i in normalized:
        integral += i
    print integral
    histe = integral * im
    cv2.imshow("qwerty", histe)
    cv2.waitKey()
    '''