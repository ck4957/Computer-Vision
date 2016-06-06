__author__ = 'Chirag'

num = 5
import numpy as np
import matplotlib.pyplot as plt
'''
r_vect = np.array([[2,3,4]])
c_vect = np.array([[2],[3],[4]])
# Check if row vector and column vector are equal
# .all() is needed because np.equal() is element wise
np.equal(r_vect,c_vect).all()
np.equal(r_vect,c_vect.T).all()

# Specify a range
int_num = np.arange(1,10,dtype=np.uint8)
# With interval
# Dtype should be specified for all non integers
dec_num = np.arange(0.0,11.5,0.5,dtype=np.float64)
# Generate a random matrix
rand_mat33 = np.random.rand(3,3)
# Get the shape of a matrix (y,x,...)
rand_mat33.shape


A = np.array([[0.251083857976031,0.830828627896291,0.285839018820374,0.567821640725221,0.779167230102011],
             [0.616044676146639,0.585264091152724,0.757200229110721,0.0758542895630636,0.934010684229183],
             [0.473288848902729,0.549723608291140,0.753729094278495,0.0539501186666072,0.129906208473730],
             [0.351659507062997,0.917193663829810,0.380445846975357,0.530797553008973,0.568823660872193]])

B = np.array([[0.263802916521990,0.579704587365570,0.622055131485066,0.0759666916908419,0.239952525664903],
             [0.145538980384717,0.549860201836332,0.350952380892271,0.239916153553658,0.417267069084370],
             [0.136068558708664,0.144954798223727,0.513249539867053,0.123318934835166,0.0496544303257421],
             [0.869292207640089,0.853031117721894,0.401808033751942,0.183907788282417,0.902716109915281]])

sum_AB = np.add(A,B)
diff_AB = np.subtract(A,B)
# A dot B isnt possible so use A dot B.T
matrixmul_AB = np.dot(A,B.T)

# r_divide A / B is the solution to the equation xA = B.
# Matrices A and B must have the same number of columns.
r_div = np.linalg.lstsq(A,B)

# Inverse of a matrix and eigen values
C = np.array([[7,10,5],[8,8,5],[1,5,4]])
np.linalg.inv(C)
V,E = np.linalg.eig(C)

D = np.array([[6,8,9],[6,7,6],[9,4,4]])
# Matrix multiplication
matrixmul_CD = np.dot(C,D)

# Element-wise opperations
C_dotpow3 = np.power(C,3)
C_dotmulD = np.multiply(C,D)
C_dotdivD = np.divide(C,D)

# Logical Operation
A = np.array([[2,5,3],[3,6,7]])
np.where(A > 4)
A[A > 4] = 0

# Flow control (remember colon)
val1 = 3
val2 = 3
if val1 > val2:
    print "{} is greater than {}".format(val1,val2)
elif val1 < val2:
    print "{} is lesser than {}".formmat(val1,val2)
else:
    print "{} is equal to {}".format(val1,val2)

# While loops!
var = 0
while var < 5:
    print "incrementing var = {}".format(var)
    var += 1

# Graph 2 lines
x = np.linspace(-2*np.pi,2*np.pi)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1)
plt.plot(x,y2)
plt.title("2-D Line Plot")
plt.xlabel("x in radians")
plt.ylabel("sin(x) and cos(x)")
plt.show()

# A 3d plot (sort of)
from mpl_toolkits.mplot3d import axes3d
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

# Save an array
np.savetxt("x.txt",x)
'''
# Read and display image
import cv2
img = cv2.imread("flowers.png", -1)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.title("RGB Flowers Simon")
#plt.show()

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to decimal
floating = img.astype(np.float64)/255

# Simple image viewer
cv2.namedWindow("Window Name",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Window Name",floating)
cv2.waitKey(0)
cv2.destroyWindow("Window Name")