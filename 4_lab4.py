import cv2
import numpy as np
#from matplotlib import pyplot as plt   


def displayImage(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Task 1: load and display image
house_img  = cv2.imread("house.png")
displayImage("house", house_img)

#convert to grayscale
gray = cv2.cvtColor(house_img, cv2.COLOR_RGB2GRAY)
displayImage("gray", gray)


# Task 2:
#Calculate and display the two gradient images using convolutions with a derivative of Gaussian kernels.
def getKernal(img, sigma):
    x , y = np.meshgrid(np.arange(0, len(img)), np.arange(0, len(img)))
    kernel_x = -(x-len(x)/2) / (2*np.pi*sigma**4)  * np.exp(-((x-len(x)/2)**2 + (y-len(x)/2)**2) / (2*sigma**2))     
    kernel_y = -(y-len(y)/2) / (2*np.pi*sigma**4)  * np.exp(-((x-len(x)/2)**2 + (y-len(x)/2)**2) / (2*sigma**2))  
    return kernel_x, kernel_y

sigma_d = 1
kernal_x, kernal_y = getKernal(gray, sigma_d)
house_x = cv2.filter2D(gray, -1, kernal_x)    
house_y = cv2.filter2D(gray, -1, kernal_y)
displayImage("house x", house_x/np.max(house_x))
displayImage("house y", house_y/np.max(house_y))


#Task 3:
#Calculate for each pixel p in the image the weighted structure tensor
sigma_w = 2
x_squared = house_x * house_x
y_squared = house_y * house_y
xy = house_x * house_y 

J11 = cv2.boxFilter(x_squared, cv2.CV_32F, (sigma_w, sigma_w))
J22 = cv2.boxFilter(y_squared, cv2.CV_32F, (sigma_w, sigma_w))
J12 = cv2.boxFilter(xy, cv2.CV_32F, (sigma_w ,sigma_w))
    

#Task 4:
#Calculate the eigenvalues for each structure tensor calculated in task 3
eigenvalues1 = np.zeros((205,286))
eigenvalues2 = np.zeros((205,286))
for i in range(J11.shape[0]):
    for j in range(J11.shape[1]):   
        matrix = np.matrix([[J11[i,j], J12[i,j]],[J12[i,j], J22[i,j]]])
        eigen = np.linalg.eigvals(matrix)
        eigenvalues1[i,j] = eigen[0]
        eigenvalues2[i,j] = eigen[1]

eigenMin = np.minimum(eigenvalues1, eigenvalues2)
displayImage("Minimum EigenValues", eigenMin)


#Task 5:
#Threshold the matrix calculated in task 4 and suppress all non-maxima      
img = cv2.imread('house.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
img[eigenMin>0.02*eigenMin.max()]=[0,0,255]
displayImage('dst', img)


#------------------------------------------------------------------------------

#Using cv2
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#Harris corner detection
img = cv2.imread('house.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
harris = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
harris = cv2.dilate(harris,None)
# Threshold for an optimal value, it may vary depending on the image.
img[harris>0.01*harris.max()]=[0,0,255]
displayImage('dst', img)


#Shi Tomasi detection
img = cv2.imread('house.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
shi = cv2.cornerMinEigenVal(gray, 10)
shi = cv2.dilate(shi,None)
img[shi>0.01*shi.max()]=[0,0,255]
displayImage('dst', img)
