import cv2
import numpy as np
from matplotlib import pyplot as plt   



# Step 1 - load image file, and display it on screen
img = cv2.imread("cat.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#Show both images
cv2.imshow("image1", img)
cv2.imshow("image2", gray)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()



# Step 2 - Create Sobel filter mask to detect edges in x- and in y-direction. 
sobel_kernel_x = np.array([[1,0,-1], 
                           [2,0,-2], 
                           [1,0,-1]])    
sobel_kernel_y = np.array([[1,  2, 1],
                           [0,  0, 0], 
                           [-1,-2,-1]])
sobel_x = cv2.filter2D(gray, -1, sobel_kernel_x)
sobel_y = cv2.filter2D(gray, -1, sobel_kernel_y)

cv2.imshow("image1", sobel_x)
cv2.imshow("image2", sobel_y)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()



# Step 3 - Create filter masks for two derivatives

#mesh grid returns 2 matrices
# returns coorinates of evenly spaced points on a 2D coordinate system
# x matrix contains x coordinates of points
# y matrix contains y coordinates of points
def getKernal(img, sigma):
    x , y = np.meshgrid(np.arange(0, 7 * sigma), np.arange(0, 7 * sigma))
    
    # the derivatives are the derivatives of the guassian bell function
    # we subtract len(x)/3 to move centre of grid to the ogigin
    # e.g. if the grid is a 6 * 6 we want grid from -3,-3 to 3,3
    kernel_x = -(x-len(x)/2) / (2*np.pi*sigma**4)  * np.exp(-((x-len(x)/2)**2 + (y-len(x)/2)**2) / (2*sigma**2))     
    kernel_y = -(y-len(y)/2) / (2*np.pi*sigma**4)  * np.exp(-((x-len(x)/2)**2 + (y-len(x)/2)**2) / (2*sigma**2))  
    return kernel_x, kernel_y

sigma = 1 #sigma is essentially a smoothing factor
kernal_x, kernal_y = getKernal(gray, sigma)
cat_x = cv2.filter2D(gray, -1, kernal_x)    
cat_y = cv2.filter2D(gray, -1, kernal_y)

cv2.imshow("dog_x", cat_x/np.max(cat_x))
cv2.imshow("dog_y", cat_y/np.max(cat_y))
cv2.waitKey(0)    
cv2.destroyAllWindows()



# Step 4 -Calculate the fft and display absolute values of spectrum as an image
ft = np.fft.fft2(gray)    
cv2.imshow("ft",np.fft.fftshift(abs(ft))/np.max(abs(ft))*255)   
 
inv_ft = np.fft.ifft2(ft)    
cv2.imshow("inv ft",abs(inv_ft)/255)
    
cv2.waitKey(0)    
cv2.destroyAllWindows() 
