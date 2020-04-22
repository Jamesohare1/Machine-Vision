import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt   


# Task 1: load and display image and target images
input_img  = cv2.imread("Girl_in_front_of_a_green_background.jpg")
target_img = cv2.imread("Tour_Eiffel.jpg")
cv2.imshow("input", input_img)
cv2.imshow("target", target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Convert the input image to HSV, extract Hue channel, and display Hue channel image
hsv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
hue, sat, value = cv2.split(hsv_img)
cv2.imshow("hue", hue)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Task 2 - calculate and display hue channel histogram
hist = cv2.calcHist([hue], [0], None, [256], [0,256])
plt.bar(range(len(hist)), hist.flatten())

# apply thresholds to the hue channel image, calculate a binary mask of the foreground, and display the foreground mask
def foregroundMask(hue, lower, upper):
    new_img = copy.deepcopy(hue)
    for i in range(hue.shape[0]):
        for j in range(hue.shape[1]):
            if hue[i][j] >= lower and hue[i][j] <= upper:
                new_img[i][j] = 0
            else:
                new_img[i][j] = 1
    return new_img       

mask = foregroundMask(hue, 46, 62)
cv2.imshow("mask", mask * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Step 3: cut out the foreground
def cutForeground(image, mask):
    new_img = copy.deepcopy(image)
    for i in range(3):
        new_img[:,:,0] = image[:,:,0] * mask
        new_img[:,:,1] = image[:,:,1] * mask
        new_img[:,:,2] = image[:,:,2] * mask
    return new_img
    
foreground = cutForeground(input_img, mask)
  
#resize the image
foreground = cv2.resize(foreground, (200, 300))

#Place cut-out at the middle bottom of target image
def imposeImage(foreground, target):
    new_img = copy.deepcopy(target)
    height_target = target.shape[0]
    width_target = target.shape[1]
    height_foreground = foreground.shape[0]
    width_foreground = foreground.shape[1]
    HShift = np.int((width_target - width_foreground)/2) 
    VShift = height_target - height_foreground 
    
    for i in range(0, height_foreground):
        for j in range(0, width_foreground):
            if foreground[i, j, 0] > 0 and foreground[i, j, 1] > 0 and foreground[i, j, 2] > 0 :
                new_img[VShift + i, HShift + j, 0] = foreground[i, j, 0]
                new_img[VShift + i, HShift + j, 1] = foreground[i, j, 1]
                new_img[VShift + i, HShift + j, 2] = foreground[i, j, 2]
                
    return new_img
  
finalImage = imposeImage(foreground, target_img)    

cv2.imshow("Final Image", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()