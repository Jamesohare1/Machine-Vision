import numpy as np
import cv2
import copy

def main():

    camera = cv2.VideoCapture(0)
    
    #capture initial image and find initial interest points
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray, 200, 0.3, 7)
    
       
    #begin video 
    while(True):
        
        #capture new image and store previous image variable
        old_gray = copy.deepcopy(gray)
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None) 
        p0 = p1[st==1].reshape(-1,1,2)      
        
        #if number of interest points falls below a certain level
        #calculate new points and discard points too close to existing points
        if len(p0) < 100:
            new_p0 = cv2.goodFeaturesToTrack(gray, 200 - len(p0), 0.3, 7)
            
            for i in new_p0:
                discard = False
                x,y = i.ravel()
                for j in p0:
                    x1, y1 = j.ravel()
                    
                    if np.sqrt((x-x1)**2 + (y-y1)**2) < 10:
                        discard = True
                        
                if discard == False:
                    p0 = np.concatenate((p0,[[[x,y]]]))
 
        #plot interest points on the video image 
        for i in p0:
            x,y = i.ravel()
            cv2.circle(img,(x,y), 4, (0, 0, 255), -1)
        
        #show the video image
        cv2.imshow("camera", img)
        k = cv2.waitKey(1)
        if k%256==27:
            break
    
    camera.release()
    cv2.destroyAllWindows()


main()
