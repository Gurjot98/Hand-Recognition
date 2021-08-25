# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:58:32 2020

@author: HP
"""

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
while(1):
    
    try:
        
        
        ret,frame=cap.read()
    
        roi=frame[100:300,100:300]
    
        kernel = np.ones((3,3),np.uint8)
        
        
    
        hsv  = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert to grayscale

        cv2.rectangle(frame,(100,100),(300,300),(0,0,255),2)

        blur = cv2.blur(hsv, (3, 3)) # blur the image

        ls=np.array([0,20,70],dtype=np.uint8)
        ups=np.array([20,255,255],dtype=np.uint8)
    
        mask=cv2.inRange(hsv,ls,ups)
        
        mask = cv2.dilate(mask,kernel,iterations = 4)

        mask = cv2.GaussianBlur(mask,(5,5),100) 


        img ,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# create hull array for convex hull points

        hull = []

        cnt=max(contours,key =lambda x: cv2.contourArea(x))


    #epsilon required for approx polydp function
        epsilon=0.0005*cv2.arcLength(cnt,1)
    
        approx=cv2.approxPolyDP(cnt,epsilon,1)

        hull=cv2.convexHull(cnt)
    
        areh=cv2.contourArea(hull)
        arec=cv2.contourArea(cnt)
    
        ario=((areh-arec)/arec)*100
    
    #print(ario)
    
        hull=cv2.convexHull(approx,returnPoints=False)
   
        de=cv2.convexityDefects(approx,hull)
   
        count=0
   
        for i in range(de.shape[0]):
                s,e,f,d = de[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt=(100,180)
    
                a=math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2 )
                b=math.sqrt((start[0]-far[0])**2 + (start[1]-far[1])**2 )
                c=math.sqrt((far[0]-end[0])**2 + (far[1]-end[1])**2 )
            
                s=(a+b+c)/2
                ar=math.sqrt(s*(s-a)*(s-b)*(s-c))
            
                d=(2*ar)/a
            
                angle=math.acos((b**2+c**2-a**2)/(2*b*c))*57
            
                if angle<90 and d>30:
                    count+=1
                    cv2.circle(roi,far,3,(0,0,234),-1)
                
                cv2.line(roi,start,end,(255,0,0),2)
            
        count+=1    
        g=0
        for y in range(1,3):
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            if count==1:
                if arec<2000:
                    cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    if ario<12:
                        cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        g=g+0
                    elif ario<17.5:
                        cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                    else:
                        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        g=g+1
            elif count==2:
                cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                g=g+2            
            elif count==3:
         
                  if ario<27:
                        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        g=g+3
                  else:
                        cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
            elif count==4:
                cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                g=g+4
            elif count==5:
                cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                g=g+5            
            elif count==6:
                cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            else :
                cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        #cv2.putText(frame,g,(0,150),2,(0,0,234),3,cv2.LINE_AA)    
   
        #show the windows
       # cv2.imshow('mask',mask)
    #    cv2.imshow('frame',frame)
   # except:
  #      pass
        
    
 #   k = cv2.waitKey(5) & 0xFF
  #  if k == 27:
   #     break
    
#cv2.destroyAllWindows()
#cap.release()    
      
                
                
   



# calculate points for each contour

    #for i in range(len(contours)):

    # creating convex hull object for each contour

     #   hull.append(cv2.convexHull(contours[i], False))
        
    #drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

 

# draw contours and hull points

    #for i in range(len(contours)):

        #color_contours = (0, 255, 0) # green - color for contours

        #color = (255, 0, 0) # blue - color for convex hull

    # draw ith contour

#        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)

    # draw ith convex hull object

 #       cv2.drawContours(drawing, hull, i, color, 1, 8)
    
            cv2.imshow('frame',frame)
            cv2.imshow('mask',mask)
    except:
            pass
   #         cv2.imshow('drawing',drawing)
    #        cv2.imshow('mask',mask)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break
cv2.destroyAllWindows()   
cap.release()