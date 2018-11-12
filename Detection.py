#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os

MAX_WIDTH = 500
height = 5.5
width = 8.5
threshold = 0.4
ratio = width/height

folder = "images"
path = os.listdir(folder)

target_path="results1/"      
if not os.path.exists(target_path):  
    os.makedirs(target_path)

for image in path:
    img = cv2.imread(folder+"/"+image)
    h, w, channels = img.shape
    if w > MAX_WIDTH:
        resize_rate = MAX_WIDTH / w	
        img = cv2.resize(img, (MAX_WIDTH, int(h*resize_rate)), interpolation=cv2.INTER_AREA)
    h, w, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)   # gaussian filtering
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #binary

#when the background is brighter than foreground, use binary inverse
#so that the value of background is 0 and foregroud if 255
    judge = 0
    if binary[0,0]==255:
        judge = judge + 1
    if binary[0,w-1]==255:
        judge = judge + 1
    if binary[h-1,0]==255:
        judge = judge + 1
    if binary[h-1,w-1]==255:
        judge = judge + 1
    if judge>=3:
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Projection
    up = 0
    down = h
    left = 0
    right = w
    maxh = 0
    maxw = 0
    
    hsum = np.sum(binary,axis=1)  # sum of each row
    
    for y in range (0,h):
        if (hsum[y]>maxh):
            maxh = hsum[y]
    
#determine up and down edge of the credit card.
    for y in range(h-1, 0,-1):
        if (hsum[y])>maxh*threshold:  #as there are noises, set a threshold
            down = y
            break
    
    for y in range(0, h):
        if (hsum[y])>maxh*threshold:
            up = y
            break
    
# do the same on each row
# determine right and left edge of the credit card.
    wsum = np.sum(binary[up:down,:],axis=0) #sum of each column
    
    for x in range (0,w):
        if (wsum[x]>maxw):
            maxw = wsum[x]

    for x in range(0, w):
        if (wsum[x]>maxw*threshold):
            left = x
            break
        
    for x in range(w-1, 0, -1):
        if (wsum[x]>maxw*threshold):
            right = x
            break
   
# as the ratio of width and height is 1.6 for credit card
# when the ratio is far away from 1.6, increase width or height to improve the results
    r = (right-left)/(down-up)
    if r<1.2:
        neww = int((down-up)*ratio)
        addw = neww - (right-left)
        if neww >= w:
            right = w-1
            left = 0
        else:
            if (right+addw//2<w):
                if (left-addw//2>0):
                    right = right+addw//2
                    left = left-addw//2
                else:
                    left = 0
                    right = neww
                    print(addw)
            else:
                
                right = w-1
                left = w - addw
    elif r>1.9:
        newh = int((right-left)//ratio)
        addh = newh-(down-up)
        if newh >= h:
            up = 0
            down = h-1
        else:
            if (down+addh//2<h):
                if (up-addh//2>0):
                    down = down+addh//2
                    up = up-addh//2
                else:
                    up = 0
                    down = newh
            else:
                down = h-1
                up = h - addh  

#use grabCut and get the foreground image
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (up,left,right-left,down-up)   #the rectangle is obtained by projection
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

#the rectangle maybe too small to cover the credit card
    size = np.sum((mask==1)|(mask==3))
    if size*2>(right-left)*(down-up):  #increase width or height of rectangle again
        edge = w//8
        right = min(w-1,right+edge)
        left = max(0,left-edge)
        up = max(0,up-edge)
        down = min(h-1,down+edge)
        print(up,down,left,right) 
        rect = (up,left,right-left,down-up)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)  #Redo grabcut
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    dst = gray[up:down, left:right]
    
#remove the noises
    mask3 = mask2*255
    img_edge1 = cv2.Canny(mask3, 100, 200)
    image, contours, hierarchy = cv2.findContours(mask3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask3, contours, -1, (0, 0, 255), 1)
    mask4 = np.where((mask3==0),0,1).astype('uint8')
    img_dst1 = img*mask4[:,:,np.newaxis]

#extract the region and use canny edge detection
    img_edge2 = cv2.Canny(img_dst1, 100, 200)
    image, contours, hierarchy = cv2.findContours(img_edge2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Overlay the contour on the raw image
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
 
    cv2.imshow("img_edge1", img_edge1)
    cv2.imshow("mask3", mask3)
    cv2.imshow("binary", binary)
    cv2.imshow("projection", dst) 
    cv2.imshow("grubcut",img_dst1)
    cv2.imshow("creditcard", img_edge2)
    cv2.imshow("result", img)
    
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
print ("no image to detect")
