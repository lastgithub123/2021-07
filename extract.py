import cv2 
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def extractBBox(image):
    boxes = []
    bb=[]
    transform = transforms.ToTensor()

    new_image = image.copy()
    #new_image = new_image[100:,:]

    gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY) # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5,5), 0) # apply gaussian blur to blur the background
    thresh0 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
    contours, _ = cv2.findContours(thresh0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)




    #with_contours = cv2.drawContours(new_image, contours, -1,(0,255,0),1)

    #print('Total number of contours detected: ' + str(len(contours)))

    count = 0
    images = []
    for i,contour in enumerate(contours):
        count+=1
        area = cv2.contourArea(contour)
        #print('area ************: ', area)
        if area < 2000:       
            continue
        #elif area > 12000:
        #    continue
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        #print(str(w)+'======================='+str(h))
        
        #if w < 20 or h < 20:
            #continue
        #if w > 2*h:
            #continue
        print(str(rect[1]),str(rect[3]),str(rect[0]),str(rect[2]))
        img = new_image[y:y+h,x:x+w]
        #print(img)
        print('Bounding Box: ',count)
        cv2.imshow('Gray image', img)
        
        cv2.waitKey(0) # Wait for keypress to continue

        cv2.destroyAllWindows()
        predImg = img.copy()
        images.append(predImg)


        img = cv2.resize(img,(32,32))
        #cv2.imshow('Gray image', img)  


        img = Image.fromarray(img)
        img = transform(img)
        img.requires_grad=False
        bb.append(img)



        draw_contour = cv2.drawContours(new_image, contours, i,(255,0,0), 2)
        cv2.rectangle(draw_contour,(x,y), (x+w,y+h), (0,0,255), 2)

        boxes.append(np.array(rect))


    print(boxes)
    print(len(images))
    print(len(bb))
    #print(bb)
    return boxes,bb,images
