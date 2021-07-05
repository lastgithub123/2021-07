import cv2
import numpy as np
from random import *
import cnn
from PIL import Image
#from extract import extractBoundingBox
from extract0704 import extractBBox

import torch
import torchvision.transforms as transforms


class ImageRecClass():
    def __init__(self):
        self.net = cnn.CNN()
        

        self.net.loadModel('newCNNmodel.pkl')
        
        self.classes = ['0','6','7','8','9','Down','Left','Right','Stop','Up','V','W','X','Y','Z']
        #self.transform = transforms.ToTensor()


    def predict(self,img):

        rects,bb,predImg = extractBBox(img)


        filtered = []
        if(len(bb)) != 0:
                            
            boxs = torch.stack(bb)
  
            predictions = self.net(boxs)


            print(torch.max(predictions, 1))
            values, predictions = torch.max(predictions, 1)
            score, index = values.max(0)
            
            # if score >= 0.5:
            print('Score')
            print(int(predictions[index]))
            print(score, self.classes[int(predictions[index])])
            
            #print(score, self.classes[int(predictions[index])])
            if score < 0.97:
                #print(score, self.classes[int(predictions[index])])
                print(score, self.classes[int(predictions[index])])
                #return None, -1
                return None,None,-1

            #return rects[index], self.classes[int(predictions[index])]
            return (rects[index], self.classes[predictions[index]], score)

        else:
            #return None, -1
            return None,None,-1
        
