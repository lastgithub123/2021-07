import time, pickle, requests, threading,os,glob
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
from imgRecFile import ImageRecClass


class takePhoto:
    def __init__(self):
    #self.count = 0

        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30

        print('Warming up camera...')
        time.sleep(2)
        print('Camera warmed up and ready')

        self.correctIdx = []

        self.imgRec = ImageRecClass()
        
        
        
        ###############################################
        self.count = 0



    def capturePic(self):
        run_timestamp = datetime.now().isoformat()
        os.makedirs('logs', exist_ok=True)
        logfile = open(os.path.join('logs', 'rpilog_' + run_timestamp + '.txt'), 'a+')


        #Clear previous images
        imagedir = "/home/pi/RPi_v2/correct_images/"

        if not os.path.exists(imagedir):
            os.makedirs(imagedir)
        
        #test = os.listdir(imagedir)
        #for f in test:
        #    if f.endswith(".jpg"):
        #        os.remove(os.path.join(imagedir, f))

        filelist = glob.glob(os.path.join(imagedir, "*.jpg"))
        for f in filelist:
            os.remove(f)


        try:
            while True:
                time.sleep(1)
                rawCapture = PiRGBArray(self.camera)
                self.camera.start_preview()
                self.camera.capture(rawCapture, format='bgr',use_video_port=True)
                image = rawCapture.array
                print(len(image))
                

                img1 = image[100:, 0:150, :]
                img2 = image[100:, 150:260, :]
                img3 = image[100:, 250:416, :]
                print(len(img1))


                rect1, leftprediction = self.imgRec.predict(img1)
                rect2, midprediction = self.imgRec.predict(img2)
                rect3, rightprediction = self.imgRec.predict(img3)

                rects = []
                rects.append(rect1)
                rects.append(rect2)
                rects.append(rect3)

                ids = []
                ids.append(leftprediction)
                ids.append(midprediction)
                ids.append(rightprediction)

                print(rects, ids)

                shift = [0, 150, 250]
                
                

                for i, rect in enumerate(rects):
                    img_copy = image.copy()

                    if rect is not None:
                        cv2.rectangle(
                            img_copy, (shift[i] + rect[0], 100 + rect[1]), (shift[i] + rect[2], 100 + rect[3]), (0, 0, 255), 2)
                        # image = cv2.rectangle(
                        #     image, (i * 120 + rect[0], 120 + rect[1]), (i * 120 + rect[2], 120 + rect[3]), (0, 255, 0), 2)
                    if ids[i] > 0:  # Don't save those that default return 0
                        # Following Algo - save first instance instead of overwriting
                        if ids[i] not in self.correctIdx:

                            cv2.imwrite(os.path.join(imagedir, str(ids[i]) + '.jpg'), img_copy)
                            self.correctIdx.append(ids[i])
                            print("Image saved: " + str(ids[i]) + '.jpg')

                data = {'com': 'Image Taken', 'left': leftprediction,
                        'middle': midprediction, 'right': rightprediction}
                #commsList[APPLET].write(json.dumps(data))
                print('Left Prediction: ', leftprediction)
                print('Middle Prediction: ', midprediction)
                print('Right Prediction: ', rightprediction)
                
                
                ##################################################################################
                #self.camera.start_preview()
                ##################################################################################
        except Exception as e:
            print('Error: '+str(e))
            
            
if __name__== "__main__":
    takePC = takePhoto()
    takePC.capturePic()
    
