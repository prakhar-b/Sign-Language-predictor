# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:02:55 2020

@author: prakhar
"""
# import the necessary packages
import imutils
import cv2
import pickle
#import joblib
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier

bg = None
x, y, w, h = 300, 100, 300, 300
text = ""
def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
    pix = image_to_feature_vector(save_img)
    #hi = extract_color_histogram(save_img)
    pix = pix.reshape(1, -1)
    #hi = hi.reshape(1, -1)
    predicted = model.predict(pix)
    #probablity = max(KNeighborsClassifier.predict_proba(model, pix))
    #print(probablity)
    #print(predicted)
    text = predicted
    #if probablity*100 > 70:
    #    text = predicted
    return text

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()

def run_avg(image, aWeight):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = gray[y:y+h, x:x+w]
    global bg
    # initialize the background
    if bg is None:
        bg = gray.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(gray, bg, aWeight)

def segment(image, threshold=25):
    global bg
    #print("I'm here")
    # find the absolute difference between background and current frame
    image_roi = image[y:y+h, x:x+w]
    diff = cv2.absdiff(bg.astype("uint8"), image_roi)
    #print("I'm here 2")
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("I'm here 3")
    # return None, if no contours detected
    if len(cnts) == 0:
    #    print("I'm here 4")
        return(image,cnts,thresholded)
    else:
    #    print("I'm here 5")
        # based on contour area, get the maximum contour which is the hand
        segmented = cnts
        return (image,segmented,thresholded)

pkl_filename = 'modelk5.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# joblib_filename = 'joblib_KNN_3.pkl'
# with open(joblib_filename, 'rb') as file:
#     model = joblib.load(file)

#model = joblib.load('saved_model_VIT.pkl')
  
num_frames = 0
aWeight = 0.25
cam = cv2.VideoCapture(0)
loop_flag = 0
prev_text = None
text_counter = 0
global_text = ""
while (True):
    ret, frame = cam.read()        
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    img_cpy = frame.copy()
    if num_frames<100 :
        if num_frames == 1 :
            print("[INFO]:Looking for background...")
        cv2.imshow("Input",frame)
        run_avg(frame, aWeight)
        num_frames = num_frames + 1
        #print(num_frames)
    else:
        if loop_flag == 0:
            print("[SUCCESS]:Background scanning successful!")
            loop_flag = 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        img, contours, thresh = segment(gray)
        cv2.rectangle(frame,(300,100), (600,400), (0,255,0), 2)
        cv2.imshow("Input",frame)
        if len(contours)>0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if prev_text is None:
                    prev_text = text
                else:
                    if prev_text == text :
                        text_counter = text_counter + 1
                        prev_text = text
                    else:
                        #print("Current sign is:" + str(text))
                        prev_text = text
                        
                if text_counter > 25 :
                    #global_text = str(global_text + str(text))
                    global_text = text
                    print("Global text is:" + str(global_text))
                    #print("Current sign is:" + str(text))
                    text_counter = 0
                    prev_text = None
            roi = frame[y:y+h, x:x+w]
            uintbg = (bg.astype(np.uint8))
            cv2.imshow("ROI", roi)
            cv2.imshow("Background" , uintbg)
            cv2.imshow("Foreground=ROI-Background", thresh)
            #pix = image_to_feature_vector(thresh)
            #hi = extract_color_histogram(frame)
            #pix = pix.reshape(1, -1)
            #hi = hi.reshape(1, -1)
            #predicted = model.predict(pix)
            #print(predicted)
        k = cv2.waitKey(1)   
        if k%256 == 27:
            print("[INFO]: Esc HIT: Closing...")
            break

cam.release()
cv2.destroyAllWindows()