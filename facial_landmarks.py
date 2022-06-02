# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time

import csv 
from scipy.spatial import distance as dist
import math
import os


# construct the argument parser and parse the arguments - ORIGINAL
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())



def dir_filter(lst): 
	if '.DS_Store' in lst:
		lst.remove('.DS_Store')
	return lst

#Construct new argument parser 
a = argparse.ArgumentParser()
a.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
a.add_argument("--box-file", required=True,
	help="path to box file")
args = vars(a.parse_args())

#AUTOMATION PROCESS - CSV FILES
# ['Patient_ID', 'Nose Angle', 'Mouth Angle', 'Eyebrow Angle']
fields = ['Patient ID', 'Time Point', 'Test Batch', 'Nose Angle', 'Mouth Angle', 'Eyebrow Angle', 'Left EAR', 'Right EAR', 'Left Eye % Open', 'Right Eye % Open', 'Matching Shapes: L/R Eye % Difference']

	

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])





# load the input image, resize it, and convert it to grayscale - ORIGINAL
# image = cv2.imread(args["image"]) ##GET FROM FOLDER
# image = imutils.resize(image, width=500)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # detect faces in the grayscale image
# rects = detector(gray, 1)

# "Box_Folder/Patient_ID/Timepoint/Test"
ogpath = args["box_file"]
# Go through each patient ...


with open('patient_output.csv', 'w') as cvsfile: 
	#create csv writer obj
	writer = csv.writer(cvsfile)
	writer.writerow(fields)
					
	direct = dir_filter(os.listdir(ogpath))
	for patient in direct: 
		path1 = ogpath + '/' + str(patient)
		direct = dir_filter(os.listdir(path1))
		for timepoint in direct: 
			path2 = path1 + '/' + str(timepoint)
			direct = dir_filter(os.listdir(path2))
			for test in direct:
				path3 = path2 + '/' + str(test)
				print(path3)
				for jpg in os.listdir(path3):
					if ('processed_' in jpg): 
						continue
					path4 = path3 + '/' + str(jpg)
					face = [patient, timepoint, test]
					image = cv2.imread(path4)
					image = imutils.resize(image, width=500)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					# detect faces in the grayscale image
					rects = detector(gray, 1)

					# loop over the face detections
					for (i, rect) in enumerate(rects):
						# determine the facial landmarks for the face region, then
						# convert the facial landmark (x, y)-coordinates to a NumPy
						# array
						shape = predictor(gray, rect)
						shape = face_utils.shape_to_np(shape)
						# convert dlib's rectangle to a OpenCV-style bounding box
						# [i.e., (x, y, w, h)], then draw the face bounding box
						(x, y, w, h) = face_utils.rect_to_bb(rect)
						cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
						# show the face number
						cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
						# loop over the (x, y)-coordinates for the facial landmarks
						# and draw them on the image
						for (x, y) in shape:
							cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

					

						## NOSE CALCULATIONS START ##
						#Nose bridge line 
						cv2.line(image,(shape[27][0], shape[27][1]), (shape[33][0],shape[33][1]), (255,0,0), 1)
						n = dist.euclidean((shape[27][0], shape[27][1]), (shape[33][0],shape[33][1]))

						#Nose bridge on y axis; might need to fix later
						cv2.line(image, (shape[27][0], shape[27][1]), (shape[27][0], shape[27][1] + 100), color=(0,0,255))
						ny = dist.euclidean((shape[27][0], shape[27][1]), (shape[27][0], shape[27][1] + 100))

						#Calculation of angle changes
						a = (shape[33][0] - shape[27][0], shape[33][1] - shape[27][1])
						b = (shape[27][0] - shape[27][0], (shape[27][1] + 100) - shape[27][1])
						dot_product = np.dot(a, b)
						dist_product = np.linalg.norm(a) * np.linalg.norm(b)
						angle = math.acos(dot_product / dist_product)
						print("Face " + str(i+1) + ": Nose Angle Deviation")
						print(angle)
						face.append(angle)
						## NOSE CALCULATIONS END ##
						

						## MOUTH CALCULATIONS START ##
						#Mouth Corners 
						cv2.line(image,(shape[48][0], shape[48][1]), (shape[54][0],shape[54][1]), (255,0,0), 1)
						#Horizontal Mouth 
						cv2.line(image, (shape[48][0], shape[48][1]), (shape[48][0] + 80, shape[48][1]), color=(0,0,255))
						#Mouth Calculations
						a = (shape[54][0] - shape[48][0], shape[54][1] - shape[48][1])
						b = ((shape[48][0] + 80) - shape[48][0], shape[48][1] - shape[48][1])
						dot_product = np.dot(a, b)
						dist_product = np.linalg.norm(a) * np.linalg.norm(b)
						angle = math.acos(dot_product / dist_product)
						print("Face " + str(i+1) + ": Mouth Angle Deviation")
						print(angle)
						face.append(angle)
						## MOUTH CALCULATIONS END ##

						## EYEBROW CALCULATIONS START ##
						#Eyebrows
						cv2.line(image,(shape[19][0], shape[19][1]), (shape[24][0],shape[24][1]), (255,0,0), 1)
						#Horizontal Eyebrows 
						cv2.line(image,(shape[19][0], shape[19][1]), (shape[19][0] + 100,shape[19][1]), (255,0,0), 1)
						#Eyebrow Calculations
						a = (shape[24][0] - shape[19][0], shape[24][1] - shape[19][1])
						b = ((shape[19][0] + 100) - shape[19][0], shape[19][1] - shape[19][1])
						dot_product = np.dot(a, b)
						dist_product = np.linalg.norm(a) * np.linalg.norm(b)
						angle = math.acos(dot_product / dist_product)
						print("Face " + str(i+1) + ": Eyebrow Angle Deviation")
						print(angle)
						face.append(angle)
						## EYEBROW CALCULATIONS END ##


						current_dir = os.getcwd()
						os.chdir(path3)
						cv2.imwrite('processed_'+str(jpg), image)
						os.chdir(current_dir)
						
						print("\n")
						print("Additional Measurements")


						#TESTER RESTART
						# img = cv2.imread('/Users/annimao/FaceTracker/new.jpeg')
						# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
						# gray = cv2.bilateralFilter(gray, 11, 17, 17)
						# edged = cv2.Canny(gray, 30, 200)
						# # cv2.imshow("outline_image", edged)
						# contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						# contours = imutils.grab_contours(contours)



						left_eye = [[shape[36][0], shape[36][1]], [shape[37][0], shape[37][1]], [shape[38][0],
						shape[38][1]], [shape[39][0], shape[39][1]], [shape[40][0], shape[40][1]], [shape[41][0], shape[41][1]]]
						right_eye = [[shape[42][0], shape[42][1]], [shape[43][0], shape[43][1]], [shape[44][0], 
						shape[44][1]], [shape[45][0], shape[45][1]], [shape[46][0], shape[46][1]], [shape[47][0], shape[47][1]]]
						#Eye Area Ratio (EAR)
						def ear(eye):
							A = dist.euclidean(eye[1], eye[5])
							B = dist.euclidean(eye[2], eye[4])
							C = dist.euclidean(eye[0], eye[3])

							ear = (A + B) / (2.0 * C)
							return ear
						l_EAR = ear(left_eye)
						r_EAR = ear(right_eye)
						print("Left Eye Area Ratio: " + str(l_EAR))
						print("Right Eye Area Ratio: " + str(r_EAR))
						face.append(l_EAR)
						face.append(r_EAR)

						converted1 = (1 - ear(left_eye)) * 100 
						converted2 = (1 - ear(right_eye)) * 100 
						print("Left EAR -> " + str(converted1) + "% of the Left Eye is Opened")
						print("Right EAR -> " + str(converted2) + "% of the Right Eye is Opened")
						face.append(converted1)
						face.append(converted2)


						##TEST LEFT 
						cont = [np.array(left_eye, dtype=np.int32)]
						drawing = np.zeros([1000, 1000],np.uint8)
						for cnt in cont:
							cv2.drawContours(drawing,[cnt],0,(255,255,255),2)
						# cv2.imshow('output_lefteye',drawing)
						c1 = cv2.findContours(drawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						c1 = np.array(c1[0])
						# print("Area for Left Eye w/ Self Drawn Contours:")
						# print(cv2.contourArea(c1[0]))


						##TEST RIGHT
						cont1 = [np.array(right_eye, dtype=np.int32)]
						drawing1 = np.zeros([1000, 1000],np.uint8)
						for cnt in cont1:
							cv2.drawContours(drawing1,[cnt],0,(255,255,255),2)
						# cv2.imshow('output_righteye',drawing1)
						c2 = cv2.findContours(drawing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						c2 = np.array(c2[0])
						# print("Area for Right Eye w/ Self Drawn Contours:")
						# print(cv2.contourArea(c2[0]))


						print("Matching Shape:")
						print(cv2.matchShapes(c1[0], c2[0], 1, 0))
						# converted3 = (1 - cv2.matchShapes(c1[0], c2[0], 1, 0)) * 100
						converted3 = cv2.matchShapes(c1[0], c2[0], 1, 0) * 100
						print("Matching Shapes -> Left & Right Eye Differ by " + str(converted3) + "%")
						print("\n")
						face.append(converted3)



					# show the output image with the face detections + facial landmarks
					# difference = np.subtract(face2, face1)
					# with open('patient_output.csv', 'w') as cvsfile: 
					# 	#create csv writer obj
					# 	writer = csv.writer(cvsfile)

					# 	writer.writerow(fields)
					# 	writer.writerow(face)
					writer.writerow(face)
					
					cv2.imshow("Output", image)
	# cv2.waitKey(0)


## ORIGINAL PYTHON COMMAND
# python3 facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat \
# --image paralysis1.jpg



## Format 
## Input Folder --> Subfolders of patient names --> subfolder of different timestamps --> jpg/png images 