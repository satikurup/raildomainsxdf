

# import standard libraries
import multiprocessing
import detection 
import cv2 as cv
import cv2
#import argparse
import sys
import numpy as np
import dlib
from imutils.video import FPS
import safetyline as saf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import seaborn as sns
import math
import multiprocessing 
import time
#from itertools import count
#import random

# import libraries for tracking
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
skip_frames = 120     #No. of frames skipped for next detection
Distance = 200
calculateConstant_x = 300
calculateConstant_y = 615
position = dict()
highRisk = set()
mediumRisk = set()   
peopleCount = []     

#COLORS
GREEN= (0,255,0)
RED= (0,0,255)
YELLOW= (0,255,255)
WHITE= (255,255,255)
ORANGE= (0,165,255)
BLUE= (255,0,0)
GREY= (192,192,192)


# Load names of classes
classesFile = "model/coco1.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "model/yolov3.cfg";
modelWeights = "model/yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Get the names of the output layers

def getOutputsNames(net):
    #begin = time.time()
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    #end = time.time()
    #print("net time: ", end-begin)  

def calculateCentroid(xmin,ymin,xmax,ymax):

    xmid = ((xmax+xmin)/2)
    ymid = ((ymax+ymin)/2)
    centroid = (xmid,ymid)

    return xmid,ymid,centroid

def get_distance(x1,x2,y1,y2):

    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)    
    return distance

def draw_detection_box(frame,x1,y1,x2,y2,color):

    cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)

def densityMap(pcount):
    
    outfile = open('my_csv.csv','w')
    out = csv.writer(outfile)
    out.writerow(['People Count'])
    for i in range(count):
        #with open(('my_csv.csv','a')) as out:
        out.writerow([pcount])
    #outfile.close()
    
            
# Draw the predicted bounding box
def MarkPeople(frame, objects, total,outs):
    
    count = 0
    i=0
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
            
        # otherwise, there is a trackable object so we can utilize it
        else:
            to.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not to.counted:
                total+=1
                to.counted = True
           
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to


        count+=1
     
    return count, total
    


# Remove the bounding boxes with low confidence using non-maxima suppression
def Fill_tracker_list(rgb, frame, outs, count):
    
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    trackers = []
    
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    centroids = []
    stat_H, stat_L = 0, 0
    box_colors = []
    boxes = []
    detectedBox = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            assert(classId < len(classes))
            confidence = scores[classId]
            # Check if confidence is more than threshold and the detected object is a person
            if(confidence > confThreshold and classes and classes[classId]=="person"):
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                # Rectangle coordinates
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                #cv.putText(frame, classIds, (center_x[0] - 10, center_y[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in range(len(boxes)):
        if i in indices:
            left ,top, width, height = boxes[i]
            tracker = dlib.correlation_tracker()

            xmin = left
            ymin = top
            xmax = (left + width)
            ymax = (top + height)
            
            #calculate centroid point for bounding boxes
            xmid, ymid, centroid = calculateCentroid(xmin,ymin,xmax,ymax)
            detectedBox.append([xmin,ymin,xmax,ymax,centroid])

            my_color = 0
            for k in range (len(centroids)):
                c = centroids[k]
                
                
                if get_distance(c[0],centroid[0],c[1],centroid[1]) <= Distance:
                    box_colors[k] = 1
                    my_color = 1
                    cv2.line(frame, (int(c[0]),int(c[1])), (int(centroid[0]),int(centroid[1])), (255,0,0), 1,cv2.LINE_AA)
                    cv2.circle(frame, (int(c[0]),int(c[1])), 3, (255,165,0), -1,cv2.LINE_AA)
                    cv2.circle(frame, (int(centroid[0]),int(centroid[1])), 3, (255,165,0), -1,cv2.LINE_AA)
                    break
            centroids.append(centroid)
            box_colors.append(my_color)
            
            rect = dlib.rectangle(left, top, left+width, top+height)
            tracker.start_track(rgb, rect)
            color = (255, 255, 0)
            cv.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            label = str(classes[classId])
            cv.putText(frame, label, (left, top - 5), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))
            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(tracker)
    

    for i in range (len(detectedBox)):
        x1 = detectedBox[i][0]
        y1 = detectedBox[i][1]
        x2 = detectedBox[i][2]
        y2 = detectedBox[i][3]
        
        #for ellipse output
        xc = ((x2+x1)/2)
        yc = y2-5
        centroide = (int(xc),int(yc))

        if box_colors[i] == 0:
            color = WHITE
            draw_detection_box(frame,x1,y1,x2,y2,color)
            label = "safe"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            y1label = max(y1, labelSize[1])
            cv2.rectangle(frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), WHITE, cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,cv2.LINE_AA)
            stat_L += 1

        else:
            color = RED
            draw_detection_box(frame,x1,y1,x2,y2,color)
            # cv2.ellipse(frame, centroide, (35, 19), 0.0, 0.0, 360.0, RED, 2)
            label = "unsafe"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            y1label = max(y1, labelSize[1])
            cv2.rectangle(frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), WHITE, cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1,cv2.LINE_AA)
            stat_H += 1

        cv2.rectangle(frame, (13, 10),(250, 60), GREY, cv2.FILLED)
        LINE = "--"
        INDICATION_H = f'HIGH RISK: {str(stat_H)} PERSON'
        cv2.putText(frame, LINE, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1,cv2.LINE_AA)
        cv2.putText(frame, INDICATION_H, (60,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1,cv2.LINE_AA)

        INDICATION_L = f'LOW RISK : {str(stat_L)} PERSON'
        cv2.putText(frame, LINE, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,cv2.LINE_AA)
        cv2.putText(frame, INDICATION_L, (60,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1,cv2.LINE_AA)    
    
    

    return trackers,frame
    


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=60, maxDistance=50)
trackers = []
trackableObjects = {}
#total = 0

fig = plt.figure()
X = []
Y = []
# start the frames per second throughput estimator
fps = FPS().start()
#totalFrames = 0
        

def run(cap,args):

    begin = time.time()

    total = 0
    totalFrames = 0
    while True:
        
        # get frame from the video
        hasFrame, frame = cap.read()
        
        #converting frame form BGR to RGB for dlib 
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        status = "Waiting"
        rects = []
        distances = []
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args.skip_frames == 0:
            status = "Detecting"
            
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))
            
            
            # Remove the bounding boxes with low confidence and store in trackers list for future tracking
            trackers,r_frame  = Fill_tracker_list(rgb, frame, outs, 0)

        
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                bboxHeight = round(endY-startY,4)
                
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

                
        objects = ct.update(rects)
        
        # Mark  all the persons in the frame
        count,total = MarkPeople(r_frame, objects, total,outs)
        peopleCount.append(count)
        #print(count)
        #densityMap(peopleCount)    

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
        
            #("Passenger Count", count),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        H = None
        W = None
        (H, W) = frame.shape[:2]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        

        
        frm=cv.resize(r_frame, (1200,1000))
        
        #time.sleep(1)
        #end = time.time()
       #print("net time: ", end-begin)

        ############################################

        #p = multiprocessing.Pool()
        # Baggage Detection
        #result = detection.baggage(frm)
        
        #frm1=cv.resize(result, (1200,1000))
        ############################################
        #cv.imshow("LAS", imag1)
        # Safety Line Crossing Detection
        #result2 = saf.SafetyLineCross(result)

        #cv2.imshow("Edges", edges)
        cv2.imshow("Result", frm)
        cv2.VideoWriter(r'Crowd3.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,440)).write(frame)         # the frame is saved for the final video

        

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
        #fps = result.get(cv2.CAP_PROP_FPS)

        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
    # stop the timer and display FPS information
    fps.stop()

    #print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("approx. FPS: {:.2f}".format(fps.fps()))
