#!/usr/bin/env python
'''yolo ROS Node'''
# license removed for brevity
from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
from core import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Float32

from test.msg import position
from test.msg import multipositions

import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from yolo_class import YOLO
import pyrealsense2 as rs
from random import randint
import math

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
camera_fx = 383.599
camera_fy = 383.599
camera_cx = 320.583
camera_cy = 238.327

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

def talker():
    '''yolo Publisher'''
    trackerType = "CSRT"
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    pub = rospy.Publisher('bbox_position1', multipositions, queue_size=10)
    #pub1 = rospy.Publisher('bbox_position', multipositions, queue_size=10)
    rospy.init_node('bbox_pub1', anonymous=False)
    #rate = rospy.Rate(10) # 10hz


    yolo = YOLO()

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    #bboxes = []

    # we create the video capture object cap
    #cap = cv2.VideoCapture(0)
    #if not cap.isOpened():
        #raise IOError("We cannot open webcam")
    
    j = 0

    while not rospy.is_shutdown():

        if(j%10 == 0):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            image_frame = np.asanyarray(color_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())

            color_frame = cv2.resize(image_frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
            depth_frame = cv2.resize(depth_frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

            position_list = multipositions()
            r_image, ObjectsList, position_list = yolo.detect_img(color_frame,depth_frame,depth_scale)


            bridge = CvBridge()
            image = bridge.cv2_to_imgmsg(color_frame,"bgr8")
            position_list.picture = image

            #print(position_list.lists)
            #bbox = (1,2,3,4)
            #print(bbox)
    
            bboxes = []
            colors = []
            multiTracker = cv2.MultiTracker_create()
            print("hello")
            for item in position_list.lists:
                bbox = (item.left,item.top,item.right,item.bottom)
                bboxes.append(bbox)
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                print(bbox)
            print(bboxes)
            print("bboxes done")

            #multiTracker = cv2.MultiTracker_create()
            #print("hello")
            img = cv2.imread("1.jpg")
            for bbox in bboxes:
              print(bbox)
              multiTracker.add(createTrackerByName(trackerType), color_frame, bbox)

            print("multitracker done")
            #pub.publish(position_list)
        j+=1

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        #ret, frame = cap.read()
        # resize our captured frame if we need
        frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

        bridge = CvBridge()
        image = bridge.cv2_to_imgmsg(frame,"bgr8")

        #position_list = multipositions()
        # detect object on our frame
        #r_image, ObjectsList, position_list = yolo.detect_img(frame)
        #position_list.picture = image

        success, boxes = multiTracker.update(frame)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        

        #cv2.imshow('MultiTracker', frame)

        #show us frame with detection
        cv2.imshow("tracking", frame)
        cv2.imshow("Web cam input", r_image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # calculate FPS
        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
            print("FPS:", fps / TIME)
            fps = 0 
            start_time = time.time()
        
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        #pub.publish(hello_str)
        pub.publish(position_list)


    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()

    #pub = rospy.Publisher('chatter', String, queue_size=10)
    #rospy.init_node('yolo', anonymous=False)
    #rate = rospy.Rate(10) # 10hz
    '''
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()
    '''
def getDistanceAngle(pixel_x, pixel_y, real_z):
    z = np.float(real_z)
    x = (pixel_x-camera_cx)*z/camera_fx
    y = (pixel_y-camera_cy)*z/camera_fy

    horizon_angle = math.atan2(x,z)
    vertical_angle = math.atan2(y,z)
    absolute_distance = math.sqrt(x*x+y*y+z*z)
    print(absolute_distance)
    print(horizon_angle)
    print(vertical_angle)
    #pass


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
