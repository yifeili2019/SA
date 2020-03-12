#!/usr/bin/env python
'''444 ROS Node'''
import rospy
from std_msgs.msg import String
from test.msg import position
from test.msg import multipositions
from cv_bridge import CvBridge,CvBridgeError
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
from random import randint


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

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

def callback(data):
    '''444 Callback Function'''
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data.picture,"bgr8")

    #cv2.imshow("Web cam output", cv_image)
    #if cv2.waitKey(25) & 0xFF == ord("q"):
        #cv2.destroyAllWindows()
    
    trackerType = "CSRT"
    bboxes = []
    colors = []

    for item in data.lists:
        print(item)
        bbox = (item.left,item.top,item.right,item.bottom)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

    multiTracker = cv2.MultiTracker_create()

    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), cv_image, bbox)

    print(type(cv_image))
    #print()
    #cv2.imshow("lala",cv_image)
    #print(data.lists)
    #print("I hear data!")
    


def listener():
    '''444 Subscriber'''
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('bbox_sub1', anonymous=False)

    rospy.Subscriber("bbox_position1", multipositions, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
