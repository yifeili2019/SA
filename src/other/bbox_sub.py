#!/usr/bin/env python
'''444 ROS Node'''
import rospy
from std_msgs.msg import String
from gdp.msg import position
from gdp.msg import multipositions
from cv_bridge import CvBridge,CvBridgeError
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
import math
import numpy as np

camera_fx = 383.599
camera_fy = 383.599
camera_cx = 320.583
camera_cy = 238.327

def getDistanceAngle(pixel_x, pixel_y, real_z):
    z = np.float(real_z)
    x = (pixel_x-camera_cx)*z/camera_fx
    y = (pixel_y-camera_cy)*z/camera_fy

    horizon_angle = math.atan2(x,z)
    vertical_angle = math.atan2(y,z)
    absolute_distance = math.sqrt(x*x+y*y+z*z)
    print("x: ",x)
    print("y: ",y)
    print("z: ",z)
    print("absolut_distance: ",absolute_distance)
    print("horizon_angle: ",horizon_angle)
    print("vertical_angle: ",vertical_angle)
    #pass

def callback(data):
    '''444 Callback Function'''
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I hear")
    #print(data.lists)
    print("drone_id:",data.drone_id)
    for item in data.lists:
        mid_u = (item.right-item.left)/2+item.left
        mid_v = (item.bottom-item.top)/2+item.top
        getDistanceAngle(mid_u,mid_v,item.distance)
    


def listener():
    '''444 Subscriber'''
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('bbox_sub', anonymous=False)

    rospy.Subscriber("bbox_position2", multipositions, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
