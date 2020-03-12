#!/usr/bin/env python
from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
from test.msg import testimage
#from cv_bridge.boost.cv_bridge_boost import getCvType
from core import CvBridge, CvBridgeError
#import roslib
#roslib.load_manifest('my_package')
#from test.msg import Message
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)
import cv2
from std_msgs.msg import String
import numpy as np
import pyrealsense2 as rs


def pubImage():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    rospy.init_node('pubImage',anonymous = True)
    pub = rospy.Publisher('ShowImage', Image, queue_size = 10)
    rate = rospy.Rate(10)
    bridge = CvBridge()
    gt_imdb = []
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #image = cv2.imread("1.jpg")
        #image = cv2.resize(image,(900,450))
        pub.publish(bridge.cv2_to_imgmsg(frame,"bgr8"))
        #cv2.imshow("lala",image)
        #cv2.waitKey(0)
        rate.sleep()

if __name__ == '__main__':
    try:
        pubImage()
    except rospy.ROSInterruptException:
        pass
