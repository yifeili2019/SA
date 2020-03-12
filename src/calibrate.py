import pyrealsense2 as rs
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
import numpy as np
pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
profile = pipeline.start(config)
i = 0
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    image_frame = np.asanyarray(color_frame.get_data())
    depth_frame = np.asanyarray(depth_frame.get_data())

    color_frame = cv2.resize(image_frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    depth_frame = cv2.resize(depth_frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

    cv2.imshow("now",color_frame)
    cv2.imshow("depth",depth_frame)
    keyCode = cv2.waitKey(30) & 0xFF
    if keyCode == 27:
        break
    if keyCode == 113:
        cv2.imwrite("hello_"+str(i)+".jpg",image_frame)

    i = i+1


