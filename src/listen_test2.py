#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''-------------------------- Program Header ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------''
  ' Add all important details regarding the program here
  '
  ' Project             : BAE Swarm Challenge - Group Design Project
  ' Program name        : SAMPLE_CODE_gdpa.py
  ' Author              : Romain Delabeye
  ' Adm. structure      : Cranfield University
  ' Date (YYYYMMDD)     : 20200205
  ' Purpose             : provide code structure
  ' User                : A_Team
  '
  ' Revision History    : 1.0
  '
  ' Date YYYYMMDD |  Author          | Ref       |Revision comment
  '-------------------------------------------------------------------
  ' 20200205      |  Romain DELABEYE |           | Developing & Structuring - howto publish&subscribe; struct
  '               |                  |           | 
  '               |                  |           | 
  '               |                  |           | 


TODO:
""" Put here all your tasks """
(You can also add TODO anywhere in the code if you need to come back to a certain line later on)

-
-
-

''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''
'------------------------------------------------------------------------------------------------------------------------
'    import libraries
'------------------------------------------------------------------------------------------------------------------------
'''

## import libraries - comment the undesired libraries
import os
import time

import math
import numpy as np
from numpy import linalg as npla
# from scipy import linalg
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import rospy
from std_msgs.msg import *
import message_filters
from sensor_msgs.msg import *
from gdp.msg import position_detection
from gdp.msg import multipositions
import math
#from autolab_core import RigidTransform
import numpy as np
#from scipy.spatial.transform import Rotation as Rt
from geometry_msgs.msg import PoseStamped
from gdp.msg import eposfixedframe
from gdp.msg import eposes,Action,Agent
## import your external programs
# import config as cf # Please put all your global constants in a separate file, config.py ()

'''
'------------------------------------------------------------------------------------------------------------------------
'    Toolbox
'------------------------------------------------------------------------------------------------------------------------
'''
'''
Put here your toolbox (functions)
'''





'''
'------------------------------------------------------------------------------------------------------------------------
'    Classes
'------------------------------------------------------------------------------------------------------------------------
'''
'''
Put here your classes.
'''




'''
'------------------------------------------------------------------------------------------------------------------------
'    Main
'------------------------------------------------------------------------------------------------------------------------
'''
'''
camera_cx = 320.583
camera_cy = 238.327
camera_fx = 383.599
camera_fy = 383.599
'''

camera_cx = 305.325
camera_cy = 245.265
camera_fx = 583.074
camera_fy = 583.391

depth_scale = 1

#position of arena(0,0,0)
arena_lat = 0
arena_lon = 0
arena_alt = 0
R = 6371000 #meter

offset = {"1":[10,-3,0],"2":[10,-6,0],"3":[13,-3,0],"4":[13,-6,0],"5":[16,-3,0],"6":[16,-6,0]}  #offset of take off position fixed frame
                                                                                                #small:1,2  medium:3,4   big:5,6

def calEnemyPose(pixel_x,pixel_y,real_z):
    print('pixel_x:',pixel_x)
    print('pixel_y:',pixel_y)
    print('real_z:',real_z)
    z = np.float(real_z)
    x = (pixel_x-camera_cx)*z/camera_fx+0.4
    y = (pixel_y-camera_cy)*z/camera_fy-0.4
    #x = (pixel_x-320.583)*z/383.599
    #y = (pixel_y-245.265)*z/383.599
    return z,y,x

def pixelPose2cameraPose(multiPositions):
    enemyPositions = []
    for item in multiPositions.lists:
        mid_u = (item.bottom-item.top)/2+item.top
        mid_v = (item.right-item.left)/2+item.left
        x,y,z = calEnemyPose(mid_u,mid_v,item.distance)
        temp = [x,y,z]
        enemyPositions.append(temp)
    return enemyPositions


def imu2RotationMatrix(pose_quat):
    orientation = {'x': pose_quat.pose.orientation.x, 'y':pose_quat.pose.orientation.y , 'z':pose_quat.pose.orientation.z , 'w':pose_quat.pose.orientation.w }
    position = {'x': pose_quat.pose.position.x, 'y': pose_quat.pose.position.y, 'z': pose_quat.pose.position.z}

    rotation_quaternion = np.asarray([orientation['w'], orientation['x'], orientation['y'], orientation['z']])
    translation = np.asarray([position['x'], position['y'], position['z']])
    
    T_qua2rota = RigidTransform(rotation_quaternion,translation)

    return T_qua2rota

def eulerAngleToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta.data[0]), -math.sin(theta.data[0]) ],
                    [0,         math.sin(theta.data[0]), math.cos(theta.data[0])  ]
                    ])

    R_y = np.array([[math.cos(theta.data[1]),    0,      math.sin(theta.data[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta.data[1]),   0,      math.cos(theta.data[1])  ]
                    ])

    R_z = np.array([[math.cos(theta.data[2]),    -math.sin(theta.data[2]),    0],
                    [math.sin(theta.data[2]),    math.cos(theta.data[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def computePixelError(multiPositions,R):
    errors = []
    for item in multiPositions.lists:
        mid_u = (item.bottom-item.top)/2+item.top
        mid_v = (item.right-item.left)/2+item.left
        real_z = item.distance
        #print("u:",mid_u)
        #print("v:",mid_v)
        #print("z:",real_z)
        temp_error = computeError(R,mid_u,mid_v,real_z)
        errors.append(temp_error)
        #print("temp_error:",temp_error)
    return errors

def computeError(R,pixel_x,pixel_y,real_z):
    #print("real_z:",real_z)
    uncertainty = 0.0046*math.pow(real_z,4)-0.0763*math.pow(real_z,3)+0.4436*math.pow(real_z,2)-1.0451*real_z+0.8528
    uncertainty = uncertainty*3
    #print("uncertainty:",uncertainty)
    delt_z = np.float(uncertainty)
    delt_x = (pixel_x-camera_cx)*delt_z/camera_fx
    delt_y = (pixel_y-camera_cy)*delt_z/camera_fy

    delt_z = abs(delt_z)
    delt_x = abs(delt_x)
    delt_y = abs(delt_y)
    #print("delt_x:",delt_x)
    #print("delt_y",delt_y)

    camPoseError = [delt_z,delt_y,delt_x]

    temp_T = 2.5/math.sqrt(3)
    T = [temp_T,temp_T,temp_T]

    error = camError(camPoseError,R,T)
    return error







'''
def camPose2fixedPose(camPose,T_qua2rota,drone_id):
    camPoseMatrix = np.array([[camPose[0]],[camPose[1]],[camPose[2]]])
    rotationMatrix = T_qua2rota.rotation

    translation = np.array([[T_qua2rota.translation[0]],[T_qua2rota.translation[1]],[T_qua2rota.translation[2]]])
    output = np.dot(rotationMatrix,camPoseMatrix) + translation

    drone_offset = np.array([[offset[drone_id][0]],[offset[drone_id][1]],[offset[drone_id][2]]])
    output = output.T + drone_offset

    return output
'''
def camError(camPose,R,T):
    offset = {1:[10,-3,0],2:[10,-6,0],3:[13,-3,0],4:[13,-6,0],5:[16,-3,0],6:[16,-6,0]}
    #print("campos:",camPose)

    camPoseMatrix = np.array([[camPose[0]],[camPose[1]],[camPose[2]]])
    #print("camPoseMatrix",camPoseMatrix)

    rotationMatrix = R
    #print("R:",R)

    translation = T
    #print("translation:",translation)

    output = np.dot(rotationMatrix,camPoseMatrix) + translation
    #print("output:",output)


    output = [output[0][0],output[1][0],output[2][0]]
    #print("output:",output)

    return output

def camPose2fixedPose(camPose,R,T,drone_id):
    offset = {1:[10,-3,0],2:[10,-6,0],3:[13,-3,0],4:[13,-6,0],5:[16,-3,0],6:[16,-6,0]}
    print("campos:",camPose)

    camPoseMatrix = np.array([[camPose[0]],[camPose[1]],[camPose[2]]])
    print("camPoseMatrix",camPoseMatrix)

    rotationMatrix = R
    print("R:",R)

    translation = T
    print("translation:",translation)

    output = np.dot(rotationMatrix,camPoseMatrix) + translation
    #print("output:",output)

    drone_offset = np.array([[offset[drone_id][0]],[offset[drone_id][1]],[offset[drone_id][2]]])
    #print("drone_offset",drone_offset)
    drone_offset = np.array([[0],[0],[0]])
    output = output + drone_offset
    

    output = np.array([[output[0][0]],[output[1][0]],[output[2][0]]])
    print("output:",output)

    return output

def callback(data1, data2):
    global pub
    global create_file
    global k



    try:
        #offset = {1:[10,-3,0],2:[10,-6,0],3:[13,-3,0],4:[13,-6,0],"5":[16,-3,0],"6":[16,-6,0]}
       
        print("I hear")
  
        
        enemyPositionsCameraFrame = pixelPose2cameraPose(data2)    #get enemy positions in camera frame
        
        R = eulerAngleToRotationMatrix(data1.attitude)
        temp = data1.pos
        T = np.array([[temp.data[0]],[temp.data[1]],[temp.data[2]]]) 

        errors = computePixelError(data2,R)
        #print("errors:",errors)


        
        enemyPositionsFixedFrame = []

        enemyPoitions = eposes()
        enemyPoitions.pos.data.append(data1.pos.data[0])
        enemyPoitions.pos.data.append(data1.pos.data[1])
        enemyPoitions.pos.data.append(data1.pos.data[2])

        for i in range(len(enemyPositionsCameraFrame)):
            item = enemyPositionsCameraFrame[i]
            a = eposfixedframe()
            temp = camPose2fixedPose(item,R,T,data2.drone_id)
            #temp_error = computeError(R,)
            a.x = temp[0][0]
            a.y = temp[1][0]
            a.z = temp[2][0]
            a.dx = errors[i][0]
            a.dy = errors[i][1]
            a.dz = errors[i][2]
            enemyPoitions.lists.append(a)
            enemyPositionsFixedFrame.append(temp)
        
        if len(enemyPositionsFixedFrame) >0:
            temp = str(enemyPositionsFixedFrame)
            create_file.write(str(k))
            create_file.write("\n")
            create_file.write(temp)
            create_file.write("\n")
        k = k+1

        '''
        for item in enemyPositionsCameraFrame:
            a = eposfixedframe()
            temp = camPose2fixedPose(item,R,T,data2.drone_id)
            #temp_error = computeError(R,)
            a.x = temp[0][0]
            a.y = temp[1][0]
            a.z = temp[2][0]
            enemyPoitions.lists.append(a)
            enemyPositionsFixedFrame.append(temp)
        '''
        

        print(enemyPoitions.lists)
        pub.publish(enemyPoitions)
        
        
        
        

        rate.sleep() # to reach rospy.Rate(...) frequency.
        pass
    except rospy.ROSInterruptException:
        pass
    
def callback2(data):
    ## add a callback if you don't want data to be imported synchronously into this node
    pass

if __name__ == "__main__":
    #drone_id = "1"
    #a = np.array([offset[drone_id][0],offset[drone_id][1],offset[drone_id][2]])
    #b = np.array([1,2,3])
    #print(np.matmul(a,b))
    #print(np.matmul(a.T,b))
    '''
    n1 = np.array([[1,2,3],[4,5,6]])
    n2 = np.array([[1,2],[3,4],[5,6]])
    n3 = np.array([[1],[2],[3]])
    n4 = np.array([4,5,6])
    print(n3.T+n4)


    orientation = {'x': 0, 'y':0 , 'z':0 , 'w':1 }
    position = {'x': 1, 'y': 2, 'z': 3}

    rotation_quaternion = np.asarray([orientation['w'], orientation['x'], orientation['y'], orientation['z']])
    translation = np.asarray([position['x'], position['y'], position['z']])
    
    T_qua2rota = RigidTransform(rotation_quaternion,translation)
    print(T_qua2rota.rotation)
    print(T_qua2rota.translation[0])
    '''
    k=0

    create_file = open("/home/dell/test1/src/gdp/src/listen_test2.txt",'w')
    
    rospy.init_node('listen_test2', anonymous=False)

    # do nothing
    quickFix_header = Header()
    quickFix_header.stamp = rospy.Time.now()
    String.header = quickFix_header # adds a header in String to synchronize subscribing


    ## Setup publishers (name, dataType)
    pub = rospy.Publisher('pub_enemy_position', eposes, queue_size=10)

    rate = rospy.Rate(10) # setup publishing frequency, in Hz

    ## Setup subscribers (name, dataType)
    ts = message_filters.TimeSynchronizer([message_filters.Subscriber('fake_pub', Agent),
                                           message_filters.Subscriber('bbox_position2',multipositions)], 10)

    ## callback for treatment of data from subscribers above
    ts.registerCallback(callback)

    ## copy -Setup subscribers lines- here to define other subscribers for which data needs to be processed asynchronously with the data treatment above
    # Ex:
    # ts = message_filters.TimeSynchronizer([message_filters.Subscriber('subs_name', String),
    #                                        message_filters.Subscriber('subs_name2', String)], 10)
    # ts.registerCallback(callback)


    # Prevent undesired program ending
    rospy.spin()
    pass