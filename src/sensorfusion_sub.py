#!/usr/bin/env python
'''sensorfusion ROS Node'''
import rospy
from std_msgs.msg import String
from sf.msg import poses, enemyAndFriends
import os
import time

import math
import numpy as np
from numpy import linalg as npla
from numpy.linalg import inv
# from scipy import linalg
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from std_msgs.msg import *
import message_filters

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
#2020/02/16
def calDistance(item1,item2):
    distance = 0
    '''
    x_dis = (item1.x - item2.x)*(item1.x - item2.x)
    y_dis = (item1.y - item2.y)*(item1.y - item2.y)
    z_dis = (item1.z - item2.z)*(item1.z - item2.z)
    '''
    x_dis = (item1[0] - item2[0])*(item1[0] - item2[0])
    y_dis = (item1[1] - item2[1])*(item1[1] - item2[1])
    z_dis = (item1[2] - item2[2])*(item1[2] - item2[2])
    
    distance = math.sqrt(x_dis+y_dis+z_dis)
    return distance

#2020/02/16
def matchEnemy(enemy_list,flag_list,threshold):
    result = []
    for i in range(len(enemy_list)):
        if(flag_list[i] == 0):
            temp = []
            temp.append(enemy_list[i])
            flag_list[i] = 1
            for j in range(i,len(enemy_list)):
                if(flag_list[j] == 0):
                    distance = calDistance(enemy_list[i],enemy_list[j])
                    if distance < threshold:
                        temp.append(enemy_list[j])
                        flag_list[j] = 1

            result.append(temp)

    return result 

###???check euqation
#2020/02/17
def fusionItem(item1,item2):
    p1 = np.identity(3)*item1[3]
    p2 = np.identity(3)*item2[3]
    x1 = np.array([[item1[0]],[item1[1]],[item1[2]]])
    x2 = np.array([[item2[0]],[item2[1]],[item2[2]]])

    p_sf = p1-p1*(inv(p1+p2))p1.T
    x_sf = x1+p1*(inv(p1+p2))(x2-x1)

    p = p_sf[0][0]
    result = [x_sf[0][0],x_sf[1][0],x_sf[2][0],p]


    return result

#2020/0217
def fusionList(list1):
    if(len(list1) == 1):
        temp = list1[0]
        return temp
    if(len(list1) == 2):
        temp = fusionItem(list1[0],list1[1])
        return temp
    if(len(list1)>=3):
        temp = fusionItem(list1[0],list1[1])
        for i in range(2,len(list1)):
            temp = fusionItem(temp,list1[i])
        
        return temp
    


#2020/02/16
def fusion(matched_enemy_list):
    result = []
    for i in range(len(matched_enemy_list)):
        temp = fusionList(matched_enemy_list[i])
        result.append(temp)

    return result

def change(flag_list):
    flag_list[0] = 1
    
#2020/02/16
def callback(data1):
    global pub
    global threshold

    ## import your variables from subscribers
    # data1, data2 = data1.data, data2.data

    try:
        #print("I hear")
        ## ----------------------- LOOP --------------------- [main loop]
        ## Your data processing here

        
        #var2publish1 = data1
        #print(var2publish1.enemylists[0])
        #rint(data1.enemylists)
        #print(len(data1.enemylists))

        
        data = data1.enemylists
        enemy_list = []
        flag_list = []
        for i in range(len(data)):
            temp = []
            temp.append(data[i].pos.data[0])
            temp.append(data[i].pos.data[1])
            temp.append(data[i].pos.data[2])
            temp.append(data[i].p)
            enemy_list.append(temp)
            flag_list.append(0)

        matched_enemy_list = matchEnemy(enemy_list,flag_list,threshold)
        result = fusion(matched_enemy_list)

        pub_data = enemyAndFriends()
        for i in range(len(result)):
            temp = poses()
            temp.pos.data.append(result[i][0])
            temp.pos.data.append(result[i][1])
            temp.pos.data.append(result[i][2])
            temp.p = result[i][3]
            pub_data.enemylists.append(temp)

        pub.publish(pub_data)





        #pub.publish(var2publish1)


        ##################################################################
        rate.sleep() # to reach rospy.Rate(...) frequency.
        pass
    except rospy.ROSInterruptException:
        pass
    
def callback2(data):
    ## add a callback if you don't want data to be imported synchronously into this node
    pass

if __name__ == "__main__":

    threshold = 10
    i = 0
    ## ----------------------- SETUP --------------------- [Code to run only once]
    ## Setup node (name)
    rospy.init_node('sensorfusion_sub', anonymous=True)

    # do nothing
    quickFix_header = Header()
    quickFix_header.stamp = rospy.Time.now()
    String.header = quickFix_header # adds a header in String to synchronize subscribing


    ## Setup publishers (name, dataType)
    pub = rospy.Publisher('pub_name', enemyAndFriends, queue_size=10)

    rate = rospy.Rate(10) # setup publishing frequency, in Hz

    ## Setup subscribers (name, dataType)
    ts = message_filters.TimeSynchronizer([message_filters.Subscriber('sensorfusion_pub', enemyAndFriends)], 10)

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
