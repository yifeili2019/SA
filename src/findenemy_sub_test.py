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
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from gdp.msg import eposfixedframeM
from gdp.msg import eposesM,Action,Agent
from gdp.msg import poses, enemyAndFriends
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

def calDistance(item1,item2):
    distance = 0
    '''
    x_dis = (item1.x - item2.x)*(item1.x - item2.x)
    y_dis = (item1.y - item2.y)*(item1.y - item2.y)
    z_dis = (item1.z - item2.z)*(item1.z - item2.z)
    '''
    x_dis = (item1[0] - item2.data[0])*(item1[0] - item2.data[0])
    y_dis = (item1[1] - item2.data[1])*(item1[1] - item2.data[1])
    z_dis = (item1[2] - item2.data[2])*(item1[2] - item2.data[2])
    
    distance = math.sqrt(x_dis+y_dis+z_dis)
    return distance

def calDistanceList(item,list1):
    global distanceError
    global list_enemy
    dis_min = 100
    for i in range(len(list1)):
        dis_temp = calDistance(item,list1[i])
        if(dis_temp<dis_min):
            dis_min = dis_temp
    
    if(dis_min > distanceError):
        list_enemy.append(item)

def compareListM(list_all,list_friends):
    for i in range(len(list_all)):
        calDistanceList(list_all[i],list_friends)


'''
def compareList(list_all,list_friends):
    global distanceError
    list_enemy = []

    for i in range(len(list_all)):
        for j in range(len(list_friends)):
            dis_temp = calDistance(list_all[i],list_friends[j])
            if dis_temp > distanceError:
                list_enemy.append(list_all[i])
    
    return list_enemy
'''


#2020/02/15
def data2list(data):
    result = []
    list1 = data.lists
    for i in range(len(list1)):
        temp = []
        temp.append(list1[i].x)
        temp.append(list1[i].y)
        temp.append(list1[i].z)
        temp.append(list1[i].p)
        temp.append(list1[i].uncertainty)
        temp.append(list1[i].dx)
        temp.append(list1[i].dy)
        temp.append(list1[i].dz)
        result.append(temp)
    return result


def callback(data1, data2):
    list_enemy = []
    global pub
    global list_enemy
    global k
    global create_file

    ## import your variables from subscribers
    # data1, data2 = data1.data, data2.data
    #offset[data2.drone_id]

    try:
        create_file.write(str(k))
        create_file.write("\n")
        ## ----------------------- LOOP --------------------- [main loop]
        ## Your data processing here


        #var2publish1 = rospy.get_caller_id() + 'I heard %s', data1.data
        #var2publish2 = rospy.get_caller_id() + 'I heard %s', data2.data
        
        var2publish1 = data1.lists
        var2publish2 = data2.lists
        #var2publish3 = data3.lists
        #var2publish4 = data4.lists
        #var2publish5 = data5.lists
        #var2publish6 = data6.lists

        ## Publish your data here
        #rospy.loginfo(var2publish1)  # [optional] print & save data in the node's terminal
        print("speaker1")
        print(var2publish1)
        print("speaker2")
        print(var2publish2)
        '''
        print("speaker3")
        print(var2publish3)
        print("speaker4")
        print(var2publish4)
        print("speaker5")
        print(var2publish5)
        print("speaker6")
        print(var2publish6)
        '''

        list1 = data2list(data1)
        list2 = data2list(data2)
        #list3 = data2list(data3)
        #list4 = data2list(data4)
        #list5 = data2list(data5)
        #list6 = data2list(data6)

        #list_all = list1+list2+list3+list4+list5+list6
        list_all = list1+list2

        list_friends = []
        list_friends.append(data1.pos)
        list_friends.append(data2.pos)
        #list_friends.append(data3.pos)
        #list_friends.append(data4.pos)
        #list_friends.append(data5.pos)
        #list_friends.append(data6.pos)

        compareListM(list_all,list_friends)  #[[x,y,z,p]]
        

        output = enemyAndFriends()
        create_file.write("enemylist:")
        create_file.write("\n")

        
        for i in range(len(list_enemy)):
            pos_temp = poses()
            pos_temp.pos.data.append(list_enemy[i][0])
            pos_temp.pos.data.append(list_enemy[i][1])
            pos_temp.pos.data.append(list_enemy[i][2])
            pos_temp.p = list_enemy[i][3]
            temp = [list_enemy[i][0],list_enemy[i][1],list_enemy[i][2]]
            create_file.write(str(temp))
            output.enemylists.append(pos_temp)


        create_file.write("\n")
        create_file.write("friendslist:")
        create_file.write("\n")
        for i in range(len(list_friends)):
            pos_temp = poses()
            pos_temp.pos = list_friends[i]
            pos_temp.p = 100
            create_file.write(str(pos_temp))
            output.friendlists.append(pos_temp)
        create_file.write("\n")
        k = k+1
        pub.publish(output)

        ##################################################################
        rate.sleep() # to reach rospy.Rate(...) frequency.
        pass
    except rospy.ROSInterruptException:
        pass
    
def callback2(data):
    ## add a callback if you don't want data to be imported synchronously into this node
    pass

if __name__ == "__main__":

    k=0
    create_file = open("/home/dell/test1/src/gdp/src/findenemy.txt","w")
    distanceError = 1
    #list_enemy = []
    
    
    rospy.init_node('findenemy_sub_test', anonymous=False)

    # do nothing
    quickFix_header = Header()
    quickFix_header.stamp = rospy.Time.now()
    String.header = quickFix_header # adds a header in String to synchronize subscribing
    ## Setup publishers (name, dataType)
    pub = rospy.Publisher('findenemy_sub_test', enemyAndFriends, queue_size=10)

    rate = rospy.Rate(10) # setup publishing frequency, in Hz

    ## Setup subscribers (name, dataType)
    #ts = message_filters.TimeSynchronizer([message_filters.Subscriber('findenemy_pub', eposes),
    #                                       message_filters.Subscriber('findenemy_pub1',eposes)], 10)

    ## callback for treatment of data from subscribers above
    #ts.registerCallback(callback)

    ## copy -Setup subscribers lines- here to define other subscribers for which data needs to be processed asynchronously with the data treatment above
    # Ex:
    ts = message_filters.TimeSynchronizer([message_filters.Subscriber('kalman', eposesM),
                                            message_filters.Subscriber('findenemy_pub2', eposesM),], 10)
    ts.registerCallback(callback)


    # Prevent undesired program ending
    rospy.spin()
    pass