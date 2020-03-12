#!/usr/bin/env python
'''kalman ROS Node'''
import rospy
#from std_msgs.msg import String
from gdp.msg import eposes, eposfixedframe,Agent,Action
from gdp.msg import eposesM, eposfixedframeM
import math
import time
import numpy as np
from numpy.linalg import inv


#2020/02/13
#calculate velocity between item1 and item2
#both item are list
def calVelocity(item1,item2,delt_time):
    vel = []

    x_vel = (item2[0] - item1[0])/delt_time
    y_vel = (item2[1] - item1[1])/delt_time
    z_vel = (item2[2] - item1[2])/delt_time
    
    vel = [x_vel,y_vel,z_vel]

    return vel

#2020/02/13
def calVelocityList(list1,list2,delt_time):
    if(len(list1) != len(list2)):
        print("length list1 is not the same as length list2")
        pass
    velocity = []
    for i in range(len(list1)):
        vel_temp = calVelocity(list1[i],list2[i],delt_time)
        velocity.append(vel_temp)

    return velocity

#2020/02/12
#calculate Eucildean distance of two target
#both items are list
def calDistance(item1,item2):
    distance = 0

    x_dis = (item1[0] - item2[0])*(item1[0] - item2[0])
    y_dis = (item1[1] - item2[1])*(item1[1] - item2[1])
    z_dis = (item1[2] - item2[2])*(item1[2] - item2[2])
    
    distance = math.sqrt(x_dis+y_dis+z_dis)
    print("dis:",distance)
    return distance

#2020/02/12
#compare Eucildean distance with threshold
def compareWithEpsilon(distance_lists,threshold):
    min_dis = 200
    count = 0
    for i in range(len(distance_lists)):
        if(distance_lists[i] < min_dis):
            min_dis = distance_lists[i]           #find the shortest distance
            count = i                             #record index of this distance
    if min_dis < threshold:
        return count                              #return this index

    return -1

#2020/02/12
#find corresponding match between list1 and list2 
def matchCase1(list1,list2,threshold):
    index = []                          
    for item2 in list2:
        dis_temp = []
        for item1 in list1:
            dis = calDistance(item2,item1)
            dis_temp.append(dis)
        index_temp = compareWithEpsilon(dis_temp,threshold)
        index.append(index_temp)                                   #get index of corresponding item of list1
    
    newlist1 = []
    newlist2 = []
    for i in range(len(index)):
        if(index[i] > -1):
            newlist1.append(list1[index[i]])                       #find corresponding item of list1
            newlist2.append(list2[i])                              #find corresponding item of list2
    
    return newlist1,newlist2

#2020/02/13
def matchCase1_M1(list1,list2,threshold):
    index = []
    for item2 in list2:
        dis_temp = []
        for item1 in list1:
            dis = calDistance(item2,item1)
            dis_temp.append(dis)
        index_temp = compareWithEpsilon(dis_temp,threshold)
        index.append(index_temp)
    
    newlist1 = []
    newlist2 = []
    list2_left = []
    for i in range(len(index)):
        if(index[i] > -1):
            newlist1.append(list1[index[i]])
            newlist2.append(list2[i])
        if(index[i] == -1):
            list2_left.append(list2[i])                            #store list2 item who does't match with list1
    
    return newlist1,newlist2,list2_left



#2020/02/12
def matchList(list1,list2,delt_time,threshold_vel):
    global threshhold
    threshold = threshold_vel*delt_time    # delt epsilon [m]   

    newlist1, newlist2 = matchCase1(list1,list2,threshold)
    
    
    return newlist1,newlist2

#2020/02/13
def matchList_M1(list1,list2,threshold):
    newlist1, newlist2, list2_left = matchCase1_M1(list1,list2,threshold)
    
    return newlist1,newlist2,list2_left
    

#2020/02/12
def storeData(data):
    posdata = []
    for item  in data.lists:
        temp = [item.x,item.y,item.z]
        posdata.append(temp)
    print(posdata)
    return posdata

#2020/02/13
#give out predicted position of next frame
def predictPositionItem(item,velocity,delt_time):
    predictPose = []
 
    for i in range(0,3):
        prePose_temp = item[i] + velocity[i]*delt_time              #calculate prredicted positions
        predictPose.append(prePose_temp)
    
    
    return predictPose

#2020/02/13
def predictPositionList(list1,velocity_list,delt_time):
    predictPoses = []
    for i in range(len(list1)):
        pre_temp = predictPositionItem(list1[i],velocity_list[i],delt_time)
        predictPoses.append(pre_temp)

    return predictPoses



#2020/02/13
#using kalman filter to give a better estimation of position
def fn_EKF_item(item_origin,velocity_origin,measurement,delt_time):
    global F, H, Q, R, P_init
    X_origin = np.array([[item_origin[0]],[item_origin[1]],[item_origin[2]]])  #3x1
    V_origin = np.array([[velocity_origin[0]],[velocity_origin[1]],[velocity_origin[2]]])  #3x1  velocity
    measure_next = np.array([[measurement[0]],[measurement[1]],[measurement[2]]])

    X_s = F*X_origin + delt_time*V_origin
    P_s = F*P_init*F.T + Q
    y_k = measure_next - X_s

    S = H*P_s*H.T + R
    k = P_s*H.T*(inv(S))
    X = X_s + k*y_k
    P = (np.identity(3)-k*H)*P_s

    P_init = P
    return X

#2020/02/14
#using kalman filter to give a better estimation of position
def fn_EKF_item_M1(item_predict,measurement):
    global F, H, Q, R, P_init
    X_s = np.array([[item_predict[0]],[item_predict[1]],[item_predict[2]]])  #3x1
    measure_next = np.array([[measurement[0]],[measurement[1]],[measurement[2]]])

    #X_s = F*X_origin + delt_time*V_origin
    P_s = F*P_init*F.T + Q
    print("P_S:",P_s)
    y_k = measure_next - X_s
    print("y_k:",y_k)

    S = H*P_s*H.T + R
    print("s:",S)
    k = P_s*H.T*(inv(S))
    print("K:",k)
    X = X_s + np.dot(k,y_k)
    print("x:",X)
    P = (np.identity(3)-k*H)*P_s
    print("p_init:",P_init)

    P_init = P
    return X,measurement[3]

#2020/02/14
def fn_EKF_list_M1(predict_list,measurement_list):
    result = []
    uncertainty = []
    for i in range(len(predict_list)):
        temp_X,temp_uncertainty = fn_EKF_item_M1(predict_list[i],measurement_list[i])
        result.append(temp_X)
        uncertainty.append(temp_uncertainty)

    return result,uncertainty
    


#2020/02/13
def callback(data):
    '''kalman Callback Function'''
    #rospy.loginfo(data.lists)
    
    print("I hear")
    global threshold_vel
    global j 
    print("j:",j)
    
    
    pub_data = eposesM()
    pub_data.pos = data.pos
    
    if(j>=3):                                                    #kalman filter should start from frame3
        #list_frame_3 = data.lists
        print("running frame 3")
        list_frame_3 = []                                        #read message from topic and store them as list[list]
        for i in range(len(data.lists)):
            list_tmp = []
  
            temp1 = data.lists[i].x
            temp2 = data.lists[i].y
            temp3 = data.lists[i].z
            temp4 = data.lists[i].uncertainty
            list_tmp.append(temp1)
            list_tmp.append(temp2)
            list_tmp.append(temp3)
            list_tmp.append(temp4)
            list_frame_3.append(list_tmp)
        
        print("list_frame_3:",list_frame_3)

        global predictPositions
        global threshold
        #find match between predictions and frame3
        newlist_prediction,newlist_frame_3,list3_left = matchList_M1(predictPositions,list_frame_3,threshold)
        print("newlist_prediction:",newlist_prediction)
        print("newlist_frame_3:",newlist_frame_3)
        #find optimal result of matched item
        optimal_frame_3,uncertainty_list = fn_EKF_list_M1(newlist_prediction,newlist_frame_3)
        print("optimal_frame_3",optimal_frame_3)


        for i in range(len(list3_left)):                          #put item left by frame3 in pub_data
            pos_temp = eposfixedframeM()
  
            pos_temp.x = list3_left[i][0]
            pos_temp.y = list3_left[i][1]
            pos_temp.z = list3_left[i][2]
            pos_temp.uncertainty = list3_left[i][3]
            pos_temp.p = P_init[0][0]
            pub_data.lists.append(pos_temp)
        
        print("list3_left:",list3_left)

        for i in range(len(optimal_frame_3)):                      #put optimal items in pub_data
            pos_temp = eposfixedframeM()

            pos_temp.x = optimal_frame_3[i][0][0]
            pos_temp.y = optimal_frame_3[i][1][0]
            pos_temp.z = optimal_frame_3[i][2][0]
            pos_temp.p = P_init[0][0]
            pos_temp.uncertainty = uncertainty_list[i]
            pub_data.lists.append(pos_temp)
        #print("optimal_frame_3[1]:",optimal_frame_3[1])




    global time_frame_1                                              #time of last frame1
    time_frame_2 = time.time()                                       #time of now
    delt_time = time_frame_2 - time_frame_1                          #delt_time
    print("delt_time:",delt_time)

    global list_frame_1                                              #object list of frame1 
    print("list_frame_1:",list_frame_1)

    if(j<3):
        list_frame_2 = data.lists                                        
        list_frame_2 = []                                                #object list of frame now
        for i in range(len(data.lists)):
            list_tmp = []
 
            temp1 = data.lists[i].x
            temp2 = data.lists[i].y
            temp3 = data.lists[i].z
            temp4 = data.lists[i].uncertainty

            list_tmp.append(temp1)
            list_tmp.append(temp2)
            list_tmp.append(temp3)
            list_tmp.append(temp4)
            list_frame_2.append(list_tmp)
    
        print("list_frame_2:",list_frame_2)


    
    if(j>=3):
        list_frame_2 = []
        list_frame_2 = list_frame_2+list3_left
        #list_frame_2.append(optimal_frame_3)
        for i in range(len(optimal_frame_3)):                      #put optimal items in pub_data
            pos_temp = []
            pos_temp.append(optimal_frame_3[i][0][0])
            pos_temp.append(optimal_frame_3[i][1][0])
            pos_temp.append(optimal_frame_3[i][2][0])
            pos_temp.append(uncertainty_list[i])
            list_frame_2.append(pos_temp)
    
    global newlist_frame_1                                            #corresponding position of frame1
    newlist_frame_1, newlist_frame_2 = matchList(list_frame_1,list_frame_2,delt_time,threshold_vel)
    print("newlist_frame_1:",newlist_frame_1)
    print("newlist_frame_2:",newlist_frame_2)

    global velocity_list                                              #velocity list
    velocity_list = calVelocityList(newlist_frame_1,newlist_frame_2,delt_time)
    print("velocity_list:",velocity_list)

    global predictPositions                                           #predicted positions 
    predictPositions = predictPositionList(newlist_frame_2,velocity_list,delt_time)
    print("predictions:",predictPositions)

    time_frame_1 = time_frame_2                                       #store time of now for next loop
    list_frame_1 = list_frame_2                                       #store frame of now for next loop
    newlist_frame_1 = newlist_frame_2                                 #store corresponding match for next loop

    global pub
    if(j<3):
        #pub.publish(data1)
        for i in range(len(data.lists)):
            temp = eposfixedframeM()
            temp.x = data.lists[i].x
            temp.y = data.lists[i].y
            temp.z = data.lists[i].z
            temp.p = 100
            pub_data.lists.append(temp)
        
        #print("pub_data[0]:",pub_data.lists[0])
        #print("pub_data[1]:",pub_data.lists[1])
        print("pub_data:",pub_data.lists)
        pub.publish(pub_data)
        j = j+1
        print("success publish")
    else:
        print("pub_data:",pub_data.lists)
        pub.publish(pub_data)
        j = j+1
        print("success publish")
        #print("pub_data[0]:",pub_data.lists[0])
        #print("pub_data[1]:",pub_data.lists[1])
    



def listener():
    '''kalman Subscriber'''
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('kalman_sub', anonymous=False)

    rospy.Subscriber("chatter", eposes, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    #EKF parameters
    F = np.identity(3)
    H = np.identity(3)
    P_init = np.identity(3)*10
    Q = np.identity(3)*0.01
    R = np.identity(3)*1
    
    threshold = 0.5
    threshold_vel = 5
    j = 0
    velocity_list = []
    predictPositions = []
    list_frame_1 = []
    newlist_frame_1=[]
    time_frame_1 = time.time()

    rospy.init_node('kalman_sub', anonymous=True)
    pub = rospy.Publisher("kalman",eposesM,queue_size=10)
    rospy.Subscriber("pub_enemy_position", eposes, callback)
    rospy.spin()

    #listener()
