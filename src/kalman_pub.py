#!/usr/bin/env python
'''kalman ROS Node'''
# license removed for brevity
import rospy
from std_msgs.msg import String
from gdp.msg import eposes, eposfixedframe
from gdp.msg import eposesM, eposfixedframeM

def talker():
    '''kalman Publisher'''
    pub = rospy.Publisher("chatter", eposes, queue_size=10)
    rospy.init_node('kalman_pub', anonymous=False)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        poses = eposes()

        for i in range(3):
            pose_temp = eposfixedframe()
            pose_temp.x = 0
            pose_temp.y = 1
            pose_temp.z = 2
            #pose_temp.pose.append(0)
            #pose_temp.pose.append(1)
            #pose_temp.pose.append(2)

            poses.lists.append(pose_temp)
        
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(poses)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
