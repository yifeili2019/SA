#!/usr/bin/env python
'''findenemy ROS Node'''
# license removed for brevity
import rospy
from std_msgs.msg import String
from gdp.msg import position_detection
from gdp.msg import multipositions
from gdp.msg import eposfixedframe
from gdp.msg import eposes,Action,Agent

def talker():
    '''findenemy Publisher'''
    #pub = rospy.Publisher('findenemy_pub1', String, queue_size=10)
    pub = rospy.Publisher('fake_pub', Agent, queue_size=10)
    rospy.init_node('fake_pub', anonymous=False)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        #hello_str = "hello world %s" % rospy.get_time()
        #rospy.loginfo(hello_str)
        poses = Agent()
        poses.pos.data.append(0)
        poses.pos.data.append(0)
        poses.pos.data.append(0)

        poses.attitude.data.append(0)   #rad
        poses.attitude.data.append(0)
        poses.attitude.data.append(0)

        print("speaker1")
        pub.publish(poses)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
