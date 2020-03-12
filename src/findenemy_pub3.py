#!/usr/bin/env python
'''findenemy ROS Node'''
# license removed for brevity
import rospy
from std_msgs.msg import String
from gdp.msg import eposfixedframeM, eposesM

def talker():
    '''findenemy Publisher'''
    #pub = rospy.Publisher('findenemy_pub1', String, queue_size=10)
    pub = rospy.Publisher('findenemy_pub3', eposesM, queue_size=10)
    rospy.init_node('findenemy_pub3', anonymous=False)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        #hello_str = "hello world %s" % rospy.get_time()
        #rospy.loginfo(hello_str)
        poses = eposesM()
        poses.pos.data.append(0)
        poses.pos.data.append(1)
        poses.pos.data.append(2)
        for i in range(3):
            pose_temp = eposfixedframeM()
            pose_temp.x = 0
            pose_temp.y = 1
            pose_temp.z = 2
            pose_temp.p = 100
            pose_temp.uncertainty = 20

            poses.lists.append(pose_temp)

        print("speaker3")
        pub.publish(poses)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
