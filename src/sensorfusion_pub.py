#!/usr/bin/env python
'''sensorfusion ROS Node'''
# license removed for brevity
import rospy
from std_msgs.msg import String
from sf.msg import poses, enemyAndFriends

def talker():
    '''sensorfusion Publisher'''
    pub = rospy.Publisher('sensorfusion_pub', enemyAndFriends, queue_size=10)
    rospy.init_node('sensorfusion_pub', anonymous=False)
    rate = rospy.Rate(10) # 10hz

    #pub_data = enemyAndFriends()
    while not rospy.is_shutdown():
        pub_data = enemyAndFriends()
        enemy_temp = poses()
        friend_temp = poses()

        enemy_temp.pos.data.append(0)
        enemy_temp.pos.data.append(1)
        enemy_temp.pos.data.append(2)
        enemy_temp.p = 100

        friend_temp.pos.data.append(0)
        friend_temp.pos.data.append(1)
        friend_temp.pos.data.append(2)
        friend_temp.p = 100
            

        pub_data.enemylists.append(enemy_temp)
        pub_data.friendlists.append(friend_temp)
        print("speaker")
        pub.publish(pub_data)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
