#!/usr/bin/env python

import rospy
#from std_msgs.msg import Float64, Empty
from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Header

class Ground_Truth:
    def __init__(self):
        self.ground_truth_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.ground_truth_callback)
        self.ground_truth_Pose_pub = rospy.Publisher('/jackal/ground_truth/pose', PoseStamped, queue_size=1)
        self.ground_truth_Twist_pub = rospy.Publisher('/jackal/ground_truth/pose', TwistStamped, queue_size=1)
        self.pose_stamp = None
        self.twist_stamp = None
        self.rate = rospy.Rate(10)
        self.index = 1
    
    def ground_truth_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == "base_link":
                break
        self.index = i
        current = rospy.get_rostime
        self.pose_stamp = PoseStamped(
            header = Header(
                stamp = current,
                frame_id = 'odom'
            ),
            pose = msg.pose[self.index]
        )

        self.twist_stamp = TwistStamped(
            header = Header(
                stamp = current,
                frame_id = 'odom'
            ),
            twist = msg.twist[self.index]
        )

        self.ground_truth_Pose_pub.publish(self.pose_stamp)
        
        self.ground_truth_Twist_pub.publish(self.twist_stamp)
        rospy.loginfo("Publish Pose {}".format(self.pose_stamp))
        rospy.loginfo("Publish Twist{}".format(self.twist_stamp))
        
def main():
    rospy.init_node('ground_truth_extraction')
    GT_find = Ground_Truth()
    GT_find.ground_truth_callback
    rospy.loginfo('starting ground_truth_extraction')
    rospy.spin()
    rospy.loginfo('done')

if __name__=='__main__':
    main()
