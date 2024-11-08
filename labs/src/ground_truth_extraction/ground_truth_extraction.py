#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Header

class Ground_Truth:
    def __init__(self):
        self.ground_truth_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.ground_truth_callback, queue_size=10)
        self.ground_truth_Pose_pub = rospy.Publisher('/jackal/ground_truth/pose', PoseStamped, queue_size=10)
        self.ground_truth_Twist_pub = rospy.Publisher('/jackal/ground_truth/twist', TwistStamped, queue_size=10)
        self.pose_stamp = None
        self.twist_stamp = None
        self.index = None  # Initialize index as None

    def ground_truth_callback(self, msg):
        if self.index is None:
            try:
                self.index = msg.name.index("jackal::base_link")
                rospy.loginfo("Found jackal::base_link at index {}".format(self.index))
            except ValueError:
                rospy.logwarn("jackal::base_link not found in link_states")
                return  # Cannot proceed without index

        current = rospy.Time.now()

        self.pose_stamp = PoseStamped(
            header=Header(
                stamp=current,
                frame_id='odom'
            ),
            pose=msg.pose[self.index]
        )

        self.twist_stamp = TwistStamped(
            header=Header(
                stamp=current,
                frame_id='odom'
            ),
            twist=msg.twist[self.index]
        )

        self.ground_truth_Pose_pub.publish(self.pose_stamp)
        self.ground_truth_Twist_pub.publish(self.twist_stamp)

        # Reduce logging to prevent delays
        rospy.loginfo("Publish Pose {}".format(self.pose_stamp))
        rospy.loginfo("Publish Twist {}".format(self.twist_stamp))

def main():
    rospy.init_node('ground_truth_extraction', anonymous=True)
    GT_find = Ground_Truth()
    rospy.loginfo('Starting ground_truth_extraction')
    rospy.spin()
    rospy.loginfo('Shutting down')

if __name__=='__main__':
    main()
