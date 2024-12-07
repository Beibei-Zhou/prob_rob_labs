#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64

class EKFErrorNode:
    def __init__(self):
        rospy.init_node('ekf_error_node', anonymous=True)

        # Parameters for topic names
        self.ground_truth_topic = rospy.get_param('~ground_truth_topic', '/jackal/ground_truth/pose')
        self.ekf_pose_topic = rospy.get_param('~ekf_pose_topic', '/ekf_pose')

        # Subscribers & Publishers
        self.gt_sub = rospy.Subscriber(self.ground_truth_topic, PoseStamped, self.ground_truth_callback)
        self.ekf_sub = rospy.Subscriber(self.ekf_pose_topic, PoseWithCovarianceStamped, self.ekf_pose_callback)
        self.pos_error_x_pub = rospy.Publisher('/ekf_error/pos_error_x', Float64, queue_size=10)
        self.pos_error_y_pub = rospy.Publisher('/ekf_error/pos_error_y', Float64, queue_size=10)
        self.orient_error_pub = rospy.Publisher('/ekf_error/orient_error', Float64, queue_size=10)

        # Store last received poses
        self.gt_pose = None
        self.ekf_pose = None

        # rospy.loginfo("EKF Error Node Initialized")

    def ground_truth_callback(self, msg):
        self.gt_pose = msg
        self.compute_and_publish_error()

    def ekf_pose_callback(self, msg):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.ekf_pose = ps
        self.compute_and_publish_error()

    def compute_and_publish_error(self):

        # Extract positions
        gt_x = self.gt_pose.pose.position.x
        gt_y = self.gt_pose.pose.position.y

        ekf_x = self.ekf_pose.pose.position.x
        ekf_y = self.ekf_pose.pose.position.y

        # Position error
        pos_error_x = ekf_x - gt_x
        pos_error_y = ekf_y - gt_y

        # Extract orientations (convert quaternion to yaw)
        gt_yaw = self.quat_to_yaw(self.gt_pose.pose.orientation)
        ekf_yaw = self.quat_to_yaw(self.ekf_pose.pose.orientation)

        # Orientation error (normalized to [-pi, pi])
        orient_error = ekf_yaw - gt_yaw
        orient_error = (orient_error + math.pi) % (2 * math.pi) - math.pi

        # Publish errors
        self.pos_error_x_pub.publish(Float64(pos_error_x))
        self.pos_error_y_pub.publish(Float64(pos_error_y))
        self.orient_error_pub.publish(Float64(orient_error))

        rospy.loginfo("Published Errors - Pos X: %.4f, Pos Y: %.4f, Orient: %.4f",pos_error_x, pos_error_y, orient_error)

    def quat_to_yaw(self, orientation):
        q = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(q)
        return yaw


def main():
    ekf_error_node = EKFErrorNode()
    rospy.spin()


if __name__ == '__main__':
    main()
