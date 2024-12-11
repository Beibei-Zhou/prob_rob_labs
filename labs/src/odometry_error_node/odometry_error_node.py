#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from message_filters import Subscriber, ApproximateTimeSynchronizer
import math

class OdometryErrorNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('odometry_error_node')
        rospy.loginfo('Starting odometry_error_node')

        # Publishers for the error metrics
        self.position_error_pub_ekf = rospy.Publisher('/position_error_ekf', Float64, queue_size=10)
        self.orientation_error_pub_ekf = rospy.Publisher('/orientation_error_ekf', Float64, queue_size=10)

        self.position_error_pub_eif = rospy.Publisher('/position_error_eif', Float64, queue_size=10)
        self.orientation_error_pub_eif = rospy.Publisher('/orientation_error_eif', Float64, queue_size=10)

        self.position_error_pub_raw = rospy.Publisher('/position_error_raw', Float64, queue_size=10)
        self.orientation_error_pub_raw = rospy.Publisher('/orientation_error_raw', Float64, queue_size=10)

        self.position_error_pub_filtered = rospy.Publisher('/position_error_filtered', Float64, queue_size=10)
        self.orientation_error_pub_filtered = rospy.Publisher('/orientation_error_filtered', Float64, queue_size=10)

        # Subscribers
        # Subscribe to the ground truth pose and the three odometry topics
        ground_truth_sub = Subscriber('/jackal/ground_truth/pose', PoseStamped)

        ekf_odom_sub = Subscriber('/ekf_odom', Odometry)
        eif_odom_sub = Subscriber('/eif_odom', Odometry)
        raw_odom_sub = Subscriber('/jackal_velocity_controller/odom', Odometry)
        filtered_odom_sub = Subscriber('/odometry/filtered', Odometry)

        # Synchronize the messages for EKF odometry
        self.ats_ekf = ApproximateTimeSynchronizer(
            [ground_truth_sub, ekf_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_ekf.registerCallback(self.odometry_error_callback_ekf)

        # Synchronize the messages for EIF odometry
        self.ats_eif = ApproximateTimeSynchronizer(
            [ground_truth_sub, eif_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_eif.registerCallback(self.odometry_error_callback_eif)

        # Synchronize the messages for raw odometry
        self.ats_raw = ApproximateTimeSynchronizer(
            [ground_truth_sub, raw_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_raw.registerCallback(self.odometry_error_callback_raw)

        # Synchronize the messages for filtered odometry
        self.ats_filtered = ApproximateTimeSynchronizer(
            [ground_truth_sub, filtered_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_filtered.registerCallback(self.odometry_error_callback_filtered)

        rospy.loginfo('Odometry Error Node Initialized')


    def odometry_error_callback_eif(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_eif.publish(position_error)
        self.orientation_error_pub_eif.publish(orientation_error)

        # Log the errors (optional)
        rospy.loginfo('EIF Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

    def odometry_error_callback_ekf(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_ekf.publish(position_error)
        self.orientation_error_pub_ekf.publish(orientation_error)

        # Log the errors (optional)
        rospy.loginfo('EKF Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

    def odometry_error_callback_raw(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_raw.publish(position_error)
        self.orientation_error_pub_raw.publish(orientation_error)

        # Log the errors (optional)
        # rospy.loginfo('Raw Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
        #     position_error, orientation_error, math.degrees(orientation_error)
        # ))

    def odometry_error_callback_filtered(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_filtered.publish(position_error)
        self.orientation_error_pub_filtered.publish(orientation_error)

        # Log the errors (optional)
        # rospy.loginfo('Filtered Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
        #     position_error, orientation_error, math.degrees(orientation_error)
        # ))

    def compute_errors(self, ground_truth_msg, odometry_msg):
        # Extract positions
        gt_position = ground_truth_msg.pose.position
        odom_position = odometry_msg.pose.pose.position

        # Compute position error (Euclidean distance in 2D)
        position_error = math.hypot(
            gt_position.x - odom_position.x,
            gt_position.y - odom_position.y
        )

        # Extract orientations (convert quaternions to Euler angles)
        gt_orientation_q = ground_truth_msg.pose.orientation
        odom_orientation_q = odometry_msg.pose.pose.orientation

        gt_orientation_list = [gt_orientation_q.x, gt_orientation_q.y, gt_orientation_q.z, gt_orientation_q.w]
        odom_orientation_list = [odom_orientation_q.x, odom_orientation_q.y, odom_orientation_q.z, odom_orientation_q.w]

        gt_roll, gt_pitch, gt_yaw = euler_from_quaternion(gt_orientation_list)
        odom_roll, odom_pitch, odom_yaw = euler_from_quaternion(odom_orientation_list)

        # Compute orientation error (angular difference)
        orientation_error = self.angular_difference(gt_yaw, odom_yaw)

        return position_error, orientation_error

    def angular_difference(self, angle1, angle2):
        """
        Computes the minimal difference between two angles in radians.
        The result is always in the range [-pi, pi].
        """
        diff = angle1 - angle2
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        return diff

def main():
    node = OdometryErrorNode()
    rospy.spin()
    rospy.loginfo('Shutting down odometry_error_node')

if __name__ == '__main__':
    main()
