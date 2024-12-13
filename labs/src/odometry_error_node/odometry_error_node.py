#!/usr/bin/env python

import rospy
import csv
import threading
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from message_filters import Subscriber, ApproximateTimeSynchronizer
import math
import os

class OdometryErrorNode:
    def __init__(self):

        rospy.init_node('odometry_error_node')
        rospy.loginfo('Starting odometry_error_node')

        # Log the current working directory
        current_directory = os.getcwd()
        rospy.loginfo(f'Current Working Directory: {current_directory}')

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
        # Subscribe to the ground truth pose and the four odometry topics
        ground_truth_sub = Subscriber('/jackal/ground_truth/pose', PoseStamped)

        ekf_odom_sub = Subscriber('/ekf_odom', Odometry)
        eif_odom_sub = Subscriber('/eif_odom', Odometry)
        raw_odom_sub = Subscriber('/jackal_velocity_controller/odom', Odometry)
        filtered_odom_sub = Subscriber('/odometry/filtered', Odometry)

        # Synchronize the messages for each odometry source with ground truth
        self.ats_ekf = ApproximateTimeSynchronizer(
            [ground_truth_sub, ekf_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_ekf.registerCallback(self.odometry_error_callback_ekf)

        self.ats_eif = ApproximateTimeSynchronizer(
            [ground_truth_sub, eif_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_eif.registerCallback(self.odometry_error_callback_eif)

        self.ats_raw = ApproximateTimeSynchronizer(
            [ground_truth_sub, raw_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_raw.registerCallback(self.odometry_error_callback_raw)

        self.ats_filtered = ApproximateTimeSynchronizer(
            [ground_truth_sub, filtered_odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats_filtered.registerCallback(self.odometry_error_callback_filtered)

        # Initialize CSV file with absolute path
        csv_filename = 'odometry_errors.csv'
        csv_directory = os.path.expanduser('~/ros_data')  # Change as needed
        csv_path = os.path.join(csv_directory, csv_filename)
        
        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)
        
        self.csv_lock = threading.Lock()
        self.csv_file = open(csv_path, mode='w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp',
            'position_error_eif_m',
            'orientation_error_eif_rad',
            'position_error_ekf_m',
            'orientation_error_ekf_rad',
            'position_error_raw_m',
            'orientation_error_raw_rad',
            'position_error_filtered_m',
            'orientation_error_filtered_rad'
        ])
        rospy.loginfo(f'Initialized CSV file for odometry errors at: {csv_path}')

        # Initialize a dictionary to store errors keyed by timestamp
        self.errors_dict = {}
        self.errors_lock = threading.Lock()

        # Start a timer to periodically write completed entries to CSV
        rospy.Timer(rospy.Duration(1.0), self.write_errors_to_csv)

        rospy.loginfo('Odometry Error Node Initialized')

    def odometry_error_callback_eif(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_eif.publish(position_error)
        self.orientation_error_pub_eif.publish(orientation_error)

        rospy.loginfo('EIF Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

        # Store the errors in the dictionary
        timestamp = ground_truth_msg.header.stamp.to_sec()
        with self.errors_lock:
            if timestamp not in self.errors_dict:
                self.errors_dict[timestamp] = {}
            self.errors_dict[timestamp]['eif'] = (position_error, orientation_error)

    def odometry_error_callback_ekf(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_ekf.publish(position_error)
        self.orientation_error_pub_ekf.publish(orientation_error)

        rospy.loginfo('EKF Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

        # Store the errors in the dictionary
        timestamp = ground_truth_msg.header.stamp.to_sec()
        with self.errors_lock:
            if timestamp not in self.errors_dict:
                self.errors_dict[timestamp] = {}
            self.errors_dict[timestamp]['ekf'] = (position_error, orientation_error)

    def odometry_error_callback_raw(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_raw.publish(position_error)
        self.orientation_error_pub_raw.publish(orientation_error)

        # Log the errors (optional)
        rospy.loginfo('Raw Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

        # Store the errors in the dictionary
        timestamp = ground_truth_msg.header.stamp.to_sec()
        with self.errors_lock:
            if timestamp not in self.errors_dict:
                self.errors_dict[timestamp] = {}
            self.errors_dict[timestamp]['raw'] = (position_error, orientation_error)

    def odometry_error_callback_filtered(self, ground_truth_msg, odometry_msg):
        position_error, orientation_error = self.compute_errors(ground_truth_msg, odometry_msg)

        # Publish the errors
        self.position_error_pub_filtered.publish(position_error)
        self.orientation_error_pub_filtered.publish(orientation_error)

        # Log the errors (optional)
        rospy.loginfo('Filtered Odometry - Position error: {:.4f} m, Orientation error: {:.4f} rad ({:.2f} degrees)'.format(
            position_error, orientation_error, math.degrees(orientation_error)
        ))

        # Store the errors in the dictionary
        timestamp = ground_truth_msg.header.stamp.to_sec()
        with self.errors_lock:
            if timestamp not in self.errors_dict:
                self.errors_dict[timestamp] = {}
            self.errors_dict[timestamp]['filtered'] = (position_error, orientation_error)

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
        
        diff = angle1 - angle2
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        return diff

    def write_errors_to_csv(self, event):
        # Periodically called function to write completed error entries to CSV.
        with self.errors_lock:
            timestamps = sorted(self.errors_dict.keys())
            for timestamp in timestamps:
                entry = self.errors_dict[timestamp]
                if all(key in entry for key in ['eif', 'ekf', 'raw', 'filtered']):
                    # All errors are present for this timestamp
                    pos_eif, ori_eif = entry['eif']
                    pos_ekf, ori_ekf = entry['ekf']
                    pos_raw, ori_raw = entry['raw']
                    pos_filtered, ori_filtered = entry['filtered']

                    # Write to CSV
                    with self.csv_lock:
                        self.csv_writer.writerow([
                            timestamp,
                            pos_eif,
                            ori_eif,
                            pos_ekf,
                            ori_ekf,
                            pos_raw,
                            ori_raw,
                            pos_filtered,
                            ori_filtered
                        ])
                        self.csv_file.flush()  # Ensure data is written to disk

                    # Remove the entry from the dictionary
                    del self.errors_dict[timestamp]

    def shutdown_hook(self):
        rospy.loginfo('Shutting down odometry_error_node. Closing CSV file.')
        with self.csv_lock:
            self.csv_file.close()

def main():
    node = OdometryErrorNode()
    rospy.on_shutdown(node.shutdown_hook)
    rospy.spin()

if __name__ == '__main__':
    main()
