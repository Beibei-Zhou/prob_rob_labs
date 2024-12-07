import rospy
from opencv_apps.msg import Point2DArrayStamped
from std_msgs.msg import Float32MultiArray
import message_filters
import numpy as np
import csv
import os

class CorrespondenceMatcher:
    def __init__(self):
        rospy.init_node('correspondence_matcher')
        rospy.loginfo('Starting correspondence_matcher')

        # Parameters
        # self.landmark_color = rospy.get_param('~landmark_color', 'cyan')
        self.landmark_color = rospy.get_param('~landmark_color','cyan')
        
        # Subscribers
        predicted_topic = '/expected_features_' + self.landmark_color
        measured_topic = '/goodfeature_' + self.landmark_color + '/corners'

        self.predicted_sub = message_filters.Subscriber(predicted_topic, Point2DArrayStamped)
        self.measured_sub = message_filters.Subscriber(measured_topic, Point2DArrayStamped)

        # Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.predicted_sub, self.measured_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # Publishers
        self.error_pub = rospy.Publisher('/feature_matching_error', Float32MultiArray, queue_size=10)

        # Dictionaries to store error data for each point
        self.x_errors = {i: [] for i in range(4)}  
        self.y_errors = {i: [] for i in range(4)}

        # Record the start time using ROS time
        self.start_time = rospy.get_time()

        # Initialize CSV file
        self.initialize_csv()

        # Register shutdown hook to close CSV file
        rospy.on_shutdown(self.shutdown_hook)

    def initialize_csv(self):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Define the CSV file path
        self.csv_filename = os.path.join(script_dir, f'feature_matching_errors_{self.landmark_color}.csv')
        try:
            self.csv_file = open(self.csv_filename, mode='w', newline='')  # Added newline=''
            self.csv_writer = csv.writer(self.csv_file)
            # Header includes time since start (seconds)
            header = ['Time_since_start_s',
                      'Point1_x_diff', 'Point1_y_diff',
                      'Point2_x_diff', 'Point2_y_diff',
                      'Point3_x_diff', 'Point3_y_diff',
                      'Point4_x_diff', 'Point4_y_diff']
            self.csv_writer.writerow(header)
            rospy.loginfo(f'Initialized CSV file for logging: {self.csv_filename}')
        except Exception as e:
            rospy.logerr(f'Failed to initialize CSV file: {e}')
            rospy.signal_shutdown('CSV Initialization Failure')

    def shutdown_hook(self):
        try:
            self.csv_file.close()
            rospy.loginfo(f'Closed CSV file: {self.csv_filename}')
        except Exception as e:
            rospy.logerr(f'Error closing CSV file: {e}')

    def callback(self, predicted_msg, measured_msg):
        # Extract predicted features
        predicted_points = predicted_msg.points
        if len(predicted_points) != 4:
            rospy.logwarn('Expected 4 predicted points, got %d', len(predicted_points))
            return

        predicted_indices = np.arange(len(predicted_points))
        predicted_features = np.array([[p.x, p.y] for p in predicted_points])

        # Extract measured features
        measured_points = measured_msg.points
        if len(measured_points) == 0:
            rospy.logwarn('No measured features received')
            return

        measured_indices = np.arange(len(measured_points))
        measured_features = np.array([[p.x, p.y] for p in measured_points])

        # Sort predicted features based on (x + y)
        pred_sort_keys = predicted_features[:, 0] + predicted_features[:, 1]
        sorted_pred_order = np.argsort(pred_sort_keys)
        predicted_features_sorted = predicted_features[sorted_pred_order]
        predicted_indices_sorted = predicted_indices[sorted_pred_order]

        # Sort measured features based on (x + y)
        meas_sort_keys = measured_features[:, 0] + measured_features[:, 1]
        sorted_meas_order = np.argsort(meas_sort_keys)
        measured_features_sorted = measured_features[sorted_meas_order]
        measured_indices_sorted = measured_indices[sorted_meas_order]

        # Determine the number of matches
        num_matches = min(len(predicted_features_sorted), len(measured_features_sorted))

        # Initialize arrays to store the differences
        x_diffs = []
        y_diffs = []

        # Compute the time since start
        time_since_start = rospy.get_time() - self.start_time

        # Prepare row for CSV
        csv_row = [time_since_start]

        # Match features based on sorted order
        for idx in range(4):
            if idx < num_matches:
                pred_point = predicted_features_sorted[idx]
                meas_point = measured_features_sorted[idx]

                orig_pred_idx = sorted_pred_order[idx] + 1
                orig_meas_idx = sorted_meas_order[idx] + 1

                x_diff = meas_point[0] - pred_point[0]
                y_diff = meas_point[1] - pred_point[1]
                x_diffs.append(x_diff)
                y_diffs.append(y_diff)

                self.x_errors[idx].append(x_diff)
                self.y_errors[idx].append(y_diff)

                rospy.loginfo(
                    'Predicted Point %d (%.2f, %.2f) matched with Measured Point %d (%.2f, %.2f) with x_diff %.2f, y_diff %.2f',
                    orig_pred_idx, pred_point[0], pred_point[1],
                    orig_meas_idx, meas_point[0], meas_point[1],
                    x_diff, y_diff
                )

                # Append to CSV row
                csv_row.extend([x_diff, y_diff])
            else:
                # No matching measured point found for this predicted point
                x_diffs.append(0.0)
                y_diffs.append(0.0)
                orig_pred_idx = sorted_pred_order[idx] + 1
                rospy.logwarn('No matching measured point found for Predicted Point %d', orig_pred_idx)
                csv_row.extend([0.0, 0.0])

        # Write the row to CSV
        try:
            self.csv_writer.writerow(csv_row)
        except Exception as e:
            rospy.logerr(f'Failed to write to CSV file: {e}')

        # Publish the differences
        error_msg = Float32MultiArray()
        error_msg.data = x_diffs + y_diffs
        self.error_pub.publish(error_msg)

        self.calculate_variance()

    def calculate_variance(self):
        variances = {}
        for idx in range(4):
            if self.x_errors[idx] and self.y_errors[idx]:
                x_var = np.var(self.x_errors[idx])
                y_var = np.var(self.y_errors[idx])
                variances[f'Point_{idx+1}_x'] = x_var
                variances[f'Point_{idx+1}_y'] = y_var
                rospy.loginfo('Variance for Point %d - X: %.4f, Y: %.4f', idx+1, x_var, y_var)
            else:
                rospy.logwarn('No error data collected yet for Point %d', idx+1)
                variances[f'Point_{idx+1}_x'] = None
                variances[f'Point_{idx+1}_y'] = None
        return variances

    def run(self):
        rate = rospy.Rate(1)
        rospy.spin()
        # while not rospy.is_shutdown():
        #     variances = self.calculate_variance()
        #     rate.sleep()
        #rospy.loginfo('Shutting down correspondence_matcher')

def main():
    matcher = CorrespondenceMatcher()
    matcher.run()

if __name__ == '__main__':
    main()
