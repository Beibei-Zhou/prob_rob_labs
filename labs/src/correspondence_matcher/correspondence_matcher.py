#!/usr/bin/env python

import rospy
from opencv_apps.msg import Point2DArrayStamped
from std_msgs.msg import Float32MultiArray
import message_filters
import numpy as np

class CorrespondenceMatcher:
    def __init__(self):
        rospy.init_node('correspondence_matcher')
        rospy.loginfo('Starting correspondence_matcher')

        # Parameters
        self.landmark_color = rospy.get_param('~landmark_color', 'cyan')

        # Subscribers
        predicted_topic = '/expected_features_' + self.landmark_color
        measured_topic = '/goodfeature_' + self.landmark_color + '/corners'

        self.predicted_sub = message_filters.Subscriber(predicted_topic, Point2DArrayStamped)
        self.measured_sub = message_filters.Subscriber(measured_topic, Point2DArrayStamped)

        # Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.predicted_sub, self.measured_sub],
            queue_size=10,
            slop=0.5 
        )
        self.ts.registerCallback(self.callback)

        # Publisher for the distances
        self.distance_pub = rospy.Publisher('/feature_matching_error', Float32MultiArray, queue_size=10)

    def callback(self, predicted_msg, measured_msg):
        rospy.loginfo('Callback function called')

        # Extract predicted features
        predicted_points = predicted_msg.points
        if len(predicted_points) != 4:
            rospy.logwarn('Expected 4 predicted points, got %d', len(predicted_points))
            return

        predicted_features = np.array([[p.x, p.y] for p in predicted_points])  # Shape (4, 2)

        # Extract measured features
        measured_points = measured_msg.points
        if len(measured_points) == 0:
            rospy.logwarn('No measured features received')
            return

        measured_features = np.array([[p.x, p.y] for p in measured_points])  # Shape (N, 2)

        # Initialize an array to store the distances
        distances = []

        # For each predicted feature, find the closest measured feature
        for idx, pred_point in enumerate(predicted_features):
            min_dist = None
            closest_meas = None
            for meas_point in measured_features:
                dist = np.linalg.norm(pred_point - meas_point)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    closest_meas = meas_point
            if min_dist is not None:
                distances.append(min_dist)
                rospy.loginfo('Predicted Point %d matched with Measured Point (%.2f, %.2f) with distance %.2f', 
                              idx+1, closest_meas[0], closest_meas[1], min_dist)
            else:
                distances.append(float('inf'))  # No matching point found
                rospy.logwarn('No matching measured point found for Predicted Point %d', idx+1)

        # Publish the distances
        distances_msg = Float32MultiArray()
        distances_msg.data = distances
        self.distance_pub.publish(distances_msg)

        # Log the distances
        rospy.loginfo('Feature matching distances: %s', distances)

    def run(self):
        rospy.spin()
        rospy.loginfo('Shutting down correspondence_matcher')

def main():
    matcher = CorrespondenceMatcher()
    matcher.run()

if __name__ == '__main__':
    main()
