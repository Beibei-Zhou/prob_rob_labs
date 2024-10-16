#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import numpy as np

class Probabilities_estimator:
    def __init__(self):
        self.feature_mean_sub = rospy.Subscriber('/feature_mean', Float64, self.feature_mean_callback)
        # Initialize the feature_mean
        self.feature_mean = None
        self.start_time = None
        self.rate = rospy.Rate(10)
        self.record = np.array([])
        self.time_window = 120
        self.threshold = 448
        self.cnt = 0

    def feature_mean_callback(self,msg):
        if self.start_time is None:
            self.start_time = rospy.get_time()
            return
        self.feature_mean = msg.data
        elapsed_time = rospy.get_time() - self.start_time
        rospy.loginfo("Elapsed Time:{}".format(elapsed_time))
        if elapsed_time <= self.time_window:
            self.record =  np.append(self.record, self.feature_mean)
            if self.feature_mean < self.threshold:
                self.cnt += 1
        else:
            rospy.loginfo('Done')

        rospy.loginfo("Number of the records:{}".format(len(self.record)))
        rospy.loginfo("P(z=open|x=closed)={}".format(self.cnt/len(self.record)))
        rospy.loginfo("P(z=closed|x=closed)={}".format(1 - (self.cnt/len(self.record))))
        #rospy.loginfo("feature_mean_callback with data:{}".format(msg.data))
def main():
    rospy.init_node('find_thres_noise')
    estimator = Probabilities_estimator()
    timer = rospy.Timer(rospy.Duration(0.1), estimator.feature_mean_callback)
    rospy.spin()
    rospy.loginfo('done')

if __name__=='__main__':
    main()
