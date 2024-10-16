#!/usr/bin/env python
'''
import rospy


def main():
    rospy.init_node('door_control_node')
    rospy.loginfo('starting door_control_node')
    rospy.spin()
    rospy.loginfo('done')
    
if __name__=='__main__':

    main()
'''
import rospy
import time

from std_msgs.msg import Float64, Empty
from geometry_msgs.msg import Twist

# Thresholds for detecting door status
#THRESHOLD_OPEN = 420.0
#THRESHOLD_CLOSED = 480.0
#MOVE_TIME = 4.0

# Thresholds for feature_mean values
MEASUREMENT_THRESHOLD = 448

PRIOR_PROB_OPEN = 0.5
PROB_FEATURE_GIVEN_OPEN = 0.9
PROB_DOOR_OPEN_REQUEST = 0.18
PROB_DOOR_NOT_OPEN_REQUEST = 0.82
PROB_FEATURE_GIVEN_CLOSED= 0.5
DESIRED_PROB_OPEN = 0.99
MOVE_TIME = 4.0

class RobotDoorController_draft:
    def __init__(self):
        # Publisher to control the door torque
        self.door_pub = rospy.Publisher('/hinged_glass_door/torque', Float64, queue_size=10)
        self.door_open_pub = rospy.Publisher('/door_open', Empty, queue_size=10)
        # Publisher to control the robot's linear velocity
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber to the feature_mean topic
        self.feature_mean_sub = rospy.Subscriber('/feature_mean', Float64, self.feature_mean_callback)
        
        # Initialize variables
        self.feature_mean = None
        self.state = 0
        self.start_time = None
        # rospy.loginfo('start time is {}'.format(self.start_time))
        self.prob_open = 0.5 # Initialize belief about the door being open
        self.measurement_count = 0 # Count for the number of measurements taken
        self.rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("RobotDoorController initialized.")
        # Read the robot speed parameter with a default value of 0.0
        self.robot_speed = rospy.get_param('~robot_speed',1.5)
        rospy.loginfo("RobotController initialized with robot speed:{}".format(self.robot_speed))
        
    def feature_mean_callback(self, msg):
        self.feature_mean = msg.data
        rospy.loginfo("feature_mean_callback called with data: {}".format(msg.data))
        self.perform_measurement_update()
    
    def perform_prediction_step(self):
        prev_prob_open = self.prob_open
        self.prob_open = prev_prob_open + (1 - prev_prob_open) * PROB_DOOR_OPEN_REQUEST


    def control_loop(self, event):
        if self.state == 0:
            if self.prob_open < DESIRED_PROB_OPEN:
                rospy.loginfo("Attempting to open the door...")
                # Send open request
                self.door_open_pub.publish(Empty())
                # Perform prediction step
                self.perform_prediction_step()
                rospy.loginfo("Predicted probability door is open:{:.4f}".format(self.prob_open))
            else:
                rospy.loginfo("Door is open with high confidence. Proceeding to move the robot...")
                self.state = 1
                self.start_time = rospy.get_time()
            #self.manipulate_door(150.0)
            #rospy.loginfo("Opening the door...")
            # if self.feature_mean is not None:
            #     self.perform_bayesian_update()
            #     if self.prob_open >= DESIRED_PROB_OPEN:
            #         rospy.loginfo("Probability door is open:{:.4f}".format(self.prob_open))
            #         rospy.loginfo("Door is open with high confidence. Proceeding to move the robot...")
            #         self.state = 1
            #         self.start_time = rospy.get_time()
                    
            #rospy.loginfo(self.feature_mean)
            #if self.feature_mean is not None and self.feature_mean < THRESHOLD_OPEN:
                #rospy.loginfo("Door is open. Proceeding to move the robot...")
                #self.state = 1
                #self.start_time = rospy.get_time()
        elif self.state == 1:
            self.move_robot()
            rospy.loginfo("Moving the robot forward...")
            elapsed_time = rospy.get_time() - self.start_time
            if elapsed_time > MOVE_TIME:
                self.state = 2
                self.stop_robot()
                rospy.loginfo("Stopping the robot...")
                self.start_time = rospy.get_time()
        elif self.state == 2:
            self.manipulate_door(-30.0)
            rospy.loginfo("Closing the door")
            #if self.feature_mean is not None and self.feature_mean > THRESHOLD_CLOSED:
                #rospy.loginfo("Door is closed...")
                #self.state = 3
            self.state = 3
        else:
            rospy.loginfo("Process completed.")

    def perform_measurement_update(self):
        if self.feature_mean > MEASUREMENT_THRESHOLD:
            measurement = 'high' # Indicative of closed door
        else:
            measurement = 'low' # Indicative of open door
        P_measurement_given_open = PROB_FEATURE_GIVEN_OPEN if measurement == 'low' else 1 - PROB_FEATURE_GIVEN_OPEN
        P_measurement_given_closed = PROB_FEATURE_GIVEN_CLOSED if measurement == 'low' else 1 - PROB_FEATURE_GIVEN_CLOSED

        unnormalized_open = P_measurement_given_open * self.prob_open
        unnormalized_closed = P_measurement_given_closed * (1 - self.prob_open)
        #P_measurement = (P_measurement_given_open * self.prob_open) + (P_measurement_given_closed * (1 - self.prob_open))
        P_measurement = unnormalized_open + unnormalized_closed
        self.prob_open = unnormalized_open / P_measurement
        self.measurement_count += 1
        rospy.loginfo("Updated probability door is open: {:.4f}".format(self.prob_open))
        rospy.loginfo("Number of measurements taken :{}".format(self.measurement_count))

    def manipulate_door(self, t):
        # Apply negative torque to close the door
        torque_msg = Float64()
        torque_msg.data = t  # Adjust as neededx
        self.door_pub.publish(torque_msg)

    def move_robot(self):
        # Move the robot forward
        twist_msg = Twist()
        # Adjust speed as needed
        twist_msg.linear.x = 1.5
        
        self.cmd_vel_pub.publish(twist_msg)

    def stop_robot(self):
        # Stop the robot
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(twist_msg)

def main():
    rospy.init_node('door_control_node')
    controller = RobotDoorController_draft()
    timer = rospy.Timer(rospy.Duration(0.1), controller.control_loop)
    rospy.spin()
    rospy.loginfo('Node has shut down.')

if __name__ == '__main__':
    main()




