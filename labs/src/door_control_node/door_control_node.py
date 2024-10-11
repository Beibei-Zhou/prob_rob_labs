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

from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import Twist

# Thresholds for detecting door status
THRESHOLD_OPEN = 420.0
THRESHOLD_CLOSED = 480.0
MOVE_TIME = 4.0

class RobotDoorController:
    def __init__(self):
        # Publisher to control the door torque
        self.door_pub = rospy.Publisher('/hinged_glass_door/torque', Float64, queue_size=10)
       
        # Publisher to control the robot's linear velocity
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber to the feature_mean topic
        self.feature_mean_sub = rospy.Subscriber('/feature_mean', Float64, self.feature_mean_callback)
        
        # Initialize variables
        self.feature_mean = None
        self.state = 0
        self.start_time = None
        # rospy.loginfo('start time is {}'.format(self.start_time))
        self.rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("RobotDoorController initialized.")
        # Read the robot speed parameter with a default value of 0.0
        self.robot_speed = rospy.get_param('~robot_speed',1.5)
        rospy.loginfo("RobotController initialized with robot speed:{}".format(self.robot_speed))
    def feature_mean_callback(self, msg):
        self.feature_mean = msg.data
        rospy.loginfo("feature_mean_callback called with data: {}".format(msg.data))
    def control_loop(self, event):
        if self.state == 0:
            self.manipulate_door(1.5)
            rospy.loginfo("Opening the door...")
            rospy.loginfo(self.feature_mean)
            if self.feature_mean is not None and self.feature_mean < THRESHOLD_OPEN:
                rospy.loginfo("Door is open. Proceeding to move the robot...")
                self.state = 1
                self.start_time = rospy.get_time()
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
            if self.feature_mean is not None and self.feature_mean > THRESHOLD_CLOSED:
                rospy.loginfo("Door is closed...")
                self.state = 3
        else:
            rospy.loginfo("Process completed.")
    def control_loop_time(self, _):
        now = rospy.get_rostime().to_sec()
        if not now:
            rospy.logwarn('not initialized')
            return
        if not self.start_time:
            rospy.loginfo('initialized time to {}'.format(now))
            self.start_time = now
            return
        elapsed_time = now - self.start_time
        rospy.loginfo('now is {}, elapsed time is {}'.format(now, elapsed_time))
        if elapsed_time < 3:
            # First 5 seconds: Open the door
            self.manipulate_door(30.0)
            rospy.loginfo("Opening the door...")
            rospy.loginfo(elapsed_time)
            rospy.loginfo(self.start_time)
            return
        if elapsed_time < 10:
            self.move_robot()
            rospy.loginfo("Move the robot...")
            return
        if elapsed_time < 12:
            self.stop_robot()
            rospy.loginfo("Stop the robot...")
            return
        self.manipulate_door(-30.0)
        rospy.loginfo("Close the door...")

 
    def manipulate_door(self, t):
        # Apply negative torque to close the door
        torque_msg = Float64()
        torque_msg.data = t  # Adjust as neededx
        self.door_pub.publish(torque_msg)

    def move_robot(self):
        # Move the robot forward
        twist_msg = Twist()
        # Adjust speed as needed
        twist_msg.linear.x = self.robot_speed
        
        self.cmd_vel_pub.publish(twist_msg)

    def stop_robot(self):
        # Stop the robot
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(twist_msg)

def main():
    rospy.init_node('door_control_node')
    controller = RobotDoorController()
    timer = rospy.Timer(rospy.Duration(0.1), controller.control_loop)
    rospy.spin()
    rospy.loginfo('Node has shut down.')

if __name__ == '__main__':
    main()



