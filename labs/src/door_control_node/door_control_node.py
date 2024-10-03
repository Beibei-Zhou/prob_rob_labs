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

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class RobotDoorController:
    def __init__(self):
        # Publisher to control the door torque
        self.door_pub = rospy.Publisher('/hinged_glass_door/torque', Float64, queue_size=10)
       
        # Publisher to control the robot's linear velocity
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
       
        # Initialize variables
        self.start_time = None
        rospy.loginfo('start time is {}'.format(self.start_time))
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("RobotDoorController initialized.")

    def control_loop(self, _):
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
        if elapsed_time < 15:
            # First 5 seconds: Open the door
            self.manipulate_door(30.0)
            rospy.loginfo("Opening the door...")
            #rospy.loginfo(elapsed_time)
            #rospy.loginfo(self.start_time)
            return
        if elapsed_time < 18:
            self.move_robot()
            rospy.loginfo("Move the robot...")
            return
        if elapsed_time < 23:
            self.stop_robot()
            rospy.loginfo("Stop the robot...")
            return
        self.manipulate_door(-5.0)
        #rospy.loginfo("Close the door...")
        #rospy.loginfo(elapsed_time)
        #rospy.loginfo(self.start_time)
 
    def manipulate_door(self, t):
        # Apply negative torque to close the door
        torque_msg = Float64()
        torque_msg.data = t  # Adjust as needed
        self.door_pub.publish(torque_msg)

    def move_robot(self):
        # Move the robot forward
        twist_msg = Twist()
        twist_msg.linear.x = 2.0  # Adjust speed as needed
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



