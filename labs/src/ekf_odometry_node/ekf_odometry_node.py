#!/usr/bin/env python

# import rospy


# def main():
#     rospy.init_node('ekf_odometry_node')
#     rospy.loginfo('starting ekf_odometry_node')
#     rospy.spin()
#     rospy.loginfo('done')

# if __name__=='__main__':
#     main()


import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf.transformations import quaternion_from_euler

class EKFOdometryNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('ekf_odometry_node')
        rospy.loginfo('Starting ekf_odometry_node')

        # Robot parameters from the URDF
        self.r_w = 0.098      # Wheel radius in meters
        self.track = 0.37559  # Distance between left and right wheels (track width)
        self.robot_radius = self.track / 2  # Half of the track width (robot radius)

        # EKF model parameters
        self.a_v = 0.383      # Forgetting factor for linear velocity
        self.G_v = 1.0        # Input gain for linear velocity
        self.a_w = 0.123      # Forgetting factor for angular velocity
        self.G_w = 1.3        # Input gain for angular velocity

        # Initialize state vector [x, y, theta, v, omega]
        self.state = np.zeros(5, dtype=np.float64)
        # Initialize covariance matrix P (5x5)
        self.P = np.eye(5, dtype=np.float64) * 0.1  # Small initial uncertainty

        # Process noise covariance Q (5x5)
        # self.Q = np.diag(np.array([0.001, 0.001, 0.001, 0.005, 0.005], dtype=np.float64))
        self.Q = np.diag(np.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float64))
        # Measurement noise covariance R (5x5)
        encoder_var = (0.05) ** 2  # Increased variance for encoder measurements
        gyro_var = (0.02) ** 2     # Increased variance for gyroscope measurement
        self.R = np.diag(np.array([encoder_var, encoder_var, encoder_var, encoder_var, gyro_var], dtype=np.float64))

        # Time of the last update
        self.last_time = None

        # Latest control inputs (assume zero initially)
        self.u_v = 0.0
        self.u_w = 0.0
        self.last_cmd_time = rospy.Time.now()

        # Gyro bias estimation
        self.gyro_bias = 0.0  # Initialize to zero

        # Publisher for the odometry message
        self.ekf_odom_pub = rospy.Publisher('/ekf_odom', Odometry, queue_size=10)

        # Subscribers with ApproximateTimeSynchronizer
        imu_sub = Subscriber('/imu/data', Imu, queue_size=10)
        joint_states_sub = Subscriber('/joint_states', JointState, queue_size=10)

        # Synchronize IMU and joint states messages Increased slop to 0.1 seconds
        self.ats = ApproximateTimeSynchronizer(
            [imu_sub, joint_states_sub],
            queue_size=10,
            slop=0.1  # Increased slop to 0.1 seconds
        )
        self.ats.registerCallback(self.ekf_callback)

        # Separate subscriber for cmd_vel
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo('EKF Odometry Node Initialized')

    def cmd_vel_callback(self, msg):
        self.u_v = msg.linear.x
        self.u_w = msg.angular.z
        self.last_cmd_time = rospy.Time.now()
        rospy.loginfo(f"cmd_vel updated: u_v={self.u_v}, u_w={self.u_w}")

    def ekf_callback(self, imu_msg, joint_states_msg):
        rospy.loginfo("ekf_callback triggered")
        # Use the latest control inputs
        rospy.loginfo(f"Using control inputs: u_v={self.u_v}, u_w={self.u_w}")

        # Get the current time from the IMU message
        current_time = imu_msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            rospy.loginfo("First callback execution. Setting last_time and returning.")
            return  # Need at least two timestamps to compute delta_t
        delta_t = current_time - self.last_time
        if delta_t <= 0:
            rospy.logwarn('Delta time is non-positive. Skipping this update.')
            return
        self.last_time = current_time

        # Handle timeout for cmd_vel
        cmd_timeout = 0.5  # Timeout in seconds
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > cmd_timeout:
            # Commands have stopped; assume zero velocities
            self.u_v = 0.0
            self.u_w = 0.0
            rospy.loginfo("cmd_vel timeout. Setting control inputs to zero.")

        # State prediction (time update)
        self.predict_state(delta_t)
        rospy.loginfo("State predicted.")

        # Covariance prediction
        self.predict_covariance(delta_t)
        rospy.loginfo("Covariance predicted.")

        # Measurement update (correction)
        self.update_state(imu_msg, joint_states_msg)
        rospy.loginfo("State updated.")

        # Publish the updated odometry message
        self.publish_odom(imu_msg.header.stamp)
        rospy.loginfo("Odometry published.")

    def predict_state(self, delta_t):
        # Extract current state variables
        x, y, theta, v, omega = self.state

        # State prediction using the motion model
        x_pred = x + v * delta_t * np.cos(theta)
        y_pred = y + v * delta_t * np.sin(theta)
        theta_pred = theta + omega * delta_t
        v_pred = self.a_v * v + self.G_v * (1 - self.a_v) * self.u_v
        omega_pred = self.a_w * omega + self.G_w * (1 - self.a_w) * self.u_w

        # Update the state prediction
        self.state = np.array([x_pred, y_pred, theta_pred, v_pred, omega_pred], dtype=np.float64)

    def predict_covariance(self, delta_t):
        # Extract current state variables
        _, _, theta, v, _ = self.state

        # Jacobian of the motion model with respect to the state (A matrix)
        A = np.array([
            [1, 0, -v * delta_t * np.sin(theta), delta_t * np.cos(theta), 0],
            [0, 1,  v * delta_t * np.cos(theta), delta_t * np.sin(theta), 0],
            [0, 0, 1, 0, delta_t],
            [0, 0, 0, self.a_v, 0],
            [0, 0, 0, 0, self.a_w]
        ], dtype=np.float64)

        # Update the covariance prediction
        self.P = A @ self.P @ A.T + self.Q

    def update_state(self, imu_msg, joint_states_msg):
        # Measurement vector z
        z = self.get_measurement_vector(imu_msg, joint_states_msg)
        if z is None:
            rospy.logwarn('Measurement vector z is None. Skipping update.')
            return

        # Measurement matrix C
        C = np.array([
            [0, 0, 0, 1 / self.r_w,  self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w,  self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.robot_radius / self.r_w],
            [0, 0, 0, 0,            1.0]
        ], dtype=np.float64)

        # Predicted measurement
        z_pred = C @ self.state

        # Innovation (measurement residual)
        y = z - z_pred

        # Innovation covariance
        S = C @ self.P @ C.T + self.R

        # Check for singularity
        if np.linalg.cond(S) > 1e12:
            rospy.logwarn('Innovation covariance matrix S is ill-conditioned.')
            return

        # Kalman gain
        K = self.P @ C.T @ np.linalg.inv(S)

        rospy.loginfo(f"Kalman gain K: {K}")

        # State update
        self.state = self.state + K @ y

        # Covariance update using Joseph form
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ self.R @ K.T

    def get_measurement_vector(self, imu_msg, joint_states_msg):
        # Extract gyroscope measurement (angular velocity around z-axis)
        omega_g = imu_msg.angular_velocity.z - self.gyro_bias
        rospy.loginfo(f"Gyro measurement omega_g: {omega_g}")

        # Extract wheel encoder measurements
        if len(joint_states_msg.velocity) > 0:
            joint_velocities = dict(zip(joint_states_msg.name, joint_states_msg.velocity))
        else:
            rospy.logwarn('Joint velocities not available. Skipping measurement update.')
            return None

        # Adjust the joint names according to your robot's URDF
        omega_fl = joint_velocities.get('front_left_wheel', 0.0)
        omega_fr = joint_velocities.get('front_right_wheel', 0.0)
        omega_rl = joint_velocities.get('rear_left_wheel', 0.0)
        omega_rr = joint_velocities.get('rear_right_wheel', 0.0)

        rospy.loginfo(f"Wheel velocities omega_fl: {omega_fl}, omega_fr: {omega_fr}, omega_rl: {omega_rl}, omega_rr: {omega_rr}")

        # Return the measurement vector
        return np.array([omega_fl, omega_fr, omega_rl, omega_rr, omega_g], dtype=np.float64)

    def publish_odom(self, timestamp):
        # Create the odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp  # Use the sensor timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set the position
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = 0.0

        # Set the orientation (convert yaw angle to quaternion)
        quaternion = quaternion_from_euler(0, 0, self.state[2])
        odom_msg.pose.pose.orientation = Quaternion(*quaternion)

        # Set the velocities
        odom_msg.twist.twist.linear.x = self.state[3]
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = self.state[4]

        # Set the pose covariance (6x6 matrix)
        odom_msg.pose.covariance = [0] * 36
        odom_msg.pose.covariance[0] = self.P[0, 0]  # x-x
        odom_msg.pose.covariance[1] = self.P[0, 1]  # x-y
        odom_msg.pose.covariance[5] = self.P[0, 2]  # x-theta
        odom_msg.pose.covariance[6] = self.P[1, 0]  # y-x
        odom_msg.pose.covariance[7] = self.P[1, 1]  # y-y
        odom_msg.pose.covariance[11] = self.P[1, 2]  # y-theta
        odom_msg.pose.covariance[30] = self.P[2, 0]  # theta-x
        odom_msg.pose.covariance[31] = self.P[2, 1]  # theta-y
        odom_msg.pose.covariance[35] = self.P[2, 2]  # theta-theta

        # Set the twist covariance (6x6 matrix)
        odom_msg.twist.covariance = [0] * 36
        odom_msg.twist.covariance[0] = self.P[3, 3]  # v-v
        odom_msg.twist.covariance[35] = self.P[4, 4]  # omega-omega

        # Publish the odometry message
        self.ekf_odom_pub.publish(odom_msg)

def main():
    ekf_node = EKFOdometryNode()
    rospy.spin()
    rospy.loginfo('Shutting down ekf_odometry_node')

if __name__ == '__main__':
    main()

