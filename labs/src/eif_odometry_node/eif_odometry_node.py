#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu, JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import Quaternion, Twist
from tf.transformations import quaternion_from_euler


class EIFOdometryNode:
    def __init__(self):
        rospy.init_node('eif_odometry_node')
        rospy.loginfo('Starting eif_odometry_node')

        # Robot Parameters from the URDF
        self.r_w = 0.098       # Wheel radius
        self.track = 0.37559   # Track width
        self.robot_radius = self.track / 2.0

        # Dynamics Model Parameters
        self.a_v = 0.383
        self.G_v = 1.0
        self.a_w = 0.123
        self.G_w = 1.3

        # Jacobian A,B matrix
        self.A = np.zeros((5, 5), dtype=np.float64)
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0.617, 0],
            [0, 1.1401]
        ])

        # Initialize State Vector :[x, y, theta, v, omega]
        self.state = np.zeros(5, dtype=np.float64)

        # Initialize covariance matrix P (5x5)
        self.P = np.eye(5, dtype=np.float64) * 0.1

        # Convert the state vector and covariance into information form
        self.Lambda = np.linalg.pinv(self.P)
        self.eta = self.Lambda.dot(self.state)

        # Process noise covariance Q (5x5)
        self.Q = np.diag([0.005, 0.005])

        # Measurement uncertainty R (5x5)
        encoder_var = (0.05) ** 2
        gyro_var = (0.02) ** 2
        self.R = np.diag([encoder_var, encoder_var, encoder_var, encoder_var, gyro_var])

        self.u_v = 0.0
        self.u_w = 0.0
        self.last_cmd_time = rospy.Time.now()

        self.gyro_bias_samples = []
        self.bias_estimation_samples = 100
        self.bias_estimation_complete = False
        self.gyro_bias = 0.0

        self.last_time = None

        # Publishers
        self.eif_odom_pub = rospy.Publisher('/eif_odom', Odometry, queue_size=10)

        # Uncertainties
        self.cov_x_pub = rospy.Publisher('/eif_covariance/x', Float64, queue_size=10)
        self.cov_y_pub = rospy.Publisher('/eif_covariance/y', Float64, queue_size=10)
        self.cov_theta_pub = rospy.Publisher('/eif_covariance/theta', Float64, queue_size=10)
        self.cov_v_pub = rospy.Publisher('/eif_covariance/v', Float64, queue_size=10)
        self.cov_omega_pub = rospy.Publisher('/eif_covariance/omega', Float64, queue_size=10)

        # Subscribers
        imu_sub = Subscriber('/imu/data', Imu, queue_size=10)
        joint_states_sub = Subscriber('/joint_states', JointState, queue_size=10)

        # Create multiple subscribers and get a single callback
        self.ats = ApproximateTimeSynchronizer(
            [imu_sub, joint_states_sub],
            queue_size=10,
            slop=0.1
        )

        self.ats.registerCallback(self.eif_callback)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo('EIF Odometry Node Initialized')

    def cmd_vel_callback(self, msg):
        self.u_v = msg.linear.x
        self.u_w = msg.angular.z
        self.last_cmd_time = rospy.Time.now()

    def eif_callback(self, imu_msg, joint_states_msg):
        if not self.bias_estimation_complete:
            omega_g_raw = imu_msg.angular_velocity.z
            self.gyro_bias_samples.append(omega_g_raw)
            if len(self.gyro_bias_samples) >= self.bias_estimation_samples:
                self.gyro_bias = np.mean(self.gyro_bias_samples)
                self.bias_estimation_complete = True
                rospy.loginfo(f"Gyro bias estimated: {self.gyro_bias}")
                self.gyro_bias_samples = []
            return

        # Handle timeout for cmd_vel
        current_time = imu_msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            rospy.loginfo('First callback execution. Setting last_time and returning.')
            return
        delta_t = current_time - self.last_time
        if delta_t <= 0:
            rospy.logwarn('Delta time is non-positive. Skipping this update.')
            return
        self.last_time = current_time

        cmd_timeout = 0.5
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > cmd_timeout:
            if self.u_v != 0.0 or self.u_w != 0.0:
                self.u_v = 0.0
                self.u_w = 0.0
                rospy.loginfo("cmd_vel timeout. Setting control inputs to zero.")


        # Prediction Step
        self.A = self.compute_A(delta_t)
        A_LambdaT_AT = self.A @ np.linalg.pinv(self.Lambda) @ self.A.T
        
        u = np.array([self.u_v, self.u_w])  
        Bu = self.B @ u  

        A_LambdaT_eta = self.A @ np.linalg.pinv(self.Lambda) @ self.eta
        self.eta = self.Lambda @ (A_LambdaT_eta + Bu)
        self.Lambda = np.linalg.pinv(A_LambdaT_AT + self.B @ self.Q @ (self.B.T))

        # self.state, self.P = self.canonical_to_moment(self.eta, self.Lambda)
        # self.state, self.P = self.predict(delta_t)
        # self.eta, self.Lambda = self.moment_to_canonical(self.state, self.P)

        # Update Step (Innovation Step)
        z = self.get_measurement_vector(imu_msg, joint_states_msg)
        if z is None:
            rospy.logwarn('Measurement vector z is None. Skipping update.')
            return

        C = self.get_measurement_matrix()
        R_inv = np.linalg.pinv(self.R)

        self.Lambda = self.Lambda + C.T @ R_inv @ C
        self.eta = self.eta + C.T @ R_inv @ z

        # Publish the results
        x_est, P_est = self.canonical_to_moment(self.eta, self.Lambda)
        rospy.loginfo(f"IF Updated state: {x_est}")

        self.publish_odom(imu_msg.header.stamp, x_est, P_est)

        # Publish covariance elements
        self.cov_x_pub.publish(P_est[0, 0])
        self.cov_y_pub.publish(P_est[1, 1])
        self.cov_theta_pub.publish(P_est[2, 2])
        self.cov_v_pub.publish(P_est[3, 3])
        self.cov_omega_pub.publish(P_est[4, 4])

    # def moment_to_canonical(self, x, P):
    #     Lambda = np.linalg.pinv(P)
    #     eta = Lambda.dot(x)
    #     return eta, Lambda

    def canonical_to_moment(self, eta, Lambda):
        P = np.linalg.pinv(Lambda)
        x = P.dot(eta).flatten()
        return x, P

    def compute_A(self, delta_t):
        P = np.linalg.pinv(self.Lambda)
        state = P.dot(self.eta)
        x, y, theta, v, omega = state

        # State prediction using the motion model
        x_pred = x + v * delta_t * np.cos(theta)
        y_pred = y + v * delta_t * np.sin(theta)
        theta_pred = theta + omega * delta_t
        v_pred = self.a_v * v + self.G_v * (1 - self.a_v) * self.u_v
        omega_pred = self.a_w * omega + self.G_w * (1 - self.a_w) * self.u_w

        self.state = np.array([x_pred, y_pred, theta_pred, v_pred, omega_pred])

        # Jacobian of the motion model
        A = np.array([
            [1, 0, -v * delta_t * np.sin(theta), delta_t * np.cos(theta), 0],
            [0, 1,  v * delta_t * np.cos(theta), delta_t * np.sin(theta), 0],
            [0, 0, 1, 0, delta_t],
            [0, 0, 0, self.a_v, 0],
            [0, 0, 0, 0, self.a_w]
        ], dtype=np.float64)

        return A
    
    # def predict(self, delta_t):
    #     # Extract the current state variables
    #     x, y, theta, v, omega = self.state

    #     # State prediction using the motion model
    #     x_pred = x + v * delta_t * np.cos(theta)
    #     y_pred = y + v * delta_t * np.sin(theta)
    #     theta_pred = theta + omega * delta_t
    #     v_pred = self.a_v * v + self.G_v * (1 - self.a_v) * self.u_v
    #     omega_pred = self.a_w * omega + self.G_w * (1 - self.a_w) * self.u_w

    #     self.state = np.array([x_pred, y_pred, theta_pred, v_pred, omega_pred])

    #     # Jacobian of the motion model
    #     A = np.array([
    #         [1, 0, -v * delta_t * np.sin(theta), delta_t * np.cos(theta), 0],
    #         [0, 1,  v * delta_t * np.cos(theta), delta_t * np.sin(theta), 0],
    #         [0, 0, 1, 0, delta_t],
    #         [0, 0, 0, self.a_v, 0],
    #         [0, 0, 0, 0, self.a_w]
    #     ], dtype=np.float64)

    #     self.P = A @ self.P @ A.T + self.Q

    #     return self.state, self.P

    def get_measurement_vector(self, imu_msg, joint_states_msg):
        omega_g = imu_msg.angular_velocity.z - self.gyro_bias

        if len(joint_states_msg.velocity) > 0:
            joint_velocities = dict(zip(joint_states_msg.name, joint_states_msg.velocity))
        else:
            rospy.logwarn('Joint velocities not available. Skipping measurement update.')
            return None

        omega_fl = joint_velocities.get('front_left_wheel', 0.0)
        omega_fr = joint_velocities.get('front_right_wheel', 0.0)
        omega_rl = joint_velocities.get('rear_left_wheel', 0.0)
        omega_rr = joint_velocities.get('rear_right_wheel', 0.0)

        #rospy.loginfo(f"Wheel velocities omega_fl: {omega_fl}, omega_fr: {omega_fr}, omega_rl: {omega_rl}, omega_rr: {omega_rr}")

        return np.array([omega_fl, omega_fr, omega_rl, omega_rr, omega_g], dtype=np.float64)

    def get_measurement_matrix(self):
        # Measurement matrix C aligned with z = [omega_fl, omega_fr, omega_rl, omega_rr, omega_g]
        C = np.array([
            [0, 0, 0, 1 / self.r_w, -self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w,  self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.robot_radius / self.r_w],
            [0, 0, 0, 1 / self.r_w,  self.robot_radius / self.r_w],
            [0, 0, 0, 0, 1.0]
        ], dtype=np.float64)

        return C

    def publish_odom(self, timestamp, x, P):
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Pose
        odom_msg.pose.pose.position.x = x[0]
        odom_msg.pose.pose.position.y = x[1]
        odom_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        quaternion = quaternion_from_euler(0, 0, x[2])
        odom_msg.pose.pose.orientation = Quaternion(*quaternion)

        # Twist
        odom_msg.twist.twist.linear.x = x[3]
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = x[4]

        # Pose covariance (map from 5x5 to 6x6)
        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = P[0, 0]
        pose_cov[0, 1] = P[0, 1]
        pose_cov[1, 0] = P[1, 0] 
        pose_cov[1, 1] = P[1, 1] 
        # yaw corresponds to index [5,5]
        pose_cov[5, 5] = P[2, 2]

        # Twist covariance
        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = P[3, 3]
        twist_cov[5, 5] = P[4, 4]

        odom_msg.pose.covariance = pose_cov.reshape(36).tolist()
        odom_msg.twist.covariance = twist_cov.reshape(36).tolist()

        self.eif_odom_pub.publish(odom_msg)

def main():
    node = EIFOdometryNode()
    rospy.spin()
    rospy.loginfo('Shutting down eif_odometry_node')


if __name__ == '__main__':
    main()
