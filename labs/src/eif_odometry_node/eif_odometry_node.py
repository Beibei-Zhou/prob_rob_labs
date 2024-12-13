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

        # Robot parameters
        self.r_w = 0.098
        self.track = 0.37559
        self.robot_radius = self.track / 2.0

        # Dynamics parameters
        self.a_v = 0.383
        self.G_v = 1.0
        self.a_w = 0.123
        self.G_w = 1.3

        # State: [x, y, theta, v, omega]
        # Initialize with zero state
        self.state = np.zeros(5, dtype=np.float64)

        # Initial covariance P and convert to information form
        self.P = np.eye(5) * 0.1
        self.Lambda = np.linalg.inv(self.P)
        self.eta = self.Lambda @ self.state

        # Process noise Q (5x5)
        self.Q = np.diag([0.000, 0.000, 0.000, 0.005, 0.005])

        # Measurement noise R (5x5)
        encoder_var = (0.05)**2
        gyro_var = (0.02)**2
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
        self.cov_x_pub = rospy.Publisher('/eif_covariance/x', Float64, queue_size=10)
        self.cov_y_pub = rospy.Publisher('/eif_covariance/y', Float64, queue_size=10)
        self.cov_theta_pub = rospy.Publisher('/eif_covariance/theta', Float64, queue_size=10)
        self.cov_v_pub = rospy.Publisher('/eif_covariance/v', Float64, queue_size=10)
        self.cov_omega_pub = rospy.Publisher('/eif_covariance/omega', Float64, queue_size=10)

        # Subscribers
        imu_sub = Subscriber('/imu/data', Imu, queue_size=10)
        joint_states_sub = Subscriber('/joint_states', JointState, queue_size=10)

        self.ats = ApproximateTimeSynchronizer([imu_sub, joint_states_sub], queue_size=10, slop=0.1)
        self.ats.registerCallback(self.eif_callback)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo('EIF Odometry Node Initialized')

    def cmd_vel_callback(self, msg):
        self.u_v = msg.linear.x
        self.u_w = msg.angular.z
        self.last_cmd_time = rospy.Time.now()

    def eif_callback(self, imu_msg, joint_states_msg):
        # Bias estimation phase
        if not self.bias_estimation_complete:
            omega_g_raw = imu_msg.angular_velocity.z
            self.gyro_bias_samples.append(omega_g_raw)
            if len(self.gyro_bias_samples) >= self.bias_estimation_samples:
                self.gyro_bias = np.mean(self.gyro_bias_samples)
                self.bias_estimation_complete = True
                rospy.loginfo(f"Gyro bias estimated: {self.gyro_bias}")
                self.gyro_bias_samples = []
            return

        # Time handling
        current_time = imu_msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            rospy.loginfo('First callback execution. Setting last_time and returning.')
            return
        delta_t = current_time - self.last_time
        if delta_t <= 0:
            rospy.logwarn('Non-positive delta_t. Skipping this update.')
            return
        self.last_time = current_time

        # cmd_vel timeout
        cmd_timeout = 0.5
        if (rospy.Time.now() - self.last_cmd_time).to_sec() > cmd_timeout:
            if self.u_v != 0.0 or self.u_w != 0.0:
                self.u_v = 0.0
                self.u_w = 0.0
                rospy.loginfo("cmd_vel timeout. Setting control inputs to zero.")

        # Compute prediction using the new function
        x_pred, A = self.compute_A(self.eta, self.Lambda, self.u_v, self.u_w, delta_t)

        # Convert back to information form
        Lambda_pred = np.linalg.inv(A @ (np.linalg.inv(self.Lambda)) @ A.T + self.Q)
        eta_pred = Lambda_pred @ x_pred

        # Update eta and Lambda
        self.eta = eta_pred
        self.Lambda = Lambda_pred

        # --- EIF Update Step ---
        z = self.get_measurement_vector(imu_msg, joint_states_msg)
        if z is None:
            rospy.logwarn('Measurement vector z is None. Skipping update.')
            return

        C = self.get_measurement_matrix()
        R_inv = np.linalg.inv(self.R)

        # Information update
        self.Lambda = self.Lambda + C.T @ R_inv @ C
        self.eta = self.eta + C.T @ R_inv @ z

        # Convert to moment form to publish
        x_est, P_est = self.canonical_to_moment(self.eta, self.Lambda)
        rospy.loginfo(f"IF Updated state: {x_est}")

        self.publish_odom(imu_msg.header.stamp, x_est, P_est)
        self.cov_x_pub.publish(P_est[0, 0])
        self.cov_y_pub.publish(P_est[1, 1])
        self.cov_theta_pub.publish(P_est[2, 2])
        self.cov_v_pub.publish(P_est[3, 3])
        self.cov_omega_pub.publish(P_est[4, 4])

    def compute_A(self, eta, Lambda, u_v, u_w, delta_t):

        # Convert information form to moment form
        # x, P = self.canonical_to_moment(eta, Lambda)
        P = np.linalg.inv(Lambda)
        x = P @ eta
        x = x.flatten()  # Ensure x is 1D
        # Predict the next state using the motion model
        X, Y, TH, V, W = x
        X_pred = X + V * delta_t * np.cos(TH)
        Y_pred = Y + V * delta_t * np.sin(TH)
        TH_pred = TH + W * delta_t
        V_pred = self.a_v * V + self.G_v * (1 - self.a_v) * u_v
        W_pred = self.a_w * W + self.G_w * (1 - self.a_w) * u_w
        x_pred = np.array([X_pred, Y_pred, TH_pred, V_pred, W_pred])        
        _, _, TH, V, _ = x_pred
        A = np.array([
            [1, 0, -V * delta_t * np.sin(TH), delta_t * np.cos(TH), 0],
            [0, 1,  V * delta_t * np.cos(TH), delta_t * np.sin(TH), 0],
            [0, 0, 1, 0, delta_t],
            [0, 0, 0, self.a_v, 0],
            [0, 0, 0, 0, self.a_w]
        ], dtype=np.float64)
        
        return x_pred, A

    def canonical_to_moment(self, eta, Lambda):
        try:
            P = np.linalg.inv(Lambda)
            x = P @ eta
            x = x.flatten()  
        except np.linalg.LinAlgError:
            rospy.logwarn('Lambda is singular during canonical_to_moment.')
            P = np.eye(5)
            x = np.zeros(5)
        return x, P

    def get_measurement_vector(self, imu_msg, joint_states_msg):

        omega_g = imu_msg.angular_velocity.z - self.gyro_bias
        if len(joint_states_msg.velocity) == 0:
            return None
        joint_velocities = dict(zip(joint_states_msg.name, joint_states_msg.velocity))
        omega_fl = joint_velocities.get('front_left_wheel', 0.0)
        omega_fr = joint_velocities.get('front_right_wheel', 0.0)
        omega_rl = joint_velocities.get('rear_left_wheel', 0.0)
        omega_rr = joint_velocities.get('rear_right_wheel', 0.0)

        return np.array([omega_fl, omega_fr, omega_rl, omega_rr, omega_g], dtype=np.float64)

    def get_measurement_matrix(self):

        C = np.array([
            [0, 0, 0, 1/self.r_w, -self.robot_radius/self.r_w],
            [0, 0, 0, 1/self.r_w,  self.robot_radius/self.r_w],
            [0, 0, 0, 1/self.r_w, -self.robot_radius/self.r_w],
            [0, 0, 0, 1/self.r_w,  self.robot_radius/self.r_w],
            [0, 0, 0, 0, 1.0]
        ], dtype=np.float64)
        return C

    def publish_odom(self, timestamp, x, P):

        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = x[0]
        odom_msg.pose.pose.position.y = x[1]
        odom_msg.pose.pose.position.z = 0.0

        quaternion = quaternion_from_euler(0, 0, x[2])
        odom_msg.pose.pose.orientation = Quaternion(*quaternion)

        odom_msg.twist.twist.linear.x = x[3]
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = x[4]

        # Map from P (5x5) to Odometry covariance (6x6)
        pose_cov = np.zeros((6,6))
        pose_cov[0,0] = P[0,0]
        pose_cov[0,1] = P[0,1]
        pose_cov[1,0] = P[1,0]
        pose_cov[1,1] = P[1,1]
        pose_cov[5,5] = P[2,2]

        twist_cov = np.zeros((6,6))
        twist_cov[0,0] = P[3,3]
        twist_cov[5,5] = P[4,4]

        odom_msg.pose.covariance = pose_cov.reshape(36).tolist()
        odom_msg.twist.covariance = twist_cov.reshape(36).tolist()

        self.eif_odom_pub.publish(odom_msg)

def main():
    node = EIFOdometryNode()
    rospy.spin()
    rospy.loginfo('Shutting down eif_odometry_node')

if __name__ == '__main__':
    main()
