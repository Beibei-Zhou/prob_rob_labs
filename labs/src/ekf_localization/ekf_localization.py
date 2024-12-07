#!/usr/bin/env python

import rospy
import numpy as np
import tf
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from opencv_apps.msg import Point2DArrayStamped
import sympy as sp
from sympy import sin, cos, atan2, simplify

class MeasurementModel:
    def __init__(self, xl, yl, hl, rl, color,
                 tcx=0.229, tcy=0.0, tcz=0.2165,
                 fx=788.408537, fy=788.408537, cx=512.5, cy=384.5,
                 R_diag=None):
        # Store parameters
        self.x_l = xl
        self.y_l = yl
        self.h_l = hl
        self.r_l = rl
        self.color = color
        self.t_cx = tcx
        self.t_cy = tcy
        self.t_cz = tcz
        self.f_x = fx
        self.f_y = fy
        self.c_x = cx
        self.c_y = cy


        self.R = np.diag(R_diag)

        # Declare symbols
        x_r, y_r, theta_r = sp.symbols('x_r y_r theta_r')
        t_cx, t_cy, t_cz = sp.symbols('t_cx t_cy t_cz')
        x_l, y_l = sp.symbols('x_l y_l')
        h_l, r_l = sp.symbols('h_l r_l')
        f_x, f_y, c_x, c_y = sp.symbols('f_x f_y c_x c_y')

        # Compute camera position in global frame
        x_c = x_r + cos(theta_r)*t_cx - sin(theta_r)*t_cy
        y_c = y_r + sin(theta_r)*t_cx + cos(theta_r)*t_cy
        z_c = t_cz

        # Compute bearing angle alpha
        delta_x = x_l - x_c
        delta_y = y_l - y_c
        alpha = atan2(delta_y, delta_x)

        # Compute feature points in global frame (four corners of the landmark)
        x1 = x_l - r_l * sin(alpha)
        y1 = y_l + r_l * cos(alpha)
        x2 = x_l + r_l * sin(alpha)
        y2 = y_l - r_l * cos(alpha)

        p_1g = sp.Matrix([x1, y1, 0, 1]) 
        p_2g = sp.Matrix([x2, y2, 0, 1]) 
        p_3g = sp.Matrix([x2, y2, h_l, 1]) 
        p_4g = sp.Matrix([x1, y1, h_l, 1]) 

        # Transformation matrices
        T_mr = sp.Matrix([
            [cos(theta_r), -sin(theta_r), 0, x_r],
            [sin(theta_r),  cos(theta_r), 0, y_r],
            [0,             0,            1, 0],
            [0,             0,            0, 1]
        ])

        T_ro = sp.Matrix([
            [0, 0, 1, tcx],
            [-1, 0, 0, tcy],
            [0, -1, 0, tcz],
            [0, 0, 0, 1]
        ])

        # Compute the transformation from the map frame to the optical frame
        T_mo = simplify(T_mr * T_ro)
        T_om = simplify(T_mo.inv())

        # Function to compute camera frame position of a point
        def compute_p_c(p_ig):
            p_io = T_om * p_ig
            return p_io

        p_1o = compute_p_c(p_1g)
        p_2o = compute_p_c(p_2g)
        p_3o = compute_p_c(p_3g)
        p_4o = compute_p_c(p_4g)

        # Function to project camera frame point to pixel coordinates
        def project_to_pixel(p_o):
            u_i = f_x * p_o[0] + c_x * p_o[2]
            v_i = f_y * p_o[1] + c_y * p_o[2]
            w_i = p_o[2]
            u_p = u_i / w_i
            v_p = v_i / w_i
            return sp.Matrix([u_p, v_p])

        p_1p = project_to_pixel(p_1o)
        p_2p = project_to_pixel(p_2o)
        p_3p = project_to_pixel(p_3o)
        p_4p = project_to_pixel(p_4o)

        # Measurement vector
        z_expected = sp.Matrix([
            p_1p[0], p_1p[1],
            p_2p[0], p_2p[1],
            p_3p[0], p_3p[1],
            p_4p[0], p_4p[1]
        ])

        # Compute Jacobian
        state_vars = [x_r, y_r, theta_r]
        H = z_expected.jacobian(state_vars)

        # Create callable functions
        variables = (x_r, y_r, theta_r,
                     x_l, y_l, h_l, r_l,
                     t_cx, t_cy, t_cz,
                     f_x, f_y, c_x, c_y)

        self.measurement_function = sp.lambdify(
            variables,
            z_expected,
            'numpy'
        )

        self.jacobian_function = sp.lambdify(
            variables,
            H,
            'numpy'
        )

    def jacobian(self, x_r, y_r, theta_r):
        # Calculate the numeric Jacobian of the measurement model with respect to the robot
        args = (x_r, y_r, theta_r,
                self.x_l, self.y_l, self.h_l, self.r_l,
                self.t_cx, self.t_cy, self.t_cz,
                self.f_x, self.f_y, self.c_x, self.c_y)
        
        return self.jacobian_function(*args)
    
    def measurement(self, x_r, y_r, theta_r, observed_features):
        args = (x_r, y_r, theta_r,
                self.x_l, self.y_l, self.h_l, self.r_l,
                self.t_cx, self.t_cy, self.t_cz,
                self.f_x, self.f_y, self.c_x, self.c_y)
        
        z_pred = self.measurement_function(*args)
        
        # Ensure z_pred is a Numpy array 8*1
        z_pred = np.array(z_pred, dtype=np.float64).flatten()

        z_obs = np.copy(z_pred)
        # Construct obeseved measurement vector
        num_points = min(len(observed_features), 4)
        for i in range(num_points):
            u_meas, v_meas = observed_features[i]
            z_obs[2*i] = u_meas
            z_obs[2*i + 1] = v_meas

        diff = z_obs - z_pred

        return z_pred, z_obs, self.R, diff

class EKFLocalizationNode:
    def __init__(self):
        rospy.init_node('ekf_localization')
        rospy.loginfo("EKF Localization Node Started")

        # Motion noise parameters
        self.alpha1 = rospy.get_param('~alpha1', 0.018)
        self.alpha2 = rospy.get_param('~alpha2', 0.018)
        self.alpha3 = rospy.get_param('~alpha3', 0.018)
        self.alpha4 = rospy.get_param('~alpha4', 0.018)
        self.alpha = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        # Initial state and covariance
        self.mu = np.array([-2.998, 0.0, 0.0]) 
        self.Sigma = np.diag([0.5, 0.5, 0.5]) 

        # Landmarks as per the provided world.sdf
        self.landmarks = [
            {'xl': 8.5, 'yl': -5.0, 'hl': 0.25, 'rl': 0.1, 'color': 'red'},
            {'xl': 8.5, 'yl': 5.0,  'hl': 0.25, 'rl': 0.1, 'color': 'green'},
            {'xl': -11.5, 'yl': 5.0, 'hl': 0.25, 'rl': 0.1, 'color': 'yellow'},
            {'xl': -11.5, 'yl': -5.0,'hl': 0.25, 'rl': 0.1, 'color': 'magenta'},
            {'xl': 0.0,  'yl': 0.0,  'hl': 0.25, 'rl': 0.1, 'color': 'cyan'}
        ]
        #rospy.loginfo("Defined %d landmarks directly in the code.", len(self.landmarks))

        # Initialize MeasurementModels for each landmark
        self.measurement_models = {}
        for lm in self.landmarks:
            color = lm['color']
            xl = lm['xl']
            yl = lm['yl']
            hl = lm['hl']
            rl = lm['rl']

            R_diag = [51.943872, 2.257994, 56.141963, 16.346652,
                        50.062509, 2.562840, 49.591893, 2.556726]
            
            mm = MeasurementModel(
                xl=xl, yl=yl, hl=hl, rl=rl, color=color,
                tcx=0.229, tcy=0.0, tcz=0.2165,
                fx=788.408537, fy=788.408537, cx=512.5, cy=384.5,
                R_diag=R_diag
            )

            self.measurement_models[color] = mm
            rospy.loginfo("Initialized MeasurementModel for landmark color: %s", color)

        # Subscribes
        self.camera_info_sub = rospy.Subscriber('/front/left/camera_info', CameraInfo, self.camera_info_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        # Subscribe to Measurement topics for each landmark color
        self.measurement_subs = []
        for color in self.measurement_models.keys():
            topic_name = f'/goodfeature_{color}/corners'
            sub = rospy.Subscriber(topic_name, Point2DArrayStamped, self.measurement_callback, callback_args=color)
            self.measurement_subs.append(sub)
            rospy.loginfo("Subscribed to measurement topic: %s", topic_name)

        # Publisher for EKF pose
        self.pose_pub = rospy.Publisher('/ekf_pose', PoseWithCovarianceStamped, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Flag to check
        self.camera_intrinsics_updated = False

        rospy.loginfo("EKF Localization Node Initialized")

    def camera_info_callback(self, msg):
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        # fx = 788.408537
        # fy = 788.408537
        # cx = 512.5
        # cy = 384.5

        #rospy.loginfo("Camera intrinsics received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx, fy, cx, cy)
        for mm in self.measurement_models.values():
            mm.f_x = fx
            mm.f_y = fy
            mm.c_x = cx
            mm.c_y = cy

        # Unregister after camera intrinsics updated in all measurement models 
        # self.camera_info_sub.unregister()
        # self.camera_intrinsics_updated = True

    def odom_callback(self, msg):

        # Extract control inputs
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        # Time step
        current_time = msg.header.stamp.to_sec()
        if not hasattr(self, 'last_odom_time'):
            self.last_odom_time = current_time
            return
        delta_t = current_time - self.last_odom_time
        self.last_odom_time = current_time

        #rospy.loginfo("Odometry Callback: v=%.2f, omega=%.2f, delta_t=%.4f", v, omega, delta_t)
        # Control input
        u_t = [v, omega]
        self.mu, self.Sigma = self.ekf_predict(self.mu, self.Sigma, u_t, delta_t, self.alpha)
        #rospy.loginfo("EKF Prediction Step Completed: mu=[%.2f, %.2f, %.2f]", self.mu[0], self.mu[1], self.mu[2])

        # Publish the predicted pose
        self.publish_pose()

    def measurement_callback(self, msg, color):
        mm = self.measurement_models[color]

        # Extract observed features
        observed_features = [(point.x, point.y) for point in msg.points]
        #rospy.loginfo("Measurement Callback: Received %d features for color: %s", len(observed_features), color)

        # Perform measurement update
        z_pred, z_obs, R, innovation = mm.measurement(self.mu[0], self.mu[1], self.mu[2], observed_features)
        H = mm.jacobian(self.mu[0], self.mu[1], self.mu[2])
        #rospy.loginfo("Computed innovation for color: %s", color)

        # Compute Kalman Gain
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        # Update state
        self.mu = self.mu + K @ innovation
        #rospy.loginfo("State updated: mu=[%.2f, %.2f, %.2f]", self.mu[0], self.mu[1], self.mu[2])

        # Update covariance
        I = np.eye(len(self.mu))
        self.Sigma = (I - K @ H) @ self.Sigma

        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi

        #rospy.loginfo("EKF Update Step Completed for color: %s", color)
        self.publish_pose()

    def ekf_predict(self, mu_prev, Sigma_prev, u_t, delta_t, alpha):
        v_t, omega_t = u_t
        theta = mu_prev[2]
        G_t = np.array([
            [1, 0, (-v_t / omega_t) * np.cos(theta) + (v_t / omega_t) * np.cos(theta + omega_t * delta_t)],
            [0, 1, (-v_t / omega_t) * np.sin(theta) + (v_t / omega_t) * np.sin(theta + omega_t * delta_t)],
            [0, 0, 1]
        ])

        V_t = np.array([
            [(-np.sin(theta) + np.sin(theta + omega_t * delta_t)) / omega_t,
                v_t * (np.cos(theta + omega_t * delta_t) * delta_t) / omega_t + v_t * (np.sin(theta) - np.sin(theta + omega_t * delta_t)) / (omega_t**2)],
            [(np.cos(theta) - np.cos(theta + omega_t * delta_t)) / omega_t,
                v_t * (np.sin(theta + omega_t * delta_t) * delta_t) / omega_t - v_t * (np.cos(theta) - np.cos(theta + omega_t * delta_t)) / (omega_t**2)],
            [0, delta_t]
        ])

        alpha1, alpha2, alpha3, alpha4 = alpha
        M_t = np.array([
            [alpha1 * v_t**2 + alpha2 * omega_t**2, 0],
            [0, alpha3 * v_t**2 + alpha4 * omega_t**2]
        ])

        mu_t = mu_prev + np.array([
            (-v_t / omega_t) * np.sin(theta) + (v_t / omega_t) * np.sin(theta + omega_t * delta_t),
            (v_t / omega_t) * np.cos(theta) - (v_t / omega_t) * np.cos(theta + omega_t * delta_t),
            omega_t * delta_t
        ])

        Sigma_t = G_t @ Sigma_prev @ G_t.T + V_t @ M_t @ V_t.T

        return mu_t, Sigma_t

    def publish_pose(self):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = self.mu[0]
        pose_msg.pose.pose.position.y = self.mu[1]
        pose_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        quat = tf.transformations.quaternion_from_euler(0, 0, self.mu[2])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        covariance_6x6 = np.zeros((6,6))
        covariance_6x6[0,0] = self.Sigma[0,0]
        covariance_6x6[0,1] = self.Sigma[0,1]
        covariance_6x6[1,0] = self.Sigma[1,0]
        covariance_6x6[1,1] = self.Sigma[1,1]
        covariance_6x6[5,5] = self.Sigma[2,2]
        pose_msg.pose.covariance = covariance_6x6.flatten().tolist()
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published EKF Pose: x=%.2f, y=%.2f, theta=%.2f", self.mu[0], self.mu[1], self.mu[2])

def main():
    try:
        ekf_node = EKFLocalizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
