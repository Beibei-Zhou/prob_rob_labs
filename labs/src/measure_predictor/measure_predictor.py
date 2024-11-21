#!/usr/bin/env python

# measure_predictor.py

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from opencv_apps.msg import Point2DArrayStamped, Point2D
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf
import numpy as np
from std_msgs.msg import Header
import sympy as sp
from sympy import sin, cos, atan2, simplify

class MeasurementModel:
    def __init__(self):
        # Step 1: Declare symbols
        x_r, y_r, theta_r = sp.symbols('x_r y_r theta_r')
        t_cx, t_cy, t_cz = sp.symbols('t_cx t_cy t_cz')
        x_l, y_l = sp.symbols('x_l y_l')
        h_l, r_l = sp.symbols('h_l r_l')
        f_x, f_y, c_x, c_y = sp.symbols('f_x f_y c_x c_y')

        # Store symbols for later use
        self.symbols = [x_r, y_r, theta_r,
                        x_l, y_l, h_l, r_l,
                        t_cx, t_cy, t_cz,
                        f_x, f_y, c_x, c_y]

        # Step 2: Compute camera position in global frame
        x_c = x_r + cos(theta_r) * t_cx - sin(theta_r) * t_cy
        y_c = y_r + sin(theta_r) * t_cx + cos(theta_r) * t_cy
        z_c = t_cz

        # Step 3: Compute bearing angle alpha
        delta_x = x_l - x_c
        delta_y = y_l - y_c
        alpha = atan2(delta_y, delta_x)

        # Step 4: Compute feature points in global frame
        x1 = x_l - r_l * sin(alpha)
        y1 = y_l + r_l * cos(alpha)
        x2 = x_l + r_l * sin(alpha)
        y2 = y_l - r_l * cos(alpha)

        p_1g = sp.Matrix([x1, y1, 0, 1])      # Bottom-left corner
        p_2g = sp.Matrix([x2, y2, 0, 1])      # Bottom-right corner
        p_3g = sp.Matrix([x2, y2, h_l, 1])    # Top-right corner
        p_4g = sp.Matrix([x1, y1, h_l, 1])    # Top-left corner

        # Step 5: Transformation matrices
        T_mr = sp.Matrix([
            [cos(theta_r), -sin(theta_r), 0, x_r],
            [sin(theta_r),  cos(theta_r), 0, y_r],
            [0,             0,            1, 0],
            [0,             0,            0, 1]
        ])

        T_ro = sp.Matrix([
            [0, 0, 1, t_cx],
            [-1, 0, 0, t_cy],
            [0, -1, 0, t_cz],
            [0, 0, 0, 1]
        ])

        # Compute the transformation from the map frame to the optical frame
        T_mo = simplify(T_mr * T_ro)
        T_om = simplify(T_mo.inv())

        # Step 6: Compute positions of feature points in camera frame
        def compute_p_c(p_ig):
            p_io = T_om * p_ig
            return p_io

        p_1o = compute_p_c(p_1g)
        p_2o = compute_p_c(p_2g)
        p_3o = compute_p_c(p_3g)
        p_4o = compute_p_c(p_4g)

        # Step 7: Project points to pixel coordinates
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

        # Step 8: Measurement vector
        z_expected = sp.Matrix([
            p_1p[0], p_1p[1],
            p_2p[0], p_2p[1],
            p_3p[0], p_3p[1],
            p_4p[0], p_4p[1]
        ])

        # Step 9: Compute Jacobian
        state_vars = [x_r, y_r, theta_r]
        H = z_expected.jacobian(state_vars)

        # Step 10: Create callable functions
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

    def compute_measurement(self, *args):
        return self.measurement_function(*args)

    def compute_jacobian(self, *args):
        return self.jacobian_function(*args)

class MeasurementPredictor:
    def __init__(self):
        rospy.init_node('measure_predictor')
        rospy.loginfo('Starting measure_predictor')

        # Get parameters
        self.landmark_color = rospy.get_param('~landmark_color', 'cyan')
        self.landmark_x = rospy.get_param('~landmark_x', 0.0)
        self.landmark_y = rospy.get_param('~landmark_y', 0.0)
        self.landmark_height = rospy.get_param('~landmark_height', 0.5)
        self.landmark_radius = rospy.get_param('~landmark_radius', 0.1)

        # Initialize variables
        self.camera_info_received = False
        self.camera_info = None
        self.camera_frame_id = None
        self.t_cx, self.t_cy, self.t_cz = 0.0, 0.0, 0.0
        self.f_x, self.f_y, self.c_x, self.c_y = 0.0, 0.0, 0.0, 0.0

        # Measurement model instance
        self.measurement_model = MeasurementModel()

        # Subscribers and Publishers
        self.camera_info_sub = rospy.Subscriber('/front/left/camera_info', CameraInfo, self.camera_info_callback)
        self.pose_sub = rospy.Subscriber('/jackal/ground_truth/pose', PoseStamped, self.ground_truth_pose_callback)
        self.feature_pub = rospy.Publisher('/expected_features_' + self.landmark_color, Point2DArrayStamped, queue_size=10)  # Modified publisher

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.f_x = msg.K[0]
        self.f_y = msg.K[4]
        self.c_x = msg.K[2]
        self.c_y = msg.K[5]
        self.camera_frame_id = msg.header.frame_id
        self.camera_info_received = True
        rospy.loginfo('Camera info received from frame %s', self.camera_frame_id)

    def get_camera_mount_parameters(self):
        if not self.camera_frame_id:
            rospy.logwarn('Camera frame ID not set yet')
            return False
        try:
            # Get the transform from 'base_link' to camera frame
            trans = self.tf_buffer.lookup_transform('base_link', self.camera_frame_id, rospy.Time(0), rospy.Duration(1.0))
            self.t_cx = trans.transform.translation.x
            self.t_cy = trans.transform.translation.y
            self.t_cz = trans.transform.translation.z
            rospy.loginfo('Camera mount parameters: t_cx=%f, t_cy=%f, t_cz=%f', self.t_cx, self.t_cy, self.t_cz)
            return True
        except tf2_ros.TransformException as e:
            rospy.logwarn('TF Exception: %s', str(e))
            return False

    def ground_truth_pose_callback(self, msg):
        if not self.camera_info_received:
            rospy.logwarn('Camera info not received yet')
            return

        if not self.get_camera_mount_parameters():
            return

        # Get robot pose
        x_r = msg.pose.position.x
        y_r = msg.pose.position.y

        # Get robot orientation (theta_r)
        orientation_q = msg.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        theta_r = euler[2]

        # Get landmark parameters
        x_l = self.landmark_x
        y_l = self.landmark_y
        h_l = self.landmark_height
        r_l = self.landmark_radius

        # Call the measurement function
        z_exp = self.measurement_model.compute_measurement(
            x_r, y_r, theta_r,
            x_l, y_l, h_l, r_l,
            self.t_cx, self.t_cy, self.t_cz,
            self.f_x, self.f_y, self.c_x, self.c_y
        )

        # Print the pixel coordinates of the four points
        rospy.loginfo('Expected pixel coordinates (u,v):')
        for i in range(0, 8, 2):
            u = z_exp[i]
            v = z_exp[i+1]
            rospy.loginfo('Point %d: u = %f, v = %f', i//2 + 1, u, v)

        # Visibility Check
        img_width = self.camera_info.width
        img_height = self.camera_info.height

        in_view = True
        for i in range(0, 8, 2):
            u = z_exp[i]
            v = z_exp[i+1]
            if u < 0 or u >= img_width or v < 0 or v >= img_height:
                in_view = False
                break

        if not in_view:
            rospy.loginfo('Landmark %s not in view', self.landmark_color)
            return

        # Publish the features as a Point2DArrayStamped
        point_array_msg = Point2DArrayStamped()
        point_array_msg.header.stamp = rospy.Time.now()
        point_array_msg.header.frame_id = self.camera_frame_id  

        # Create Point2D objects for each feature point
        for i in range(0, 8, 2):
            u = z_exp[i]
            v = z_exp[i+1]
            point = Point2D()
            point.x = u
            point.y = v
            point_array_msg.points.append(point)

        # Publish the point array
        self.feature_pub.publish(point_array_msg)
        rospy.loginfo('Published expected features for landmark %s', self.landmark_color)

    def run(self):
        rospy.spin()
        rospy.loginfo('Done')

def main():
    predictor = MeasurementPredictor()
    predictor.run()

if __name__ == '__main__':
    main()