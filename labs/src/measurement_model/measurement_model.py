# #!/usr/bin/env python
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
    def __init__(self, xl, yl, hl, rl, color,
                 tcx=0.229, tcy=0.0, tcz=0.2165,
                 fx=788.408537, fy=788.408537, cx=512.5, cy=384.5):

        # Store parameters:
        self.x_l = xl
        self.y_l = yl
        self.h_l = hl
        self.r_l = rl
        self.color = color

        # Statuc camera parameters
        self.t_cx = tcx
        self.t_cy = tcy
        self.t_cz = tcz
        self.f_x = fx
        self.f_y = fy
        self.c_x = cx
        self.c_y = cy

        # Predefined according to Assignment1
        self.R = np.diag([51.943872, 2.257994, 56.141963, 16.346652, 50.062509, 2.562840, 49.591893, 2.556726])

        # Deriving the models
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

    


