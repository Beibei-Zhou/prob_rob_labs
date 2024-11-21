# measurement_model.py

import sympy as sp
import numpy as np
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

        p_1g = sp.Matrix([x1, y1, 0, 1])  # Bottom-left corner
        p_2g = sp.Matrix([x2, y2, 0, 1])  # Bottom-right corner
        p_3g = sp.Matrix([x2, y2, h_l, 1])  # Top-right corner
        p_4g = sp.Matrix([x1, y1, h_l, 1])  # Top-left corner

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

        T_om = simplify((T_mr * T_ro).inv())

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
