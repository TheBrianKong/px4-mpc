import casadi as cs
from acados_template import AcadosModel

class FixedwingModel():
    def __init__(self):
        self.name = 'fixedwing_model'
        self.gravity = -9.81
        self.max_thrust_acc = 1.0
        self.max_lift_acc = 20.0
        self.max_roll_rate = 1.0

    def get_acados_model(self) -> AcadosModel:
        # helper functions
        def skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0)
            )
        # converts from quaternion in wxyz to rotation matrix
        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            return cs.vertcat(
                cs.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
                cs.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
                cs.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
            )

        model = AcadosModel()
        
        # states
        p      = cs.MX.sym('p', 3)
        speed  = cs.MX.sym('speed', 1)
        q      = cs.MX.sym('q', 4)
        x      = cs.vertcat(p, speed, q)
        # controls
        thrust = cs.MX.sym('thrust', 1)
        lift   = cs.MX.sym('lift', 1)
        roll_r = cs.MX.sym('roll_rate', 1)
        u      = cs.vertcat(thrust, lift, roll_r)
        
        xdot   = cs.MX.sym('xdot', 8) # p_dot(3), s_dot(1), q_dot(4)

        # rotate velocity: B → I frame
        q_norm = q / cs.norm_2(q)
        R = q_to_rot_mat(q_norm)
        velocity_i = cs.mtimes(R, cs.vertcat(speed, 0, 0))

        
        # safe inverse to avoid div 0
        speed_safe = cs.if_else(cs.fabs(speed) < 1e-6, 1e-6, speed)
        
        # rotate gravity vector from inertial to body frame
        g_inertial = cs.vertcat(0, 0, self.gravity)
        g_body = cs.mtimes(R.T, g_inertial)

        # describe gravity vector in terms of angular rates in wind frame (pitch q, yaw r)
        q_wind = -(lift + g_body[2]) / speed_safe
        r_wind = g_body[1] / speed_safe
        # fwd accel with gravity
        accel  = thrust

        # quaternion kinematics based on angular rates
        omega = cs.vertcat(roll_r, q_wind, r_wind)
        
        q_dot = 0.5 * cs.mtimes(skew_symmetric(omega), q)
        # pull q_dot towards unit norm for integrator
        q_dot += 0.5 * (1.0- cs.sumsqr(q))*q 
        
        f_expl = cs.vertcat(velocity_i, accel, q_dot)
        
        model.f_impl_expr = xdot - f_expl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name
        
        return model
