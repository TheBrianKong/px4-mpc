import casadi as cs
import numpy as np
class CBFSafetyFilter:
    def __init__(self, model, filter_mode ="HOCBF", vmin=1.0, gamma1=2.0, gamma2=2.0):
        self.model = model
        self.vmin = vmin
        self.filter_mode= filter_mode
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        # hyperparams for CBF bounds
        self.lh = 0.1
        self.uh = 1e5
        # might want to offload section into a function
        x_sym = cs.SX.sym('x', 8)
        u_sym = cs.SX.sym('u', 3)
        
        speed = x_sym[3]
        q = x_sym[4:8]
        f_xw = u_sym[0]
        
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = cs.vertcat(
            cs.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            cs.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            cs.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )
        
        # gravity projected into the wind frame (z-up)
        g_inertial = cs.vertcat(0, 0, self.model.gravity)
        g_wind = cs.mtimes(R.T, g_inertial)
        
        # CBF Formulation; psi0 = h = V-Vmin
        psi0 = speed - self.vmin
        h_dot = f_xw + g_wind[0] # g_xw
        # extract symbolic expression of cbf for solver
        cbf_val = self.get_cbf_expr(x_sym,u_sym)
        
        self._eval_func = cs.Function('eval_cbf', [x_sym, u_sym], [psi0, h_dot, cbf_val])
        
        # gradient for safety filter:
        # penalty function to be smoothly differentiable and positive when cbf_val < 0
        penalty =  0.5* cs.fmax(0, -cbf_val)**2
        # gradient of penalty w.r.t. control input u (f_xw, f_zw, roll_rate)
        grad_u = cs.gradient(penalty,u_sym)
        # casadi function for eval
        self._grad_func = cs.Function('grad_cbf', [x_sym, u_sym], [penalty, grad_u])

    def get_cbf_expr(self,x,u):
        """
        BUild CBF symbolic to inject into acados or internal use for shield layer
        """
        speed = x[3]
        q = x[4:8]
        f_xw, f_zw, roll_r = u[0], u[1], u[2]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        R = cs.vertcat(
            cs.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            cs.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            cs.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )
        
        speed_safe = cs.sqrt(speed**2 + 1.0)
        g_inertial = cs.vertcat(0, 0, self.model.gravity)
        g_wind = cs.mtimes(R.T, g_inertial)
        
        # psi_0 = h(x) = V - Vmin
        psi0 = speed - self.vmin
        # 1st order dynamics (accel)
        h_dot = f_xw + g_wind[0]
        
        if self.filter_mode == "FIRST_ORDER":
            # psi_1 (x,u) = psi_0 + alpha1 (psi_0) = dot h1 + gamma_1 * h1
            cbf_val = h_dot + self.gamma1 * psi0
            
        elif self.filter_mode == "HOCBF":
            # relative degree 2; pitch rate affects acceleration
            # q_wind = a_lift + g_zw /V
            psi1 = h_dot + self.gamma1 * psi0
            q_wind = -(f_zw + g_wind[2]) / speed_safe
            r_wind = g_wind[1] / speed_safe
            omega = cs.vertcat(roll_r, q_wind, r_wind)
            
            def skew_symmetric(v):
                return cs.vertcat(
                    cs.horzcat(0, -v[0], -v[1], -v[2]),
                    cs.horzcat(v[0], 0, v[2], -v[1]),
                    cs.horzcat(v[1], -v[2], 0, v[0]),
                    cs.horzcat(v[2], v[1], -v[0], 0)
                )
                
            q_dot = 0.5 * cs.mtimes(skew_symmetric(omega), q)
            p_dot = cs.mtimes(R, cs.vertcat(speed, 0, 0))
            
            xdot = cs.vertcat(p_dot, h_dot, q_dot)
            
            # outer barrier psi_1
            
            # \ddot h = d/dt(g_xw) + \dot a_thrust =...
            h2_dot = cs.jacobian(psi1, x) @ xdot
            
            # Final Condition
            cbf_val = h2_dot + self.gamma2 * psi1
            
        else:
            raise ValueError(f"Unknown filter_mode: {self.filter_mode}")
            
        return cbf_val

    def evaluate_cbf(self, x_val, u_val):
        """
        Evaluate the CBF condition purely using CasADi for logging/plotting.
        Returns:
            h: CBF (scalar) value function
            h_dot: Time derivative of CBF
            cbf_val: condition value: h_dot(x, u) + gamma_cbf * h(x), >= 0 for safety
        """
        h, h_dot, cbf_val = self._eval_func(x_val, u_val)
        # not  sure if i have to typecast, but it's safe
        return float(h), float(h_dot), float(cbf_val)
        
    def filter_horizon(self, X_seq, U_seq, max_iters=10):
        """
        Iterate over the entire MPC predictive horizon. 
        Use CasADi gradients to pull/project unsafe control actions back into the safe envelope.
        gradient descent with adaptive step size for rapid convergence
        
        Returns:
            U_eval (np.ndarray)
            shield_activated (bool)
        """
        
        N = U_seq.shape[0]
        # lazy build of mapped casadi function to allow evaluation of N inputs simultaneously in C++
        if getattr(self, '_map_N', None) != N:
            self._grad_func_map = self._grad_func.map(N)
            self._map_N = N
        # load into casadi dense mats once
        X_eval = cs.DM(X_seq[:-1]).T  # 8 by N
        U_eval = cs.DM(U_seq).T  # 8 by N
        # speed up clipping by pre-extracting bounds
        min_b = cs.DM([self.model.min_fxw, self.model.min_fzw, -self.model.max_roll_rate])
        max_b = cs.DM([self.model.max_fxw, self.model.max_fzw, self.model.max_roll_rate])
        
        # stretches the 3x1 bounds into 3xN matrices to vectorize clip
        min_bounds = cs.repmat(min_b, 1, N)
        max_bounds = cs.repmat(max_b, 1, N)
        
        shield_activated = False
        for iteration in range(max_iters):
            # evaluate using with casadi
            pen_vals, grad_vals = self._grad_func_map(X_eval, U_eval)
            
            # sum penalties across horizon into scalar
            total_penalty = float(cs.sum(pen_vals))
            if total_penalty < 1e-3:
                break
            shield_activated = True
            # unit direction of gradient using 2-norm w/ safety
            unit_grad = grad_vals/ cs.repmat(cs.sqrt(cs.sum1(grad_vals**2))+1e-6, 3,1)

            step_size = 20.0* cs.sqrt(2.0 * pen_vals)
            U_eval -= unit_grad * cs.repmat(step_size, 3, 1)        
            U_eval = cs.fmin(cs.fmax(U_eval, min_bounds), max_bounds)
        # only translate back to NumPy at the very end to pass to the rest of stack
        else: # trigger if loop finishes without hitting the break condition
            print(f"Filter hit max iterations ({max_iters}). Total residual penalty: {total_penalty:.4f}")
        return np.array(U_eval.T), shield_activated