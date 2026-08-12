import casadi as cs

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
        x_sym = cs.MX.sym('x', 8)
        u_sym = cs.MX.sym('u', 3)
        
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
    
    def filter_horizon(self, X_seq, U_seq, max_iters=50, learning_rate=5.0):
        """
        Iterate over the entire MPC predictive horizon. 
        Use CasADi gradients to pull unsafe control actions back into the safe envelope.
        """
        U_safe = cs.DM(U_seq)
        shield_activated = False
        for iteration in range(max_iters):
            total_penalty = 0.0
            
            for k in range(U_safe.shape[0]):
                x_k = X_seq[k]
                # extract control at time step k: 1x3 → 3x1 col vector
                u_k = U_safe[k,:].T 
                
                # eval casadi gradient
                pen_val, grad_val = self._grad_func(x_k, u_k)
                total_penalty += float(pen_val)
                
                if float(pen_val) > 0:
                    shield_activated = True
                    # gradient away from the danger zone
                    # may have to check for NaN or Inf values in grad_val
                    U_safe[k, :] -= learning_rate * grad_val.T  # transpose to match the shape of U_safe[k]
                    
                    # ensure filter outputs are constrained within the control bounds
                    U_safe[k, 0] = cs.fmin(cs.fmax(U_safe[k, 0], self.model.min_fxw), self.model.max_fxw)
                    U_safe[k, 1] = cs.fmin(cs.fmax(U_safe[k, 1], self.model.min_fzw), self.model.max_fzw)
                    U_safe[k, 2] = cs.fmin(cs.fmax(U_safe[k, 2], -self.model.max_roll_rate), self.model.max_roll_rate)
            
            # if the entire horizon is safe, stop iterating early to save CPU
            if total_penalty < 1e-3:
                break
        return U_safe.full(), shield_activated