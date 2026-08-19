import casadi as cs
import numpy as np
import time
class CBFSafetyFilter:
    def __init__(self, model, N, repair_horizon, Ts, filter_mode ="HOCBF", vmin=1.0, gamma1=2.0, gamma2=2.0,beta=5.0,max_iters=10):
        self.model = model
        self.Ts = Ts
        self.N = N
        self.vmin = vmin
        self.filter_mode= filter_mode
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_iters = max_iters
        self.beta = beta # zeno's paradox
        self.K = repair_horizon
        # hyperparams for CBF bounds
        self.lh = 0.1
        self.uh = 1e5
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
        
        self._eval_func = cs.Function('eval_cbf', [x_sym, u_sym], [psi0, h_dot, cbf_val]).expand()
        
        # gradient for safety filter:
        # penalty function to be smoothly differentiable and positive when cbf_val < 0
        penalty =  0.5* cs.fmax(0, -cbf_val)**2
        # gradient of penalty w.r.t. control input u (f_xw, f_zw, roll_rate)
        grad_u = cs.gradient(penalty,u_sym)
        
        grad_norm_sq = cs.sumsqr(grad_u)+1e-8 # grad_u is 3x1 vector here, so sumsqr works
        grad_step = 2.0* penalty / grad_norm_sq # exact distance between u* and u_nom
        # casadi function for eval, make it compile as a binary in C
        self._grad_func = cs.Function('grad_cbf', [x_sym, u_sym], [penalty, grad_u, grad_step]).expand()
        self._map_N = None
        self.last_perf_breakdown = {
                    "ms_attr_check":0.0,
                    "ms_copy": 0.0,
                    "ms_map" : 0.0,
                    "ms_loop": 0.0,
                    "iters"  : 0
                    }
        
        acados_mod = self.model.get_acados_model()        
        x_mx = acados_mod.x            # cs.MX symbols [p, speed, q]
        u_mx = acados_mod.u            # cs.MX symbols [f_xw, f_zw, roll_r]
        xdot_mx = acados_mod.f_expl_expr # Exact continuous dynamics \dot{x}
        
        # continuous ODE function in casadi
        f_ode = cs.Function('f_ode', [x_mx, u_mx], [xdot_mx])
        
        # integrate with rk4
        dt = self.Ts
        k1 = f_ode(x_mx, u_mx)
        k2 = f_ode(x_mx + 0.5 * dt * k1, u_mx)
        k3 = f_ode(x_mx + 0.5 * dt * k2, u_mx)
        k4 = f_ode(x_mx + dt * k3, u_mx)
        x_next = x_mx + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # normalize the quaternion
        q_next = x_next[4:8]
        q_next_norm = q_next / cs.sqrt(cs.sumsqr(q_next) + 1e-6)
        x_next_safe = cs.vertcat(x_next[0:4], q_next_norm)
        
        # compile as C binary
        self._step_func = cs.Function('step_dyn', [x_mx, u_mx], [x_next_safe]).expand()
        # c binary to turn single-timestep input into held input across horizon
        u_single = cs.MX.sym('u_single',3,1)
        self._hold_input = cs.Function('hold_input', [u_single],[cs.repmat(u_single,1,N)]).expand()
        
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
        
        speed_safe = cs.sqrt(speed**2 + 1e-6)
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
            # omega = cs.vertcat(roll_r, q_wind, r_wind)
            
            # def skew_symmetric(v):
            #     return cs.vertcat(
            #         cs.horzcat(0, -v[0], -v[1], -v[2]),
            #         cs.horzcat(v[0], 0, v[2], -v[1]),
            #         cs.horzcat(v[1], -v[2], 0, v[0]),
            #         cs.horzcat(v[2], v[1], -v[0], 0)
            #     )
                
            # q_dot = 0.5 * cs.mtimes(skew_symmetric(omega), q)
            # p_dot = cs.mtimes(R, cs.vertcat(speed, 0, 0))
            
            # xdot = cs.vertcat(p_dot, h_dot, q_dot)
            
            # This is a more general expression, too much bloat
            # h_ddot = cs.jacobian(psi1, x) @ xdot
            # \ddot h = d/dt(g_xw) + \dot a_thrust =...
            
            # outer barrier psi_1
            h_ddot = r_wind * g_wind[1] - q_wind * g_wind[2] + self.gamma1 * h_dot
            
            # Final Condition
            cbf_val = h_ddot + self.gamma2 * psi1
            
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

    def filter(self, x0, U_seq):
        """
        Safety filter that propagates any input sequence on x0 over repair horizon K
        Minimally adjust unsafe control actions back into the safe set
        using gradient descent with adaptive step size for faster convergence
        
        Returns:
            U_eval (np.ndarray)
            shield_activated (bool)s
        """
        t = time.perf_counter()
        N = self.N
        K = self.K
        if not isinstance(U_seq, cs.DM):
            U_seq = cs.DM(np.atleast_2d(U_seq))
        # convert x0 to be casadi dense matrix if it's not already
        x0_dm = cs.DM(x0) if not isinstance(x0, cs.DM) else x0
        # hold the input if it's (3,1), otherwise use the (3,N) input provided
        if U_seq.numel() == 3:
            # Single control vector passed (e.g., 3x1 or 1x3) -> reshape using a tuple or single argument
            U_seq = self._hold_input(cs.reshape(U_seq, 3, 1)) # Returns (3, N) via compiled repmat
        else:
            U_seq = U_seq.T if (U_seq.shape == (N, 3)) else cs.reshape(U_seq, 3, N)
        
        U_repair= U_seq[:,:K]
        
        # lazy build of mapped casadi function to allow evaluation of N inputs simultaneously in C++
        if getattr(self, '_map_N', None) != K:
            self._grad_func_map = self._grad_func.map(K)
            self._map_N = K
            self._X_eval = cs.DM.zeros(8, K+1) # add extra for fwd prop
            self._U_eval = cs.DM.zeros(3, K)
            # speed up clipping by pre-extracting bounds
            min_b = cs.DM([self.model.min_fxw, self.model.min_fzw, -self.model.max_roll_rate])
            max_b = cs.DM([self.model.max_fxw, self.model.max_fzw, self.model.max_roll_rate])
            # stretches the 3x1 bounds into 3xN matrices to vectorize clip
            self._min_bounds = cs.repmat(min_b, 1, K)
            self._max_bounds = cs.repmat(max_b, 1, K)
            
            # unroll RK4 physics in casadi once
            x0_sym = cs.MX.sym('x0', 8)
            U_seq_sym = cs.MX.sym('U_seq', 3, K)
            X_out = [x0_sym]
            x_curr = x0_sym
            for k in range(K):
                x_curr = self._step_func(x_curr, U_seq_sym[:, k])
                X_out.append(x_curr)
            
            # 8x(K+1) matrix
            X_full_sym = cs.horzcat(*X_out)
            
            # flattens RK4 loop for entire horizon
            self._rollout_func = cs.Function('rollout', [x0_sym, U_seq_sym], [X_full_sym]).expand()
        
        t1 = time.perf_counter()
        self.last_perf_breakdown["ms_attr_check"] = (t1-t) * 1000.0
        
        # overwrite raw memory block
        self._U_eval[:,:] = U_repair # (3, K)
        self._X_eval[:,:] = self._rollout_func(x0_dm, self._U_eval)
        
        t2 = time.perf_counter()
        self.last_perf_breakdown["ms_copy"] = (t2-t1) * 1000.0
        
        # remove last point from fwd prop for dim mismatch
        pen_vals, grad_vals,grad_step = self._grad_func_map(self._X_eval[:,:-1], self._U_eval)
        
        t3 = time.perf_counter()
        self.last_perf_breakdown["ms_map"] = (t3-t2) * 1000.0
        self.last_perf_breakdown["ms_loop"] = 0.0
        self.last_perf_breakdown["iters"] = 0
        # sum penalties across horizon into scalar
        total_penalty = float(cs.sum(pen_vals))
        if total_penalty < 1e-8:
            return np.array(U_seq.T), total_penalty, False
        # print(f"[Shield] Triggered! P_I: {total_penalty:2.3e} ", end="")
        prev_penalty = float('inf')
        beta = self.beta
        for iteration in range(self.max_iters):
            # exact projection back onto boundary of safe set:  u* - u_nom = v/(\| grad h \|^2) * grad h
            if abs (prev_penalty - total_penalty) <1e-9:
                print(f"HARDWARE LIMIT: P_F: {total_penalty:2.3e} P_prev: {prev_penalty:2.3e}\t| {iteration+1:3d}/{self.max_iters} iters")
                break
            prev_penalty = total_penalty
            # gradient step of controls
            self._U_eval -= grad_vals * cs.repmat(grad_step*beta,3,1)
            self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
            
            # nonlinear fwd propagation using rk4, starting at x0
            self._X_eval = self._rollout_func(x0_dm, self._U_eval)
            # reevaluate penalties (also trim _X_eval to match dims)
            pen_vals, grad_vals,grad_step = self._grad_func_map(self._X_eval [:,:-1], self._U_eval)
            
            total_penalty = float(cs.sum(pen_vals))
            if total_penalty < 1e-8:    
                # print(f"| {iteration+1:3d}/{max_iters} iters | SUCCESS")
                # print()
                break
            t4 = time.perf_counter()
            self.last_perf_breakdown["ms_loop"] = (t4-t3) * 1000.0
            self.last_perf_breakdown["iters"] = iteration + 1
            
        U_seq[:,:K]=self._U_eval
        # only translate back to NumPy at the very end to pass to the rest of stack
        return np.array(U_seq.T), total_penalty, True