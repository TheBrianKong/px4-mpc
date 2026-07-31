import numpy as np
import scipy.linalg
import casadi as cs
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

class FixedwingMPC:
    def __init__(self, model, x0_init = None, N=40, Tf=2.0,trackingAttitude=False):
        self.N = N
        self.Tf = Tf
        self.model = model # this is an instance of FixedwingModel, not the AcadosModel

        self.nx = self.model.get_acados_model().x.size()[0]
        self.nu = self.model.get_acados_model().u.size()[0]
        

        # need to initialize x0_init if not provided
        if x0_init is None:
            print("Warning: No initial state provided, using default initialization.")
            x0_init = np.zeros(self.nx)
            x0_init[3] = 10.0  # Initialize speed to 10 m/s
            x0_init[4] = 1.0  # Initialize quaternion w=1
        
        self.ocp_solver, self.integrator = self.setup(x0_init, self.N, self.Tf)

    def setup(self, x0, N_horizon, Tf,trackingAttitude=False):
        ocp = AcadosOcp()
        
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'mpc_codegen')
        json_path = os.path.join(codegen_dir, 'acados_ocp.json')
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_export_directory = codegen_dir

        model = self.model.get_acados_model()
        ocp.model = model
        # the model stored in self.model is an instance of FixedwingModel 
        Fmax = self.model.max_thrust_acc
        Lmax = self.model.max_lift_acc
        wmax = self.model.max_roll_rate
        
        nx = self.nx
        nu = self.nu
        ny = nx + nu
        ny_e = nx
        # COST FORMULATION (Aligned with path_following_cost.py)
        # x: [px, py, pz, speed, qw, qx, qy, qz]
        Q_all = np.diag([2e2, 2e2, 2e2,   # Position tracking priority
                         25.0,                  # Target airspeed tracking
                         50.0, 50.0, 50.0, 50.0]) # Attitude
        
        Q_pos = np.diag([8e2,8e2,8e2, 0.0,1.0,0.0,0.0,0.0])
        Q_mat = Q_all if trackingAttitude else Q_pos
        # u: [thrust, lift, roll_rate]
        R_mat = np.diag([5.0, 20.0, 20.0])    # Control effort penalties
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # weights
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        # TODO: add different terminal cost weights for better convergence
        ocp.cost.W_e = Q_mat
        
        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
        
        # CONTROL CONSTRAINTS
        ocp.constraints.lbu = np.array([0.0, 0.1, -wmax])
        ocp.constraints.ubu = np.array([Fmax, Lmax, wmax])
        ocp.constraints.idxbu = np.array([0, 1, 2])
        
        # STATE CONSTRAINTS (Stall Protection / CBF proxy)
        ocp.constraints.idxbx = np.array([3]) # Index 3 is speed
        ocp.constraints.lbx = np.array([0.0]) # Minimum airspeed (stall limit)
        ocp.constraints.ubx = np.array([30.0])# Max structural airspeed

        ocp.constraints.x0 = x0

        # SOLVER OPTIONS
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.tf = Tf

        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path)
        acados_integrator = AcadosSimSolver(ocp, json_file=json_path)

        return ocp_solver, acados_integrator
    
    def solve(self, x0, yref_trajectory, verbose=False):
        """
        Receives the receding horizon trajectory (N+1 points) and updates the solver.
        """
        ocp_solver = self.ocp_solver # Local variable aliasing for speed

        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        # Inject the lookahead trajectory
        for i in range(self.N):
            yref = np.zeros(self.nx + self.nu)
            yref[0:self.nx] = yref_trajectory[i, :]
            # Provide baseline aerodynamic targets to reduce solver effort
            yref[8] = 0.5   # Target Thrust (allow solver to handle deviations)
            yref[9] = 9.81  # Target Lift (roughly 1G for level flight)
            ocp_solver.set(i, "yref", yref)
            
        ocp_solver.set(self.N, "yref", yref_trajectory[self.N, :])

        status = ocp_solver.solve()
        if verbose:
            ocp_solver.print_statistics()

        if status != 0:
            print(f'Warning: acados returned status {status}.')
            print(f'Current state x0: {np.round(x0,2)}')
            print(f'Target Ref y[0]: {np.round(yref,2)}')

        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))

        for i in range(self.N):
            simX[i, :] = ocp_solver.get(i, "x")
            simU[i, :] = ocp_solver.get(i, "u")
        simX[self.N, :] = ocp_solver.get(self.N, "x")

        return simU, simX