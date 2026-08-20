import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import os
import time
import casadi as cs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fixedwing_model import FixedWingModel
from px4_mpc.safety_filters import CBFSafetyFilter

def quat_to_R(q):
    """Converts a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to a quaternion [qw, qx, qy, qz].
    Standard aerospace ZYX rotation sequence.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def cs_func_cache(expr, x, u):
    return cs.Function('f_ode', [x, u], [expr])

def integrate_step(model, x, u, dt):
    """Helper to step system forward using RK4 with model equations"""
    acados_mod = model.get_acados_model()
    f_expl = acados_mod.f_expl_expr
    x_sym = acados_mod.x
    u_sym = acados_mod.u
    
    f_ode = cs_func_cache(f_expl, x_sym, u_sym)
    k1 = f_ode(x, u)
    k2 = f_ode(x + 0.5 * dt * k1, u)
    k3 = f_ode(x + 0.5 * dt * k2, u)
    k4 = f_ode(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    if isinstance(x_next, (cs.DM, cs.MX)):
        x_next = np.array(x_next).flatten()
    
    q_next = x_next[4:8]
    q_next_norm = q_next / np.sqrt(np.sum(q_next**2) + 1e-6)
    return np.concatenate([x_next[0:4], q_next_norm])

def setup_control_subplots(fig, gs, col_idx, t, U, model_obj, shield_log=None):
    """Builds the 3 control telemetry subplots"""
    ax_fxw = fig.add_subplot(gs[0, col_idx])
    ax_fzw = fig.add_subplot(gs[1, col_idx], sharex=ax_fxw)
    ax_roll = fig.add_subplot(gs[2, col_idx], sharex=ax_fxw)
    
    ax_fxw.plot(t, U[:, 0], 'k-', label='Filtered f_xw')
    ax_fxw.axhline(y=model_obj.max_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.7, label='Hardware bound')
    ax_fxw.axhline(y=model_obj.min_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fxw.set_ylabel('f_xw (m/s^2)')
    ax_fxw.set_title('Filter Commands')
    ax_fxw.grid(True)
    ax_fxw.legend(loc='upper right', fontsize='small')
    
    ax_fzw.plot(t, U[:, 1], 'b-', label='Filtered f_zw')
    ax_fzw.axhline(y=model_obj.max_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fzw.axhline(y=model_obj.min_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fzw.set_ylabel('f_zw (m/s^2)')
    ax_fzw.grid(True)
    
    ax_roll.plot(t, U[:, 2], 'g-', label='Filtered Roll Rate')
    ax_roll.axhline(y=model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_roll.axhline(y=-model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_roll.set_ylabel('roll rate (rad/s)')
    ax_roll.set_xlabel('time (s)')
    ax_roll.grid(True)
    
    if shield_log is not None and np.any(shield_log):
        ax_fxw.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_fxw.get_xaxis_transform(), label='Filter Active')
        ax_fzw.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_fzw.get_xaxis_transform())
        ax_roll.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_roll.get_xaxis_transform())

    vlines = [ax_fxw.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5),
              ax_fzw.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5),
              ax_roll.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)]
    return vlines

def setup_safety_subplots(fig, gs, col_idx, t, X, U, cbf_obj, shield_log=None):
    """Builds the strictly requested Safety Subplots: Airspeed, h(x), and CBF Condition"""
    ax_v = fig.add_subplot(gs[0, col_idx])
    ax_h = fig.add_subplot(gs[1, col_idx], sharex=ax_v)
    ax_cond = fig.add_subplot(gs[2, col_idx], sharex=ax_v)
    
    # Subplot 1: Airspeed
    speed = X[:, 3]
    ax_v.plot(t, speed, 'm-', label='Airspeed (V)')
    ax_v.axhline(y=cbf_obj.vmin, color='r', linestyle='--', label='V_min (Stall Limit)')
    ax_v.set_ylabel('Airspeed (m/s)')
    ax_v.set_title('CBF Safety Metrics')
    ax_v.grid(True)
    ax_v.legend(loc='upper right', fontsize='small')
    
    h_vals, cond_vals = [], []
    for i in range(len(t)):
        h, _, cbf_val = cbf_obj.evaluate_cbf(X[i], U[i])
        h_vals.append(h)
        cond_vals.append(cbf_val)
        
    h_vals = np.array(h_vals)
    cond_vals = np.array(cond_vals)
    
    # Subplot 2: h(x)
    ax_h.plot(t, h_vals, 'c-', label='h(x) = V - V_min')
    ax_h.axhline(y=0.0, color='r', linestyle='--', label='Boundary (h=0)')
    ax_h.set_ylabel('h(x)')
    ax_h.grid(True)
    ax_h.legend(loc='upper right', fontsize='small')
    
    # Subplot 3: CBF Condition (h_dot + gamma*h)
    ax_cond.plot(t, cond_vals, 'k-', linewidth=1.5, label=r'$\dot{h} + \gamma h$')
    ax_cond.axhline(y=0.0, color='r', linestyle='--', label='Safe Set Threshold')
    ax_cond.set_ylabel('CBF Condition')
    ax_cond.set_xlabel('time (s)')
    ax_cond.grid(True)
    ax_cond.legend(loc='upper right', fontsize='small')
    
    if shield_log is not None and np.any(shield_log):
        ax_v.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_v.get_xaxis_transform())
        ax_h.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_h.get_xaxis_transform())
        ax_cond.fill_between(t, 0, 1, where=shield_log, color='red', edgecolor='none', alpha=0.15, transform=ax_cond.get_xaxis_transform())

    vlines = [ax_v.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5),
              ax_h.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5),
              ax_cond.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)]
    return vlines

def plot_filter_results(X, U_safe, shield_log, dt, cbf_filter, model_obj):
    N_sim = len(U_safe)
    t = np.arange(N_sim) * dt
    px, py, pz = X[:-1, 0], X[:-1, 1], X[:-1, 2]
    
    fig = plt.figure(figsize=(24, 12))
    fig.canvas.manager.set_window_title("Safety Filter Standalone Dashboard")
    
    # Fixed GridSpec with proper width ratios to ensure 3D plot aspect isn't squashed
    gs = fig.add_gridspec(3, 4, width_ratios=[1.5, 1.5, 1, 1])
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15, wspace=0.3, hspace=0.3)
    
    # 3D Trajectory Plot with proper cubic aspect ratio
    ax_3d = fig.add_subplot(gs[:, :2], projection='3d')
    ax_3d.set_box_aspect((1, 1, 1)) # Forces perfect cubic layout
    ax_3d.scatter(px[0], py[0], pz[0], color='yellow', s=150, marker='*', label="Start Position", zorder=5)
    ax_3d.plot(px, py, pz, 'k-', linewidth=1.5, label='Flight Trajectory')
    
    current_pt, = ax_3d.plot([px[0]], [py[0]], [pz[0]], 'ro', markersize=8, label='UAV Position')
    
    # Orientation Triads
    quiver_scale = 30.0
    R_init = quat_to_R(X[0, 4:8])
    q_fwd = ax_3d.quiver(px[0], py[0], pz[0], R_init[0,0], R_init[1,0], R_init[2,0], color='r', length=quiver_scale, label='Body X (Fwd)')
    q_lat = ax_3d.quiver(px[0], py[0], pz[0], R_init[0,1], R_init[1,1], R_init[2,1], color='g', length=quiver_scale, label='Body Y (Left)')
    q_up  = ax_3d.quiver(px[0], py[0], pz[0], R_init[0,2], R_init[1,2], R_init[2,2], color='b', length=quiver_scale, label='Body Z (Up)')
    
    # Symmetric 3D limits constraint
    max_range = np.array([px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()]).max() / 2.0
    if max_range < 2.0: max_range = 5.0
    mid_x, mid_y, mid_z = (px.max()+px.min()) * 0.5, (py.max()+py.min()) * 0.5, (pz.max()+pz.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_3d.set_title(f"3D Trajectory & Orientation Triads (Ts={dt}s)")
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Altitude Z (m)')
    ax_3d.legend(loc='upper left', fontsize='small')

    vlines_ctrl = setup_control_subplots(fig, gs, 2, t, U_safe, model_obj, shield_log)
    vlines_safe = setup_safety_subplots(fig, gs, 3, t, X[:-1], U_safe, cbf_filter, shield_log)
    all_vlines = vlines_ctrl + vlines_safe

    # Slider & Play Button Layout
    ax_play = plt.axes([0.05, 0.03, 0.05, 0.04])
    ax_slider = plt.axes([0.15, 0.03, 0.8, 0.04])
    
    play_btn = Button(ax_play, 'Play', color='lightgoldenrodyellow', hovercolor='0.975')
    time_slider = Slider(ax=ax_slider, label='Time Index', valmin=0, valmax=N_sim-1, valinit=0, valstep=1, color='magenta')

    playing = False
    
    def update(val):
        nonlocal q_fwd, q_lat, q_up
        idx = int(time_slider.val)
        
        current_pt.set_data([px[idx]], [py[idx]])
        current_pt.set_3d_properties([pz[idx]])
        
        q_fwd.remove(); q_lat.remove(); q_up.remove()
        R_curr = quat_to_R(X[idx, 4:8])
        q_fwd = ax_3d.quiver(px[idx], py[idx], pz[idx], R_curr[0,0], R_curr[1,0], R_curr[2,0], color='r', length=quiver_scale)
        q_lat = ax_3d.quiver(px[idx], py[idx], pz[idx], R_curr[0,1], R_curr[1,1], R_curr[2,1], color='g', length=quiver_scale)
        q_up  = ax_3d.quiver(px[idx], py[idx], pz[idx], R_curr[0,2], R_curr[1,2], R_curr[2,2], color='b', length=quiver_scale)
        
        for vl in all_vlines:
            vl.set_xdata([t[idx], t[idx]])
            
        fig.canvas.draw_idle()

    def toggle_play(event):
        nonlocal playing
        playing = not playing
        if playing:
            play_btn.label.set_text('Pause')
            while playing:
                current_val = time_slider.val
                if current_val >= N_sim - 1:
                    time_slider.set_val(0)
                else:
                    time_slider.set_val(current_val + 1)
                fig.canvas.flush_events()
                time.sleep(dt)
        else:
            play_btn.label.set_text('Play')
            fig.canvas.draw_idle()

    time_slider.on_changed(update)
    play_btn.on_clicked(toggle_play)
    
    plt.show()

def run_filter_test():
    print("Initializing safety filter standalone test...")
    model = FixedWingModel()
    dt = 0.05
    N_horizon = 40
    total_sim_steps = 300
    
    v_min = 10.0
    cbf_filter = CBFSafetyFilter(model, Ts=dt, N=N_horizon, filter_mode="HOCBF", vmin=v_min, gamma1=2.0, gamma2=2.0, beta=5.0, max_iters=10)
    
    x0 = np.zeros(8)
    x0[0:3] = [1.0, -1.0, 0.0]
    x0[3] = v_min
    x0[4:8] = euler_to_quaternion(cs.pi/4, -cs.pi/4,cs.pi/6)
    
    raw_u_command = np.array([1.0, 9.8, 0.0])
    history_X = [x0.copy()]
    history_U_safe = []
    shield_logs = []
    
    x_curr = x0.copy()
    
    for step in range(total_sim_steps):
        u_seq_input = raw_u_command.reshape(1, 3)
        u_safe, active = cbf_filter.filter(x_curr, u_seq_input)
        u_to_execute = u_safe[0]
        
        x_next = integrate_step(model, x_curr, u_to_execute, dt)
        
        history_X.append(x_next)
        history_U_safe.append(u_to_execute.copy())
        shield_logs.append(active)
        
        x_curr = x_next
        
    history_X = np.array(history_X)
    history_U_safe = np.array(history_U_safe)
    shield_logs = np.array(shield_logs)
    
    print("Simulation finished. Launching interactive visualization dashboard...")
    plot_filter_results(history_X, history_U_safe, shield_logs, dt, cbf_filter, model)

if __name__ == '__main__':
    run_filter_test()