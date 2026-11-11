"""Hover test for the attitude controller.

RBE 502 Quadrotor Control Project. Use this test to debug `_attitudeControl` in
`GeometricControl.py` on its own, before you track the trajectories.

The quadrotor starts at rest, 1 m above the ground, with a tilted attitude. The test
skips your position controller and commands, at every control step,
    - the total thrust f = m g,
    - the desired attitude R_d = I (level, with zero yaw),
    - the desired angular velocity Omega_d = 0,
so only your `_attitudeControl` and your gains K_R and K_Omega act on the quadrotor.
With a correct attitude controller, R converges to I within a fraction of a second.

The position is not controlled in this test. While the quadrotor is tilted, its
thrust pushes it sideways and its vertical component is below m g, so the quadrotor
drifts horizontally and sinks a little. This drift is expected.

Example
-------
From the `examples` folder, run:

    $ python hover_test.py
    $ python hover_test.py --init_rpy 30 0 0
    $ python hover_test.py --init_rpy 0 0 45 --gui false

The script prints whether the attitude converged, saves the log and the figures
to `results/`, and shows the figures.

"""
import os
import time
import argparse
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.GeometricControl import GeometricControl
from gym_pybullet_drones.utils.TrackingLogger import attitude_error, euler_zxy
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.examples.trajectory_tracking import SIMULATION_FREQ_HZ, CONTROL_FREQ_HZ

#### Test settings ###############################################################
INIT_HEIGHT = 1.0           # initial height [m]
CONVERGENCE_TOL_DEG = 1.0   # the attitude has converged once the error angle stays below this value [deg]
HOLD_SEC = 0.5              # ... until the end of the test, for at least this long [s]
GROUND_HEIGHT = 0.1         # below this height [m], the quadrotor has reached the ground

DEFAULT_INIT_RPY_DEG = [20.0, -15.0, 30.0]
DEFAULT_DURATION_SEC = 3.0
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'


def run(init_rpy=DEFAULT_INIT_RPY_DEG,
        duration=DEFAULT_DURATION_SEC,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        record_video=DEFAULT_RECORD_VIDEO,
        output_folder=DEFAULT_OUTPUT_FOLDER
        ):
    """Runs the hover test and returns whether the attitude converged.

    Parameters
    ----------
    init_rpy : list of float
        Initial roll, pitch, and yaw [deg], as Z-X-Y Euler angles,
        R = Rz(yaw) Rx(roll) Ry(pitch), the convention of the plots.
    duration : float
        Duration of the test [s].
    gui : bool
        Whether to show the PyBullet GUI.
    plot : bool
        Whether to show the figures. The figures are saved in both cases.
    record_video : bool
        Whether to record a video of the simulation.
    output_folder : str
        Folder where the log and the figures are saved.

    Returns
    -------
    dict
        'converged' (bool), 'settling_time' [s] (None if the attitude did not converge),
        and 'initial_angle_deg' and 'final_angle_deg', the attitude error angle [deg],
        that is, the rotation angle between R and R_d.

    """
    ctrl = GeometricControl(drone_model=DroneModel.CF2X)

    #### Commands of the hover test ################################################
    thrust = ctrl.m * ctrl.g        # f = m g
    target_rot = np.eye(3)          # R_d = I, and Omega_d = 0 in _attitudeControl

    #### Initial attitude ###########################################################
    # PyBullet builds the initial attitude from Euler angles with R = Rz(yaw) Ry(pitch) Rx(roll),
    # so convert the Z-X-Y angles of the plots into these angles.
    roll, pitch, yaw = np.radians(np.asarray(init_rpy, dtype=float))
    init_rot = Rotation.from_euler('ZXY', [yaw, roll, pitch])
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=np.array([[0.0, 0.0, INIT_HEIGHT]]),
                     initial_rpys=init_rot.as_euler('xyz').reshape(1, 3),
                     physics=Physics.PYB,
                     pyb_freq=SIMULATION_FREQ_HZ,
                     ctrl_freq=CONTROL_FREQ_HZ,
                     gui=gui,
                     record=record_video
                     )
    obs, _ = env.reset()

    #### Run the simulation ########################################################
    log = {key: [] for key in ['t', 'pos', 'rot', 'ang_vel', 'moment', 'rotor_thrusts']}
    num_steps = int(round(duration * env.CTRL_FREQ))
    start = time.time()
    for i in range(num_steps):
        rpm, info = ctrl.computeAttitudeControlFromState(state=obs[0], thrust=thrust, target_rot=target_rot)
        log['t'].append(i * env.CTRL_TIMESTEP)
        log['pos'].append(np.array(obs[0][0:3]))
        for key in ['rot', 'ang_vel', 'moment', 'rotor_thrusts']:
            log[key].append(info[key])
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)
    env.close()

    a = {key: np.array(val) for key, val in log.items()}
    a['att_err'] = attitude_error(a['rot'], np.broadcast_to(target_rot, a['rot'].shape))
    a['rpy'] = euler_zxy(a['rot'])
    # Attitude error angle: the rotation angle of R_d^T R. The test uses this angle rather
    # than ||e_R||, since ||e_R|| is also zero when the quadrotor is upside down.
    cos_angle = (np.einsum('ij,kij->k', target_rot, a['rot']) - 1.0) / 2.0
    a['att_err_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    angle = a['att_err_angle']

    #### Report ####################################################################
    # Only the data before the quadrotor reaches the ground are used, since the
    # contact with the ground disturbs the attitude.
    on_ground = np.nonzero(a['pos'][:, 2] < GROUND_HEIGHT)[0]
    n = on_ground[0] if len(on_ground) else len(angle)
    t_end = a['t'][n - 1] + env.CTRL_TIMESTEP if n > 0 else 0.0
    # Settling time: the first time after which the error angle stays below the tolerance.
    above = np.nonzero(~(angle[:n] < CONVERGENCE_TOL_DEG))[0]
    if len(above) == 0:
        settling_time = 0.0
    elif above[-1] < n - 1:
        settling_time = float(a['t'][above[-1] + 1])
    else:
        settling_time = None
    converged = n > 0 and settling_time is not None and settling_time <= t_end - HOLD_SEC
    if not converged:
        settling_time = None
    flipped = np.any(a['rot'][:n, 2, 2] < 0.0)      # b3 pointed downward

    print(f"\nHover test: initial roll {init_rpy[0]:g} deg, pitch {init_rpy[1]:g} deg, "
          f"yaw {init_rpy[2]:g} deg, attitude error angle {angle[0]:.1f} deg")
    print(f"Commands: f = m g = {thrust:.4f} N, R_d = I, Omega_d = 0")
    if len(on_ground):
        print(f"Note: the quadrotor reached the ground at t = {t_end:.2f} s. The test uses the data "
              f"before this time only.")
    if converged:
        print(f"PASS: the attitude converged. The attitude error angle stays below "
              f"{CONVERGENCE_TOL_DEG:g} deg after t = {settling_time:.2f} s.")
    else:
        print(f"FAIL: the attitude did not converge. The attitude error angle is {angle[max(n, 1) - 1]:.1f} deg "
              f"at t = {t_end:.2f} s (tolerance {CONVERGENCE_TOL_DEG:g} deg).")
        if flipped:
            print("The quadrotor flipped upside down.")
        print("Things to check in _attitudeControl and your gains:")
        print("  - e_R = 1/2 (R_d^T R - R^T R_d)^vee, with the matrices in this order,")
        print("  - the signs of the K_R and K_Omega terms in the moment,")
        print("  - K_R and K_Omega are positive definite. If the attitude oscillates with a growing")
        print("    amplitude, reduce them: the controller runs at "
              f"{CONTROL_FREQ_HZ} Hz, and the rotor thrusts are limited.")
        if len(on_ground) and not flipped and angle[n - 1] < angle[0]:
            print("  - The quadrotor may have reached the ground before its attitude converged. "
                  "Try a smaller initial tilt.")

    run_folder = os.path.join(output_folder, f"hover_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_folder, exist_ok=True)
    np.savez(os.path.join(run_folder, 'log.npz'), init_rpy_deg=np.asarray(init_rpy, dtype=float),
             thrust=thrust, target_rot=target_rot, **a)
    _plot(a, ctrl.MAX_ROTOR_THRUST, run_folder, show=plot)
    print(f"Log and figures saved to {os.path.abspath(run_folder)}")

    return {'converged': bool(converged),
            'settling_time': settling_time,
            'initial_angle_deg': float(angle[0]),
            'final_angle_deg': float(angle[max(n, 1) - 1])}


def _plot(a, max_rotor_thrust, folder, show=True):
    """Plots the attitude, the angular velocity and moment, and the rotor thrusts."""
    import matplotlib.pyplot as plt

    t = a['t']
    figs = {}

    #### Attitude ##################################################################
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
    for i, name in enumerate(['roll', 'pitch', 'yaw']):
        axs[i].axhline(0.0, color='k', linestyle='--', label='$R_d = I$')
        axs[i].plot(t, np.degrees(a['rpy'][:, i]), label='$R$')
        axs[i].set_ylabel(f'{name} [deg]')
    axs[0].legend(loc='upper right')
    axs[3].plot(t, a['att_err_angle'])
    axs[3].axhline(CONVERGENCE_TOL_DEG, color='k', linestyle=':', label='tolerance')
    axs[3].set_ylabel('error angle [deg]')
    axs[3].legend(loc='upper right')
    axs[0].set_title('Hover test: attitude')
    figs['hover_attitude'] = fig

    #### Angular velocity and moment ###############################################
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
    for i in range(3):
        axs[i, 0].plot(t, a['ang_vel'][:, i])
        axs[i, 0].set_ylabel(rf'$\Omega_{i + 1}$ [rad/s]')
        axs[i, 1].plot(t, a['moment'][:, i])
        axs[i, 1].set_ylabel(f'$M_{i + 1}$ [N m]')
    axs[0, 0].set_title('Hover test: body angular velocity')
    axs[0, 1].set_title('Hover test: moment')
    figs['hover_rates_moment'] = fig

    #### Rotor thrusts #############################################################
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(4):
        ax.plot(t, a['rotor_thrusts'][:, i], label=f'$F_{i + 1}$')
    ax.axhline(0.0, color='k', linestyle=':')
    ax.axhline(max_rotor_thrust, color='k', linestyle=':', label='limits')
    ax.set_ylabel('rotor thrust [N]')
    ax.legend(loc='upper right', ncol=5)
    ax.set_title('Hover test: rotor thrusts')
    figs['hover_rotor_thrusts'] = fig

    for name, fig in figs.items():
        for ax in fig.axes:
            ax.grid(True)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('t [s]')
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f'{name}.png'), dpi=150)
    if show:
        plt.show()
    else:
        for fig in figs.values():
            plt.close(fig)
    return figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hover test for the attitude controller')
    parser.add_argument('--init_rpy', default=DEFAULT_INIT_RPY_DEG, type=float, nargs=3,
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help=f'Initial roll, pitch, and yaw in degrees (default: {DEFAULT_INIT_RPY_DEG})')
    parser.add_argument('--duration', default=DEFAULT_DURATION_SEC, type=float,
                        help=f'Duration of the test in seconds (default: {DEFAULT_DURATION_SEC})')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, metavar='',
                        help=f'Whether to use the PyBullet GUI (default: {DEFAULT_GUI})')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, metavar='',
                        help=f'Whether to show the figures (default: {DEFAULT_PLOT})')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, metavar='',
                        help=f'Whether to record a video (default: {DEFAULT_RECORD_VIDEO})')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, metavar='',
                        help=f'Folder where the log and figures are saved (default: "{DEFAULT_OUTPUT_FOLDER}")')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
