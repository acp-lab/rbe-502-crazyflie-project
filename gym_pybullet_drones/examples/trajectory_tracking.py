"""Trajectory tracking with the geometric controller.

RBE 502 Quadrotor Control Project. The simulation is run by a `CtrlAviary`
environment, and the control is given by your `GeometricControl`.

Example
-------
From the `examples` folder, run:

    $ python trajectory_tracking.py --traj circle
    $ python trajectory_tracking.py --traj diamond
    $ python trajectory_tracking.py --traj diamond --traj_duration 6 --gui false

The script prints the RMSE of the position error, saves the log and the
figures to `results/`, and shows the figures.

"""
import os
import time
import argparse
from datetime import datetime

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.GeometricControl import GeometricControl
from gym_pybullet_drones.utils.TrackingLogger import TrackingLogger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.trajectory import circle, diamond

#### Trajectories: name -> (function of (t, duration), default duration [s]) ####
TRAJECTORIES = {
    'circle': (circle.circle, 15.0),
    'diamond': (diamond.diamond, 8.0),
}

#### Simulation settings (do not change; used for grading) ######################
SIMULATION_FREQ_HZ = 1000
CONTROL_FREQ_HZ = 200
HOVER_SEC = 2.0     # hover at the final point after the trajectory ends

#### Controller evaluation: (upper bound on the RMSE of the position error norm [m], score)
EVALUATION_TIERS = [(0.05, 100), (0.10, 90), (0.15, 80), (0.20, 60)]

DEFAULT_TRAJ = 'circle'
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'


def evaluation_score(rmse_norm):
    """Returns the controller evaluation score for one trajectory, as in the project statement."""
    if not np.isfinite(rmse_norm):
        return 0
    for i, (bound, score) in enumerate(EVALUATION_TIERS):
        is_last = i == len(EVALUATION_TIERS) - 1
        if rmse_norm < bound or (is_last and rmse_norm <= bound):
            return score
    return 0


def run(traj=DEFAULT_TRAJ,
        traj_duration=None,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        record_video=DEFAULT_RECORD_VIDEO,
        output_folder=DEFAULT_OUTPUT_FOLDER
        ):
    """Tracks a trajectory and returns the RMSE of the position error.

    Parameters
    ----------
    traj : str
        'circle' or 'diamond'.
    traj_duration : float, optional
        Total duration of the trajectory [s]. Defaults to the default duration of the trajectory.
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
        RMSE of each component of the position error and of its norm [m].

    """
    if traj not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory '{traj}'. Choose from {list(TRAJECTORIES)}.")
    traj_fn, default_duration = TRAJECTORIES[traj]
    duration = default_duration if traj_duration is None else float(traj_duration)
    if traj == 'circle' and duration != default_duration:
        print(f"[WARNING] circle.py does not support custom durations yet; it always runs for {default_duration:g} s.")
        duration = default_duration

    def desired_state(t):
        return traj_fn(t, duration)

    ctrl = GeometricControl(drone_model=DroneModel.CF2X)
    logger = TrackingLogger(traj_name=traj, traj_duration=duration)

    #### Create the environment, with the quadrotor at rest at the start of the trajectory.
    #### A trajectory that starts on the ground (z = 0) starts with the quadrotor resting
    #### on the ground, since its collision body cannot be spawned below the ground plane.
    init_pos = np.array(desired_state(0.0)['pos'], dtype=float)
    rest_height = ctrl._getURDFParameter('length') / 2 - ctrl._getURDFParameter('collision_z_offset')
    init_pos[2] = max(init_pos[2], rest_height)
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=init_pos.reshape(1, 3),
                     initial_rpys=np.zeros((1, 3)),
                     physics=Physics.PYB,
                     pyb_freq=SIMULATION_FREQ_HZ,
                     ctrl_freq=CONTROL_FREQ_HZ,
                     gui=gui,
                     record=record_video
                     )
    obs, _ = env.reset()

    #### Draw the desired path in the GUI ##########################################
    if gui:
        path = np.array([desired_state(t)['pos'] for t in np.linspace(0, duration, 200)])
        for k in range(len(path) - 1):
            p.addUserDebugLine(path[k], path[k + 1], lineColorRGB=[1, 0, 0], lineWidth=2,
                               physicsClientId=env.getPyBulletClient())

    #### Run the simulation ########################################################
    num_steps = int(round((duration + HOVER_SEC) * env.CTRL_FREQ))
    start = time.time()
    for i in range(num_steps):
        t = i * env.CTRL_TIMESTEP
        des = desired_state(t)
        rpm, info = ctrl.computeControlFromState(state=obs[0],
                                                 target_pos=des['pos'],
                                                 target_vel=des['vel'],
                                                 target_acc=des['acc'],
                                                 target_yaw=des['yaw']
                                                 )
        logger.log(t, obs[0], des, info, rpm)
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)
    env.close()

    #### Report and save the results ###############################################
    rmse = logger.rmse()
    print(f"\nTrajectory: {traj}, duration: {duration:g} s")
    print(f"RMSE [m]: x = {rmse['x']:.4f}, y = {rmse['y']:.4f}, z = {rmse['z']:.4f}, "
          f"position error norm = {rmse['norm']:.4f}")
    if duration == default_duration:
        print(f"Controller evaluation score for this trajectory: {evaluation_score(rmse['norm'])}")

    run_folder = os.path.join(output_folder,
                              f"{traj}_{duration:g}s_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    log_path = logger.save(run_folder)
    logger.plot(folder=run_folder, show=plot)
    print(f"Log and figures saved to {os.path.dirname(os.path.abspath(log_path))}")
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trajectory tracking with the geometric controller')
    parser.add_argument('--traj', default=DEFAULT_TRAJ, type=str, choices=list(TRAJECTORIES),
                        help=f'Trajectory to track (default: {DEFAULT_TRAJ})')
    parser.add_argument('--traj_duration', default=None, type=float,
                        help='Total duration of the trajectory in seconds (default: the default duration of the trajectory)')
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
