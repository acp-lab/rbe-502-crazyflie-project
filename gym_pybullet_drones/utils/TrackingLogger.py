"""Logging, RMSE, and plots for the RBE 502 trajectory tracking project."""
import os

import numpy as np
from scipy.spatial.transform import Rotation


def attitude_error(rot, rot_d):
    """Returns the attitude error e_R = 1/2 (R_d^T R - R^T R_d)^vee.

    `rot` and `rot_d` are (N, 3, 3)-shaped arrays of rotation matrices R and R_d.
    Returns an (N, 3)-shaped array.
    """
    S = np.einsum('kji,kjl->kil', rot_d, rot) - np.einsum('kji,kjl->kil', rot, rot_d)
    return 0.5 * np.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], axis=1)


def euler_zxy(rot):
    """Returns the roll, pitch, and yaw angles [rad] of (N, 3, 3)-shaped rotation matrices,
    as Z-X-Y Euler angles, R = Rz(yaw) Rx(roll) Ry(pitch). Used for plots only.

    With this convention, the yaw angle of R_d equals the desired yaw angle psi_d.
    Returns an (N, 3)-shaped array of [roll, pitch, yaw].
    """
    return Rotation.from_matrix(rot).as_euler('ZXY')[:, [1, 2, 0]]


class TrackingLogger:
    """Records the desired and actual states and the control inputs at every control step."""

    ################################################################################

    def __init__(self, traj_name, traj_duration):
        """
        Parameters
        ----------
        traj_name : str
            Name of the trajectory, used in plot titles and file names.
        traj_duration : float
            Duration of the trajectory [s]. The RMSE is computed over [0, traj_duration].

        """
        self.traj_name = traj_name
        self.traj_duration = traj_duration
        self._data = {key: [] for key in ['t', 'pos', 'vel', 'rot', 'ang_vel',
                                          'pos_d', 'vel_d', 'acc_d', 'rot_d',
                                          'thrust', 'moment', 'rpm']}

    ################################################################################

    def log(self, t, state, des_state, info, rpm):
        """Stores one control step.

        Parameters
        ----------
        t : float
            Time [s].
        state : ndarray
            (20,)-shaped state vector from the environment.
        des_state : dict
            Desired state with keys 'pos', 'vel', and 'acc'.
        info : dict
            Second output of `GeometricControl.computeControlFromState`.
        rpm : ndarray
            (4,)-shaped rotor speed command.

        """
        d = self._data
        d['t'].append(t)
        d['pos'].append(np.array(state[0:3]))
        d['vel'].append(np.array(state[10:13]))
        d['rot'].append(info['rot'])
        d['ang_vel'].append(info['ang_vel'])
        d['pos_d'].append(np.array(des_state['pos'], dtype=float))
        d['vel_d'].append(np.array(des_state['vel'], dtype=float))
        d['acc_d'].append(np.array(des_state['acc'], dtype=float))
        d['rot_d'].append(info['target_rot'])
        d['thrust'].append(info['thrust'])
        d['moment'].append(info['moment'])
        d['rpm'].append(np.array(rpm))

    ################################################################################

    def arrays(self):
        """Returns the log as a dictionary of arrays, with the derived errors."""
        a = {key: np.array(val) for key, val in self._data.items()}
        a['pos_err'] = a['pos'] - a['pos_d']
        a['att_err'] = attitude_error(a['rot'], a['rot_d'])
        a['rpy'] = euler_zxy(a['rot'])
        a['rpy_d'] = euler_zxy(a['rot_d'])
        return a

    ################################################################################

    def rmse(self):
        """Returns the RMSE of each component of the position error and of its norm [m],
        computed over the duration of the trajectory."""
        a = self.arrays()
        err = a['pos_err'][a['t'] <= self.traj_duration + 1e-9]
        rmse_xyz = np.sqrt(np.mean(err**2, axis=0))
        return {'x': rmse_xyz[0],
                'y': rmse_xyz[1],
                'z': rmse_xyz[2],
                'norm': np.sqrt(np.mean(np.sum(err**2, axis=1)))}

    ################################################################################

    def save(self, folder):
        """Saves the log to `folder/log.npz`."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'log.npz')
        np.savez(path, traj_name=self.traj_name, traj_duration=self.traj_duration, **self.arrays())
        return path

    ################################################################################

    def plot(self, folder=None, show=True):
        """Plots the tracking results. Saves the figures to `folder` if it is given."""
        import matplotlib.pyplot as plt

        a = self.arrays()
        t = a['t']
        title = f"{self.traj_name} ({self.traj_duration:g} s)"
        axes_names = ['x', 'y', 'z']
        figs = {}

        #### Position ##############################################################
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for i, ax in enumerate(axs):
            ax.plot(t, a['pos_d'][:, i], 'k--', label='desired')
            ax.plot(t, a['pos'][:, i], label='actual')
            ax.set_ylabel(f'{axes_names[i]} [m]')
            ax.grid(True)
        axs[0].legend(loc='upper right')
        axs[0].set_title(f'Position, {title}')
        axs[-1].set_xlabel('t [s]')
        figs['position'] = fig

        #### 3D path ###############################################################
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot(*a['pos_d'].T, 'k--', label='desired')
        ax.plot(*a['pos'].T, label='actual')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title(f'3D path, {title}')
        ax.legend()
        figs['path_3d'] = fig

        #### Position error ########################################################
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for i, ax in enumerate(axs):
            ax.plot(t, a['pos_err'][:, i])
            ax.set_ylabel(f'$e_{{{axes_names[i]}}}$ [m]')
            ax.grid(True)
        axs[0].set_title(f'Position error, {title}')
        axs[-1].set_xlabel('t [s]')
        figs['position_error'] = fig

        #### Attitude ##############################################################
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
        for i, name in enumerate(['roll', 'pitch', 'yaw']):
            axs[i].plot(t, np.degrees(a['rpy_d'][:, i]), 'k--', label='$R_d$')
            axs[i].plot(t, np.degrees(a['rpy'][:, i]), label='$R$')
            axs[i].set_ylabel(f'{name} [deg]')
            axs[i].grid(True)
        axs[0].legend(loc='upper right')
        axs[3].plot(t, np.linalg.norm(a['att_err'], axis=1))
        axs[3].set_ylabel(r'$\|e_R\|$')
        axs[3].grid(True)
        axs[0].set_title(f'Attitude, {title}')
        axs[-1].set_xlabel('t [s]')
        figs['attitude'] = fig

        #### Control inputs ########################################################
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
        axs[0].plot(t, a['thrust'])
        axs[0].set_ylabel('f [N]')
        for i in range(3):
            axs[i + 1].plot(t, a['moment'][:, i])
            axs[i + 1].set_ylabel(f'$M_{i + 1}$ [N m]')
        for ax in axs:
            ax.grid(True)
        axs[0].set_title(f'Control inputs, {title}')
        axs[-1].set_xlabel('t [s]')
        figs['control'] = fig

        for fig in figs.values():
            fig.tight_layout()
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(os.path.join(folder, f'{name}.png'), dpi=150)
        if show:
            plt.show()
        else:
            for fig in figs.values():
                plt.close(fig)
        return figs
