"""Smoke tests: the example scripts run end to end without the GUI."""
import numpy as np
import pytest

from gym_pybullet_drones.examples import hover_test
from gym_pybullet_drones.examples.trajectory_tracking import run, TRAJECTORIES


@pytest.mark.parametrize('traj', list(TRAJECTORIES))
def test_trajectory_tracking(traj, tmp_path):
    rmse = run(traj=traj, gui=False, plot=False, output_folder=str(tmp_path))
    assert set(rmse) == {'x', 'y', 'z', 'norm'}
    assert all(np.isfinite(value) for value in rmse.values())


def test_hover_test(tmp_path):
    result = hover_test.run(duration=1.0, gui=False, plot=False, output_folder=str(tmp_path))
    assert set(result) == {'converged', 'settling_time', 'initial_angle_deg', 'final_angle_deg'}
    assert np.isfinite(result['initial_angle_deg'])
