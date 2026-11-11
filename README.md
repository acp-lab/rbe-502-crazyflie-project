# RBE 502 Quadrotor Control Project: Geometric Tracking Control

In this project, you implement a geometric tracking controller for the Crazyflie 2.X quadrotor and use it to track a circle and a diamond trajectory in a PyBullet simulator. The project statement on Canvas describes the controller, the tasks, and the grading. The simulator is based on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

## Installation

The project uses Python 3.10 in a conda environment.

### 1. Install conda

Skip this step if `conda --version` already prints a version in your terminal, for example from an existing Anaconda or Miniconda installation.

Otherwise, install [Miniforge](https://github.com/conda-forge/miniforge), a free, minimal conda installer that uses the community conda-forge channel.

**macOS and Linux.** In a terminal, run:

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the license, keep the default install location, and answer `yes` when the installer asks whether to initialize conda. Then close the terminal and open a new one.

**Windows.** Download `Miniforge3-Windows-x86_64.exe` from the [latest Miniforge release](https://github.com/conda-forge/miniforge/releases/latest) and run it with the default options. Then open **Miniforge Prompt** from the Start menu, and run all the commands in this README in it.

Check that conda works:

```sh
conda --version
```

### 2. Create the environment and install the simulator

```sh
git clone https://github.com/acp-lab/rbe-502-crazyflie-project.git
cd rbe-502-crazyflie-project

conda create -n drones python=3.10
conda activate drones

python -m pip install --upgrade pip
python -m pip install -e .
```

If `git` is not installed, download the repository as a ZIP file from GitHub (**Code > Download ZIP**), unzip it, and `cd` into the unzipped folder instead.

Run `conda activate drones` in every new terminal before you work on the project.

### Building `pybullet`

`pip` downloads a prebuilt `pybullet` only on Linux (x86_64). On macOS and Windows, it builds `pybullet` from source, which takes several minutes and needs a C++ compiler:

- **macOS:** install the Xcode command line tools with `xcode-select --install`. If the build fails with `error: expected identifier or '('` in `_stdio.h` (recent Xcode command line tools), install with:

  ```sh
  CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen" python -m pip install -e .
  ```

- **Windows:** install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload, then run `python -m pip install -e .` again.
- **Linux:** if the build is needed and fails, install a compiler with `sudo apt install build-essential`.

## What you implement

Complete the three blocks marked `YOUR CODE STARTS HERE` in [`gym_pybullet_drones/control/GeometricControl.py`](gym_pybullet_drones/control/GeometricControl.py):

1. `__init__`: the gain matrices `K_x`, `K_v`, `K_R`, and `K_Omega`.
2. `_positionControl`: the total thrust `f` and the desired rotation matrix `R_d`.
3. `_attitudeControl`: the moment `M`.

The rest of the class is provided: it converts the simulator state into the position, velocity, rotation matrix, and body-frame angular velocity, and converts your thrust and moment into rotor speeds. Do not change the function signatures or any other file, since we grade your controller by running your `GeometricControl.py` with the provided code.

## Testing your attitude controller: hover test

Implement and test `_attitudeControl` before you track the trajectories. From the `examples` folder:

```sh
cd gym_pybullet_drones/examples
python hover_test.py
python hover_test.py --init_rpy 45 0 0   # initial roll, pitch, and yaw in degrees
```

The quadrotor starts at rest, 1 m above the ground, with roll 20°, pitch -15°, and yaw 30° by default. The test skips your position controller and commands the total thrust `f = m g`, the desired attitude `R_d = I`, and the desired angular velocity `Omega_d = 0`, so only your `_attitudeControl` and your gains `K_R` and `K_Omega` act on the quadrotor. The script prints `PASS` if the attitude error angle, the rotation angle between `R` and `R_d`, falls below 1° and stays there, and `FAIL` with hints otherwise. It plots the roll, pitch, and yaw, the attitude error angle, the body angular velocity, the moment, and the rotor thrusts.

The position is not controlled in this test: while it is tilted, the quadrotor drifts sideways and sinks a little. This is expected.

| Option | Default | Description |
|---|---|---|
| `--init_rpy` | `20 -15 30` | Initial roll, pitch, and yaw in degrees (Z-X-Y Euler angles). |
| `--duration` | `3` | Duration of the test in seconds. |
| `--gui`, `--plot`, `--record_video`, `--output_folder` | | As in `trajectory_tracking.py` below. |

## Running a simulation

From the `examples` folder:

```sh
cd gym_pybullet_drones/examples
python trajectory_tracking.py --traj circle
python trajectory_tracking.py --traj diamond
```

| Option | Default | Description |
|---|---|---|
| `--traj` | `circle` | Trajectory to track: `circle` or `diamond`. |
| `--traj_duration` | trajectory default | Total duration of the trajectory in seconds (circle: 15 s, diamond: 8 s). |
| `--gui` | `true` | Show the PyBullet GUI. The desired path is drawn in red. Use `--gui false` to run faster than real time. |
| `--plot` | `true` | Show the figures at the end of the run. |
| `--record_video` | `false` | Record a video of the run. |
| `--output_folder` | `results` | Folder where the log and the figures are saved. |

The simulator runs at 1000 Hz and calls your controller at 200 Hz. At the end of each run, the script:

- prints the RMSE of each component of the position error and of its norm,
- prints the controller evaluation score for the trajectory, when it runs with its default duration,
- saves the log (`log.npz`) and the figures (position, 3D path, position error, attitude, and control inputs) to `results/<trajectory>_<duration>s_<date-time>/`.

You can load the log for your own plots with `numpy.load('log.npz')`.

## Code base overview

| Path | Content |
|---|---|
| `gym_pybullet_drones/control/GeometricControl.py` | Your controller. |
| `gym_pybullet_drones/examples/hover_test.py` | Tests your attitude controller on its own. |
| `gym_pybullet_drones/examples/trajectory_tracking.py` | Runs the simulation, prints the RMSE, and saves the results. |
| `gym_pybullet_drones/trajectory/` | The desired circle and diamond trajectories. |
| `gym_pybullet_drones/utils/TrackingLogger.py` | Logging, RMSE, and plots. |
| `gym_pybullet_drones/envs/CtrlAviary.py` | The simulation environment. |
| `gym_pybullet_drones/assets/cf2x.urdf` | The Crazyflie 2.X model and its parameters. |
