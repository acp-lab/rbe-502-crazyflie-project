## Installation

Tested on Intel x64/Ubuntu 22.04.

```sh
git clone https://github.com/acp-lab/rbe-502-crazyflie-project.git

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

## Use

### PD controller implementation
In the control folder, find the `DSLPIDControl.py` file and complete the `_dslPIDPositionControl` function. Then execute the following
```sh
cd gym_pybullet_drones/examples/
python pid_1_drone.py
```
