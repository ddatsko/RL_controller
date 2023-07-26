## This repository contains ROS nodes and scripts for real-world and simulation experiments with a [Reinforcement Learning controller](https://mrs.felk.cvut.cz/gitlab/agile-flight/flightsim)

Inside of `packages` directory, there are two packages that are designed to complement each other:
* `rl_controller` contains a controller module that can be inserted into MRS system `control_manager` as a plugin. It loads the RL policy and creates a service waiting for a waypoint to go to
* `rl_goals_checker` loads the environment file with goals and publishes the next goal to `rl_controller` when the current one is reached. It also implements some failsafe like switching to an emergency controller when a goal is missed or UAV speed is too high


Steps for deploying your trained policy:
* Train your policy, making sure that its input/output is the same as in the `mrs_system_integration` branch of RL controller
* Now you should obtain a `.zip` policy file
* Convert it into a `.pt` file using the `scripts/export_policy.py` e.g. `python export_policy.py -i <zip file> -o <out pt file> -d <cpu/cuda depending if you have torch compatible with cuda>`
* For making things work, the main things to change are:
  * Make custom `control_manager.yaml` config for MRS `control_manager` with following lines added:
  ```
  RLController:
    address: "rl_controller/RLController"
    namespace: "rl_controller"
    eland_threshold: 0.0 # [m], position error triggering eland
    failsafe_threshold: 0.0 # [m], position error triggering failsafe land
    odometry_innovation_threshold: 0.0 # [m], position odometry innovation threshold
    human_switchable: true
  
  rl_controller:
    rl_policy_filename: "<Path to your policy .pt file>"
  ```
  * For `rl_goals_checker,` create a custom config `.yaml` file containing following lines:
  ```
  environment_configuration_file: "<Config .yml file used for policy training. Goals will be taken from it>"
  controller_after_following: "Se3Controller"
  ```

You can find examples of the configuration in `tmux` (for simulation) and `real_world_tmux` (for the real world) directories.

Currently, all the control is done in `local_origin` reference frame. This means that the UAV starts at (0, 0, 0) position, and if the policy was trained with the initial position being at (0, 0, Z) after the takeoff it is enough to call `goto_altitude` service and switch to the `RLController`  
