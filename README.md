# DLO_Singularity_Mocap


__The simulation is a closed loop experiment of a deformable rod detected by three markers with the Qualisys mocap system. The goal is to drive this rod into a objective position and validate the estimation of the jacobian computed in the initial position.__

#### Steps of Simulation:


> ##### Set the target position of the rod
> 
> python3 record_s_target.pyç


> ##### Move the robots into the initial configuration


> ##### Compute the initial jacobian estimation
> 
> python3 jacobian_estimator_dual.py --delta_t 0.025 --delta_r 0.1 --home_xy_tolerance 0.012 --feature_return_tolerance 0.009 --max_return_extension 15


> ##### Run ibvs controller
>
> python3 ibvs_controller.py
