

### Overview

Most of the codes here, particularly for [youbot_navigation](/youbot_navigation) are an implementation of my IROS 2018 minimax iterative dynamic game submission.

The following website describes the results of the paper and the videos of experiments. 

[Minimax Iterative Dynamic Game: Application to Nonlinear Robot Control Tasks](http://ecs.utdallas.edu/~opo140030/iros18/iros2018.html#/)

I am currently working on implementing the algorithm with policy search methods on the kuka youbot platform.

### Docker Image
Everything is contained in the docker image at the following tag, `iros18_submission`:

[Docker Image](hubs.docker.com/r/lakehanne)

Please pull the image like so:

```
	docker pull lakehanne/youbotbuntu14:iros18_submission
```

### Running on a native Linux distro

* Gazebo model
  * `./run-gazebo-model`
  * This launches the world, robot and the orange square obstacle


* Run ILQG/DDP Trajectory optimization
	* `./run_trajopt`
	* This launches the DPP/ILQG algorithm used in navigating the robot from the start pose to the orange square box. Also launches the sensor nodes


#### For Adaptive Monte Carlo Only

Old code that uses the adaptive monte carlo localization algorithm in navigating the robot in the cartesian space of the inertial frame. This is based on Sebastian Thrun et. al's Probabilistic Robotics book.


* AMCL 
	* `./run-navstack`

* ActionLib Example
	* `./run-goal-nav`	
