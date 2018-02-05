rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: -0.0, y: 0, z: 0}}'

# apply force wrench
# see http://gazebosim.org/tutorials/?tut=ros_comm#Services:Forcecontrol
rosservice call /gazebo/apply_body_wrench '{body_name: "youbot::base_footprint" , wrench: { torque: { x: 0.1, y: 0 , z: 0 } }, start_time: 10000000000, duration: 1000000000 }'

# apply joint effots to the four wheel joints


rosservice call /gazebo/apply_joint_effort "joint_name: 'youbot::arm_joint_1'
effort: 10.0
start_time:
  secs: 0
  nsecs: 0
duration:
  secs: 40
  nsecs: 0"