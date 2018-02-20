#! /bin/bash

# rosservice call /gazebo/apply_body_wrench '{body_name: "youbot::arm_link_2" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/clear_body_wrenches '{body_name: "youbot::arm_link_2"}'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_fl" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_fr" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_bl" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "base_footprint" , wrench: { torque: { x: 0, y: 0, z: 0 }, force: { x: 0, y: 0 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rostopic pub cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0., y: 0, z: 0} }'
rosrun gazebo_ros spawn_model -file `rospack find youbot_description`/urdf/youbot_obstacle/obstacle.urdf  -urdf -x 1.3 -y 0.8 -z 1.0 -model box -robot_namespace youbot
# reset all gazebo spawned
ps ax | grep youbot.launch | awk '{print $1}' | xargs kill -9
rosservice call /gazebo/set_model_state '{model_state: "youbot", pose: {position: {x: 0, y: 0} } }'
