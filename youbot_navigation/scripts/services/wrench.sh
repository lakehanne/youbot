#! /bin/bash

rosservice call /gazebo/apply_body_wrench '{body_name: "youbot::arm_link_2" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'

rosservice call /gazebo/clear_body_wrenches '{body_name: "youbot::arm_link_2"}'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_fl" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_fr" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_bl" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'
# rosservice call /gazebo/apply_body_wrench '{body_name: "wheel_link_br" , wrench: { torque: { x: 40, y: 40, z: 0 }, force: { x: 40, y: 40 , z: 0 }  }, start_time: 10000000000, duration: 1000000000 }'