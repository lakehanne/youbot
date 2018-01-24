cmake_minimum_required(VERSION 2.8.3)
project(youbot_navigation)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  move_base
  roscpp
  rospy
  tf
  trajectory_msgs
  roscpp
  actionlib
  nav_msgs
  move_base_msgs
)


## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES youbot_navigation
#  CATKIN_DEPENDS move_base roscpp rospy tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

macro(add_exec name)
  add_executable(${name} src/${name}.cpp)
  target_link_libraries(${name} ${catkin_LIBRARIES})
  install(TARGETS ${name} ${name}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
endmacro()


add_exec(nav_target)