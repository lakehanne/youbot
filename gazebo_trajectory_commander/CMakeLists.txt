cmake_minimum_required(VERSION 2.8.3)
project(gazebo_trajectory_commander)

find_package(catkin REQUIRED COMPONENTS
  brics_actuator
  geometry_msgs
  trajectory_msgs
  roscpp
  actionlib
  move_base_msgs
)

catkin_package()

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

add_exec(simple_ilqr)
#add_exec(nav_target)