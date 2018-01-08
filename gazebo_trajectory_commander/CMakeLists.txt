cmake_minimum_required(VERSION 2.8.3)
project(gazebo_trajectory_commander)

find_package(catkin REQUIRED COMPONENTS
  brics_actuator
  geometry_msgs
  trajectory_msgs
  roscpp
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(gazebo_trajectory_commander src/gazebo_trajectory_commander.cpp)
add_executable(simple_ilqr src/simple_ilqr.cpp)

target_link_libraries(gazebo_trajectory_commander
  ${catkin_LIBRARIES}
)
target_link_libraries(simple_ilqr
  ${catkin_LIBRARIES}
)

install(TARGETS gazebo_trajectory_commander gazebo_trajectory_commander
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS simple_ilqr simple_ilqr
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
