cmake_minimum_required(VERSION 2.8.3)
project(youbot_trajectory_commander)

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

add_executable(trajectory_commader src/trajectory_commader.cpp)
add_executable(simple_ilqr src/simple_ilqr.cpp)

target_link_libraries(trajectory_commader
  ${catkin_LIBRARIES}
)
target_link_libraries(simple_ilqr
  ${catkin_LIBRARIES}
)

install(TARGETS trajectory_commader trajectory_commader
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS simple_ilqr simple_ilqr
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
