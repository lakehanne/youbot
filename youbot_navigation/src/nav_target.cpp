/*
<<<<<<< HEAD
[Y, P, R] = [0.000000, 0.000002, -0.000002] // from gazebo
[x, y, z] = [0.00,  0.000001, 0.149996]
*/

#include <thread>
#include <vector>
#include <memory>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <tf/transform_datatypes.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
=======
 <origin xyz="${5*0.6} -0.15 0.0" rpy="0 0 0"/>
*/

#include <ros/ros.h>
>>>>>>> 079dcb22473abff3b1987741d111c4207d58aa69
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

<<<<<<< HEAD
class NavTarget
{
public:
  NavTarget(ros::NodeHandle nh)
  : nh_(nh), hardware_threads_(std::thread::hardware_concurrency()), 
  spinner_(hardware_threads_/2)
  {
    odom_subscriber_ = nh_.subscribe("/odom", 1000, \
                                          &NavTarget::odom_callback, this);
    goal_quat_.setEuler(0.000000, 0.000002, -0.000002);
  }

  ~NavTarget()
  {

  }

  NavTarget(NavTarget const&) = delete;
  NavTarget& operator=(NavTarget const&) = delete;

  void run()
  {
    begin();
    end();
  }

private:
  void begin()
  {
    if(spinner_.canStart())
      spinner_.start();

    running_ = true;

    threads_vector_.push_back(std::thread(&NavTarget::move_to_goal_pose, this));
    std::for_each(threads_vector_.begin(), threads_vector_.end(), 
                  std::mem_fn(&std::thread::join));
    // move_to_start_pose();
    // move_to_goal_pose();
  }

  void end()
  {
    spinner_.stop();
    running_ = false;
  }

  void odom_callback(const nav_msgs::Odometry& odom_msg)
  {
    robot_pose_ = odom_msg.pose.pose;
    robot_twist_ = odom_msg.twist.twist;
  }

  void move_to_start_pose()
  {
    MoveBaseClient ac("move_base", true);

    //wait for the action server to come up
    while(!ac.waitForServer(ros::Duration(5.0))){
      ROS_INFO("Waiting for the move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal start_pose;
    start_pose.target_pose.header.frame_id = "base_link";
    start_pose.target_pose.header.stamp = ros::Time::now();


    //these values were retrieved by drivin the robot to the start pose at the left hand corner of the setup
    start_pose.target_pose.pose.position.x = -0.951560942553;
    start_pose.target_pose.pose.position.y = -0.792228574408;
    start_pose.target_pose.pose.position.z =  0.0;
    start_pose.target_pose.pose.orientation.x =  2.47502744851e-06;
    start_pose.target_pose.pose.orientation.y = -1.10358239184e-05;
    start_pose.target_pose.pose.orientation.z = -0.0306063933736;
    start_pose.target_pose.pose.orientation.w = -0.999531514539;


    ac.sendGoal(start_pose);

    ac.waitForResult();

    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_INFO("Base moved to start position");
    else
      ROS_INFO("Base failed to move to start position for some reason");

  }

  void move_to_goal_pose()
  {

    //tell the action client that we want to spin a thread by default
    MoveBaseClient ac("move_base", true);

    //wait for the action server to come up
    while(!ac.waitForServer(ros::Duration(5.0))){
      ROS_INFO("Waiting for the move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal goal_;

    //we'll send a goal to the robot to move 1 meter forward
    goal_.target_pose.header.frame_id = "base_link";
    goal_.target_pose.header.stamp = ros::Time::now();

    goal_.target_pose.pose.position.x = 0.736239556594;//0.00;
    goal_.target_pose.pose.position.y = 0.803042138661;//0.000001;
    goal_.target_pose.pose.position.z = 0.0;  

    goal_.target_pose.pose.orientation.x =  -3.18874459879e-06;
    goal_.target_pose.pose.orientation.y = -1.08345476892e-05;
    goal_.target_pose.pose.orientation.z = 0.0432270282388;
    goal_.target_pose.pose.orientation.w = -0.999065275096;

    // tf::quaternionTFToMsg(goal_quat_, goal_.target_pose.pose.orientation);

    // ROS_INFO("Going to position [%.4f], [%.4f], [%.4f] ", goal_.target_pose.pose.position.x, 
    //           goal_.target_pose.pose.position.y, goal_.target_pose.pose.position.z);
    // ROS_INFO("Going to orientation [%.4f], [%.4f], [%.4f] ", goal_.target_pose.pose.orientation.x, 
    //           goal_.target_pose.pose.orientation.y, goal_.target_pose.pose.orientation.z);

    ROS_INFO("Sending robot to goal position");
    ac.sendGoal(goal_);

    ac.waitForResult();

    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_INFO("Hooray, the base moved to target");
    else
      ROS_INFO("The base failed to move to target for some reason");
  }

private:
  unsigned long const hardware_threads_;
  ros::Subscriber odom_subscriber_;
  geometry_msgs::Pose robot_pose_;
  geometry_msgs::Twist robot_twist_;
  ros::NodeHandle nh_;
  ros::AsyncSpinner spinner_;
  bool running_;
  std::vector<std::thread> threads_vector_;
  tf::Quaternion goal_quat_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "navigation_goals");

  ros::NodeHandle nh;

  NavTarget nav_target(nh);
  nav_target.run();

  if (!ros::ok())
    return 0;

  ros::shutdown();
=======
int main(int argc, char** argv){
  ros::init(argc, argv, "simple_navigation_goals");

  //tell the action client that we want to spin a thread by default
  MoveBaseClient ac("move_base", true);

  //wait for the action server to come up
  while(!ac.waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  move_base_msgs::MoveBaseGoal goal;

  //we'll send a goal to the robot to move 1 meter forward
  goal.target_pose.header.frame_id = "base_link";
  goal.target_pose.header.stamp = ros::Time::now();

  goal.target_pose.pose.position.x = 5*0.6;
  goal.target_pose.pose.position.y = -0.15;
  goal.target_pose.pose.orientation.w = 1.0;

  ROS_INFO("Sending goal");
  ac.sendGoal(goal);

  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Hooray, the base moved 1 meter forward");
  else
    ROS_INFO("The base failed to move forward 1 meter for some reason");

  return 0;
>>>>>>> 079dcb22473abff3b1987741d111c4207d58aa69
}