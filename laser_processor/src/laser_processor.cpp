#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <iostream>
#include <mutex>

#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <tf/transform_listener.h>
#include <laser_geometry/laser_geometry.h>

// for publishing point clouds
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>


using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using pcl_viz = pcl::visualization::PCLVisualizer;

class laser_clouds
{
private:
  laser_geometry::LaserProjection projector_;
  tf::TransformListener listener_;
  ros::Publisher pub;
  ros::NodeHandle n_laser_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud;

  bool updateCloud;
  bool spill;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

public:
  laser_clouds(ros::NodeHandle n_laser, bool spill);
  ~laser_clouds(){}
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerCreator();
  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                              void* viewer_void);
  //high fidelity projection
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg )
  {
    if(!listener_.waitForTransform(msg->header.frame_id, \
                  "/base_link", msg->header.stamp +\
                  ros::Duration().fromSec(msg->ranges.size()* \
                  msg->time_increment), ros::Duration(1.0)))
    {
       return;
    }

    pub = n_laser_.advertise<PointCloudT>("laser_cloud", 4);

    sensor_msgs::PointCloud cloud;
    PointCloudT::Ptr cloud_msg (new (PointCloudT)); // for publishing
    cloud_msg->header.frame_id = "laser_cloud";
    cloud_msg->header.stamp = ros::Time::now().toNSec();
    cloud_msg->height = cloud_msg->width = 1;
    projector_.transformLaserScanToPointCloud("/base_link", *msg, cloud,listener_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud (new pcl::PointCloud<pcl::PointXYZ>);
    readCloud(cloud, pclCloud, cloud_msg);    
    ros::Rate looper(30);

    // if(ros::ok() && !viewer->wasStopped())
    if(ros::ok())
    {            
      // viewer->setSize(400, 400);
      // viewer->addPointCloud<pcl::PointXYZ> (this->pclCloud, "laser_cloud");     
      // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "laser_cloud"); 
      // viewer->spinOnce(10);
      // boost::this_thread::sleep(boost::posix_time::microseconds(100));

      if(updateCloud)
      {      
        // viewer->removePointCloud("laser_cloud");
        // viewer->updatePointCloud(this->pclCloud, "laser_cloud");
        pub.publish(cloud_msg);
      }
      looper.sleep();
      pub.publish(cloud_msg);
    }
    updateCloud = true;
  }

  void readCloud(const sensor_msgs::PointCloud& sensorCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud, PointCloudT::Ptr cloud_msg)
  {
    pcl::PointXYZ points;

    for(auto it=sensorCloud.points.begin(); it!=sensorCloud.points.end(); ++it )
    {      
      points.x = (*it).x;
      points.y = (*it).y;
      points.z = (*it).z;
      pclCloud->points.push_back(points);
      // populate publishable cloud msgs
      cloud_msg->points.push_back(points);

      if(spill)
        ROS_INFO("Laser Points: x: %.4f, y: %.4f, z: %.4f", points.x, points.y, points.z);
    }  
    this->pclCloud = pclCloud;
  }
};

laser_clouds::laser_clouds(ros::NodeHandle n_laser, bool spill)
  : updateCloud(false), n_laser_(n_laser), spill(spill){
    // viewer = laser_clouds::viewerCreator();
    pub = n_laser_.advertise<PointCloudT>("laser_cloud", 1);
  }

boost::shared_ptr<pcl::visualization::PCLVisualizer> laser_clouds::viewerCreator()
{    
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Laser Scans 2D"));
  viewer->setSize(400, 400);
  viewer->setBackgroundColor  (0.2, 0.3, 0.3);
  viewer->addCoordinateSystem (1.0);    //don't want me no cylinder
  viewer->initCameraParameters ();
  viewer->registerKeyboardCallback (&laser_clouds::keyboardEventOccurred, *this);
  return viewer;
}

unsigned int text_id = 0;
void laser_clouds::keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i)
    {
      sprintf (str, "text#%03d", i);
      viewer->removeShape (str);
    }
    text_id = 0;
  }
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "laser_scans");

  bool spill(false);
	ros::NodeHandle n_laser;
  laser_clouds ls(n_laser, spill);
	ros::Subscriber sub = n_laser.subscribe("/scan", 1000, &laser_clouds::laserCallback, &ls);

	ros::spin();

	return 0;
}
