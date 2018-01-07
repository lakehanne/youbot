#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <string>


using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using pcl_viz = pcl::visualization::PCLVisualizer;


class LaserScansProcessor
{
public:
	LaserScansProcessor()
	{

	}

	~LaserScansProcessor()
	{

	}

	void callback(const sensor_msgs::LaserScanConstPtr& laser_msg)
	{

	}

	/* data */

private:
	std::string laser_topic_;
	sensor_msgs::LaserScan scan_msgs_;
};