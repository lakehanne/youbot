# include <TH/TH.h>

#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <iostream>
#include <cmath>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>


#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class Receiver
{
private:
  /*aliases*/
  using imageMsgSub = message_filters::Subscriber<sensor_msgs::Image>;
  using irMsgSub   = message_filters::Subscriber<sensor_msgs::Image>;
  using syncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image>;

  bool running, updateImage, updateIr, save;
  size_t counter;
  std::ostringstream oss;

  std::vector<int> params;

  ros::NodeHandle nh;
  std::mutex mutex;
  cv::Mat ir, rgb;
  std::string windowName;
  const std::string basetopic;
  std::string subNameDepth;

  unsigned long const hardware_threads;
  ros::AsyncSpinner spinner;
  std::string subNameRGB, subNameIr;
  imageMsgSub subImageRGB;
  irMsgSub subImageIr;

  std::vector<std::thread> threads;
  message_filters::Synchronizer<syncPolicy> sync;

public:
  //constructor
  Receiver()
  : updateIr(false), updateImage(false), save(false), counter(0),
  windowName("Kinect images"), basetopic("/head_mount_kinect2"),
  hardware_threads(std::thread::hardware_concurrency()),  spinner(hardware_threads/2),
  subNameIr(basetopic + "/ir/image_raw"), subNameRGB(basetopic + "/rgb/image_raw"),
  subImageIr(nh, subNameIr, 1), subImageRGB(nh, subNameRGB, 1),
  sync(syncPolicy(10), subImageIr, subImageRGB)
  {
    sync.registerCallback(boost::bind(&Receiver::callback, this, _1, _2));
    ROS_INFO_STREAM("#Hardware Concurrency: " << hardware_threads <<
      "\t. Spinning with " << hardware_threads/4 << " threads");
    // params.push_back(cv::IMWRITE_JPEG_QUALITY);
    // params.push_back(80);
  }
  //destructor
  ~Receiver()
  {
  }

  Receiver(Receiver const&) =delete;
  Receiver& operator=(Receiver const&) = delete;

  void run()
  {
    begin();
    end();
  }
private:
  void begin()
  {
    if(spinner.canStart())
    {
      spinner.start();
    }
    running = true;
    while(!updateImage || !updateIr)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    //spawn the threads
    // threads.push_back(std::thread(&Receiver::irDisp, this));
    // threads.push_back(std::thread(&Receiver::imageDisp, this));
    // //call join on each thread in turn
    // std::for_each(threads.begin(), threads.end(), \
    //               std::mem_fn(&std::thread::join));
  }

  void end()
  {
    spinner.stop();
    running = false;
  }

  void callback(const sensor_msgs::ImageConstPtr& kinectIr, const sensor_msgs::ImageConstPtr& kinectRGB)
  {
    cv::Mat ir;
    cv::Mat rgb;
    getImage(kinectRGB, rgb);
    getImage(kinectIr, ir);

    std::lock_guard<std::mutex> lock(mutex);
    this->ir = ir;
    this->rgb = rgb;
    updateImage = true;
    updateIr = true;
    ROS_INFO_STREAM("ir: " << ir.size() << " rgb: " << rgb.size());
  }

  void getImage(const sensor_msgs::ImageConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(image);
  }

  void imageDisp()
  {
    cv::Mat rgb;
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 640, 480) ;

    for(; running && ros::ok();)
    {
      if(updateImage)
      {
        std::lock_guard<std::mutex> lock(mutex);
        rgb = this->rgb;
        updateImage = false;

        cv::imshow(windowName, rgb);
      }

      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
        case 27:
        case 'q':
          running = false;
          break;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void irDisp()
  {
    cv::Mat ir;
    cv::namedWindow("kinect ir", cv::WINDOW_NORMAL);
    cv::resizeWindow("kinect ir", 640, 480) ;

    for(; running && ros::ok();)
    {
      if(updateIr)
      {
        std::lock_guard<std::mutex> lock(mutex);
        ir = this->ir;
        updateImage = false;

        cv::imshow("kinect ir", ir);
      }

      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
        case 27:
        case 'q':
          running = false;
          break;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "kinect_viewer_node");

  ROS_INFO_STREAM("Started node " << ros::this_node::getName().c_str());

  Receiver r;
  while (ros::ok())
  	r.run();

  if(!ros::ok())
  {
    return 0;
  }

  ros::shutdown();
}