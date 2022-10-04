#include "aruco_slam/aruco_slam.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

ArucoSlam *g_aruco_loca;

void ImageCallback(const sensor_msgs::ImageConstPtr &img_msg_ptr, const sensor_msgs::CameraInfoConstPtr &camera_information);
void EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &encoder_msg_ptr);
std::pair<cv::Mat, cv::Mat> parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &camera_information);
inline void getTransformStamped(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transform_stamped);
struct ArucoSlamIniteData getArucoSlamIniteData(const ros::NodeHandle &nh);

ros::Publisher g_real_map_pub;// real accurate markers
ros::Publisher g_detected_map_pub;// detected map
ros::Publisher g_detected_markers_pub;//markers detected now
ros::Publisher g_robot_pose_pub;
image_transport::Publisher g_img_pub;

tf2_ros::Buffer g_tfBuffer_;

int main(int argc, char **argv)
{
    /***** ROS init *****/
    ros::init(argc, argv, "my_node_name");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::ImageTransport it_sub(nh);
    tf2_ros::TransformListener tfListener(g_tfBuffer_);
    
    /***** deal with parameters *****/
    std::string image_topic_name, encoder_topic_name, real_map_file;
    nh.getParam("/aruco_slam_node/topic/image", image_topic_name);
    nh.getParam("/aruco_slam_node/topic/encoder", encoder_topic_name);
    nh.getParam("/aruco_slam_node/map/map_file", real_map_file);
    ROS_INFO("map_f selecteed: %s \n", real_map_file.c_str());
    
    /***** Pubisher Init *****/
    g_real_map_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/real_map", 1, true);
    g_detected_markers_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/detected_markers", 1, false);
    g_detected_map_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/detected_map", 1, false);
    g_robot_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("aruco_slam_node/pose", 1);
    g_img_pub = it.advertise("aruco_slam_node/image", 1);
    
    /***** Subscriber Init *****/
    image_transport::CameraSubscriber img_sub_ = it_sub.subscribeCamera(image_topic_name, 1, ImageCallback);
    ros::Subscriber encoder_sub = nh.subscribe(encoder_topic_name, 1, EncoderCallback);
    
    /***** ArucoSlam init*****/
    g_aruco_loca = new ArucoSlam(getArucoSlamIniteData(nh));
    g_aruco_loca->loadMap(real_map_file);
    g_real_map_pub.publish(g_aruco_loca->get_real_map());
    ROS_INFO("\n\n\n\nROS_NODE Initialed \n\n\n\n\n");
    
    /***** SYSTEM START *****/
    ros::spin();

    delete g_aruco_loca;
    return 0;
}

void ImageCallback(const sensor_msgs::ImageConstPtr &img_ptr, const sensor_msgs::CameraInfoConstPtr &cinfo)
{
    g_aruco_loca->setcameraparameters(parseCameraInfo(cinfo));
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img_ptr, "bgr8");
    
    /***** add image *****/
    g_aruco_loca->addImage(cv_ptr->image);
    
    /***** pubish marked image*****/
    cv::Mat img = g_aruco_loca->markedImg();
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    g_img_pub.publish(msg);

    g_detected_markers_pub.publish(g_aruco_loca->get_detected_markers());
    g_detected_map_pub.publish(g_aruco_loca->get_detected_map());
}

void EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &en_ptr)
{

    std::vector<float> encoder_data = en_ptr->data;
    double enl = encoder_data.at(0);
    double enr = encoder_data.at(1);

    /***** add encoder data *****/
    g_aruco_loca->addEncoder(enl, enr);

    /* publish  robot pose */
    g_robot_pose_pub.publish(g_aruco_loca->toRosPose());
}

std::pair<cv::Mat, cv::Mat> parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &cinfo)
{   
    cv::Mat matrix = cv::Mat::zeros(3, 3, CV_64F);
    // cv::Mat matrix(cinfo->K, true); // can't be used for boots::array
    cv::Mat dist(cinfo->D, true);
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            matrix.at<double>(i, j) = cinfo->K[3 * i + j];
    return std::pair<cv::Mat, cv::Mat>{matrix, dist};
}

inline void getTransformStamped(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transform_stamped)
{
    try
    {
        transform_stamped = g_tfBuffer_.lookupTransform(target_frame, source_frame,
                                                     ros::Time(0), ros::Duration(0.1));
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
        ros::Duration(1.0).sleep();
    }
}

struct ArucoSlamIniteData getArucoSlamIniteData(const ros::NodeHandle &nh){
    struct ArucoSlamIniteData inite_data;
    nh.getParam("/aruco_slam_node/odom/kl", inite_data.kl);
    nh.getParam("/aruco_slam_node/odom/kr", inite_data.kr);
    nh.getParam("/aruco_slam_node/odom/b", inite_data.b);
    nh.getParam("/aruco_slam_node/covariance/k", inite_data.k); // encoder error coefficient
    nh.getParam("/aruco_slam_node/covariance/k_r", inite_data.k_r);
    nh.getParam("/aruco_slam_node/covariance/k_phi", inite_data.k_phi);
    nh.getParam("/aruco_slam_node/aruco/markers_dictionary", inite_data.markers_dictionary);
    nh.getParam("/aruco_slam_node/aruco/marker_length", inite_data.marker_length);
    getTransformStamped("base_link", "camera_frame_optical", inite_data.transformStamped_r2c);
    return inite_data;
}