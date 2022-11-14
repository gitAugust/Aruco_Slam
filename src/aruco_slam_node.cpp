#include "aruco_slam/aruco_slam.h"
#include "aruco_slam/map_loader.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

class ArucoSlamRosNode
{
public:
    ArucoSlamRosNode(const ros::NodeHandle &nh)
    {
        ArucoSlamIniteData initedata = parseArucoSlamIniteData(nh);
        aruco_slam_ptr_ = new ArucoSlam(initedata);
    }
    void ImageCallback(const sensor_msgs::ImageConstPtr &img_msg_ptr, const sensor_msgs::CameraInfoConstPtr &camera_information);
    void EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &encoder_msg_ptr);
    std::pair<cv::Mat, cv::Mat> parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &camera_information);
    void getTransformStamped(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transform_stamped);
    struct ArucoSlamIniteData parseArucoSlamIniteData(const ros::NodeHandle &nh);

private:
    ArucoSlam *aruco_slam_ptr_;
    bool camera_inited_ = false;
};

ros::Publisher g_real_map_pub;          /**< publish real accurate markers */
ros::Publisher g_detected_map_pub;      /**< publish detected map */
ros::Publisher g_detected_markers_pub;  /**< publish markers detected now */
ros::Publisher g_robot_pose_pub;
image_transport::Publisher g_img_pub;   /**< publish detected image */

tf2_ros::Buffer g_tfBuffer_;            /**< get robot parameters */

int main(int argc, char **argv)
{
    /***** ROS init *****/
    ros::init(argc, argv, "my_node_name");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh); /**< Image subscribe and publish */
    tf2_ros::TransformListener tfListener(g_tfBuffer_);

    /***** deal with parameters *****/
    std::string image_topic_name, encoder_topic_name;
    nh.getParam("/aruco_slam_node/topic/image", image_topic_name);
    nh.getParam("/aruco_slam_node/topic/encoder", encoder_topic_name);

    /***** Pubisher Init *****/
    g_real_map_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/real_map", 1, true);
    g_detected_markers_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/detected_markers", 1, false);
    g_detected_map_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/detected_map", 1, false);
    g_robot_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("aruco_slam_node/pose", 1);
    g_img_pub = it.advertise("aruco_slam_node/image", 1);

    /** load real accurate map **/
    std::string real_map_file;
    if (nh.getParam("/aruco_slam_node/map/map_file", real_map_file))
    {
        MapLoader *map_loader_ptr = new MapLoader(real_map_file);
        g_real_map_pub.publish(map_loader_ptr->toRosRealMapMarkers());
        delete map_loader_ptr;
    }

    /***** ArucoSlam init*****/
    ArucoSlamRosNode ros_aruco_slam(nh);

    /***** Subscriber Init *****/
    image_transport::CameraSubscriber img_sub_ = it.subscribeCamera(image_topic_name, 1, &ArucoSlamRosNode::ImageCallback, &ros_aruco_slam);
    ros::Subscriber encoder_sub = nh.subscribe(encoder_topic_name, 5, &ArucoSlamRosNode::EncoderCallback, &ros_aruco_slam);

    ROS_INFO("\n\n\n\nROS_NODE Initialed \n\n\n\n\n");

    /***** SYSTEM START *****/
    ros::spin();

    ros::shutdown();
    return 0;
}

void ArucoSlamRosNode::ImageCallback(const sensor_msgs::ImageConstPtr &img_ptr, const sensor_msgs::CameraInfoConstPtr &cinfo)
{
    if (!camera_inited_)
    {
        aruco_slam_ptr_->setCameraParameters(parseCameraInfo(cinfo));
        camera_inited_ = true;
    }

    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img_ptr, "bgr8");

    /***** add image *****/
    aruco_slam_ptr_->addImage(cv_ptr->image);

    /***** pubish marked image*****/
    cv::Mat img = aruco_slam_ptr_->getMarkedImg();
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    g_img_pub.publish(msg);

    g_detected_markers_pub.publish(aruco_slam_ptr_->toRosDetectedMarkers());
    g_detected_map_pub.publish(aruco_slam_ptr_->toRosMappedMarkers());
}

void ArucoSlamRosNode::EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &en_ptr)
{
    std::vector<float> encoder_data = en_ptr->data;
    double enl = encoder_data.at(0);
    double enr = encoder_data.at(1);

    /***** add encoder data *****/
    aruco_slam_ptr_->addEncoder(enl, enr);

    /* publish  robot pose */
    // ROS_INFO_STREAM("g_aruco_loca->sigma:"<<g_aruco_loca->sigma());
    g_robot_pose_pub.publish(aruco_slam_ptr_->toRosPose());
}

std::pair<cv::Mat, cv::Mat> ArucoSlamRosNode::parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &cinfo)
{
    cv::Mat matrix = cv::Mat::zeros(3, 3, CV_64F);
    // cv::Mat matrix(cinfo->K, true); // can't be used for boots::array
    cv::Mat dist(cinfo->D, true);
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            matrix.at<double>(i, j) = cinfo->K[3 * i + j];
    return std::pair<cv::Mat, cv::Mat>{matrix, dist};
}

void ArucoSlamRosNode::getTransformStamped(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transform_stamped)
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

struct ArucoSlamIniteData ArucoSlamRosNode::parseArucoSlamIniteData(const ros::NodeHandle &nh)
{
    struct ArucoSlamIniteData inite_data;
    nh.getParam("/aruco_slam_node/odom/kl", inite_data.kl);
    nh.getParam("/aruco_slam_node/odom/kr", inite_data.kr);
    nh.getParam("/aruco_slam_node/odom/b", inite_data.b);
    nh.getParam("/aruco_slam_node/covariance/Q_k", inite_data.Q_k); // encoder error coefficient
    nh.getParam("/aruco_slam_node/covariance/R_x", inite_data.R_x); // encoder error coefficient
    nh.getParam("/aruco_slam_node/covariance/R_y", inite_data.R_y); // encoder error coefficient
    nh.getParam("/aruco_slam_node/covariance/R_theta", inite_data.R_theta); // encoder error coefficient
    nh.getParam("/aruco_slam_node/aruco/markers_dictionary", inite_data.markers_dictionary);
    nh.getParam("/aruco_slam_node/aruco/marker_length", inite_data.marker_length);
    std::string robot_frame_base, camera_frame_optical;
    nh.getParam("/aruco_slam_node/frame/robot_frame_base", robot_frame_base);
    nh.getParam("/aruco_slam_node/frame/camera_frame_optical", camera_frame_optical);
    nh.getParam("/aruco_slam_node/const/USEFUL_DISTANCE_THRESHOLD_", inite_data.USEFUL_DISTANCE_THRESHOLD);
    
    getTransformStamped(robot_frame_base, camera_frame_optical, inite_data.transformStamped_r2c);
    return inite_data;
}
