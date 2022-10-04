#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>

#include <aruco_slam/aruco_slam.h>

ArucoSlam *aruco_loca;

void ImageCallback(const sensor_msgs::ImageConstPtr &img_ptr, const sensor_msgs::CameraInfoConstPtr &cinfo);
void EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &en_ptr);
static void parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &cinfo, cv::Mat &matrix, cv::Mat &dist);
inline void gettransform(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transformStamped);

ros::Publisher g_map_landmark_pub;
ros::Publisher g_detedted_landmark_pub;
ros::Publisher g_detedted_map_pub;
ros::Publisher g_robot_pose_pub;
image_transport::Publisher g_img_pub;

tf2_ros::Buffer tfBuffer_;
cv::Mat dist_coeffs;
cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64F);

int main(int argc, char **argv)
{
    /***** 初始化ROS *****/
    
    ros::init(argc, argv, "my_node_name");
    
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::ImageTransport it_sub(nh);
    tf2_ros::TransformListener tfListener(tfBuffer_);
    /***** 获取参数 *****/
    /* TODO 错误处理 */
    std::string image_topic_name, encoder_topic_name, map_f;
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    double kl, kr, b;
    // Eigen::Matrix4d T_r_c;
    geometry_msgs::TransformStamped transformStamped_r2c;
    double k, k_r, k_phi;
    int markers_dictionary;
    double marker_length;

    nh.getParam("/aruco_slam_node/topic/image", image_topic_name);
    nh.getParam("/aruco_slam_node/topic/encoder", encoder_topic_name);

    nh.getParam("/aruco_slam_node/camera/fx", fx);
    nh.getParam("/aruco_slam_node/camera/fy", fy);
    nh.getParam("/aruco_slam_node/camera/cx", cx);
    nh.getParam("/aruco_slam_node/camera/cy", cy);
    nh.getParam("/aruco_slam_node/camera/k1", k1);
    nh.getParam("/aruco_slam_node/camera/k2", k2);
    nh.getParam("/aruco_slam_node/camera/p1", p1);
    nh.getParam("/aruco_slam_node/camera/p2", p2);
    nh.getParam("/aruco_slam_node/camera/k3", k3);
    cv::Mat K = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat dist = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);

    nh.getParam("/aruco_slam_node/odom/kl", kl);
    nh.getParam("/aruco_slam_node/odom/kr", kr);
    nh.getParam("/aruco_slam_node/odom/b", b);

    // std::vector<double> Trc;
    // nh.getParam("/aruco_slam_node/extrinsic/Trc", Trc);
    // T_r_c << Trc[0], Trc[1], Trc[2], Trc[3],
    //     Trc[4], Trc[5], Trc[6], Trc[7],
    //     Trc[8], Trc[9], Trc[10], Trc[11],
    //     0.0, 0.0, 0.0, 1.0;

    nh.getParam("/aruco_slam_node/covariance/k", k);//encoder error coefficient
    nh.getParam("/aruco_slam_node/covariance/k_r", k_r);
    nh.getParam("/aruco_slam_node/covariance/k_phi", k_phi);

    nh.getParam("/aruco_slam_node/aruco/markers_dictionary", markers_dictionary);
    nh.getParam("/aruco_slam_node/aruco/marker_length", marker_length);

    nh.getParam("/aruco_slam_node/map/map_file", map_f);
    ROS_INFO("map_f selecteed: %s \n", map_f.c_str());

    gettransform("base_link","camera_frame_optical",transformStamped_r2c);
    


    /***** 初始化 Aruco EKF SLAM *****/
    aruco_loca = new ArucoSlam(K, dist, kl, kr, b, transformStamped_r2c, k, k_r, k_phi, markers_dictionary, marker_length);
    
    /***** 初始化消息发布 *****/
    g_map_landmark_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/map_landmark", 1, true);
    g_detedted_landmark_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/deteected_landmark", 1, false);
    g_detedted_map_pub = nh.advertise<visualization_msgs::MarkerArray>("aruco_slam_node/deteected_map", 1, false);
    g_robot_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("aruco_slam_node/pose", 1);
    g_img_pub = it.advertise("aruco_slam_node/image", 1);
    ROS_INFO("\n\n\n\nROS_NODE Initialed \n\n\n\n\n");
    /***** 初始化消息订阅 *****/
    image_transport::CameraSubscriber img_sub_ = it_sub.subscribeCamera(image_topic_name, 1, ImageCallback);
    ros::Subscriber encoder_sub = nh.subscribe(encoder_topic_name, 1, EncoderCallback);
    
    aruco_loca->loadMap(map_f);
    g_map_landmark_pub.publish(aruco_loca->get_mapmarkerarray());
    std::cout << "\n\nSYSTEM START \n\n";
    ros::spin();

    delete aruco_loca;
    return 0;
}

// void ImageCallback ( const sensor_msgs::ImageConstPtr& img_ptr )
void ImageCallback(const sensor_msgs::ImageConstPtr &img_ptr, const sensor_msgs::CameraInfoConstPtr &cinfo)
{
    //     ROS_INFO("0");
    parseCameraInfo(cinfo, camera_matrix, dist_coeffs);
    aruco_loca->setcameraparameters(camera_matrix, dist_coeffs);
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img_ptr, "bgr8");
    //     ROS_INFO("1");
    //     /* add image */
    aruco_loca->addImage(cv_ptr->image);
    // cv::Mat image = cv_ptr->image;

    // 	std::vector<int> ids;
    // 	std::vector<std::vector<cv::Point2f>> corners, rejected;
    // 	std::vector<cv::Vec3d> rvecs, tvecs;
    // 	std::vector<cv::Point3f> obj_points;
    // 	cv::aruco::detectMarkers(image, aruco_loca->dictionary_, corners, ids, parameters_, rejected);
    //    ROS_INFO("2");
    //     /* publish markerd image */
    cv::Mat img = aruco_loca->markedImg();
    //     ROS_INFO("3");
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    //    ROS_INFO("4");
    g_img_pub.publish(msg);
    g_detedted_landmark_pub.publish(aruco_loca->get_detectedmarkerarray());
    g_detedted_map_pub.publish(aruco_loca->detectedMAParray_);
}

void EncoderCallback(const std_msgs::Float32MultiArray::ConstPtr &en_ptr)
{

    std::vector<float> encoder_data = en_ptr->data;
    double enl = encoder_data.at(0);
    double enr = encoder_data.at(1);
    // ROS_INFO("enl:%lf, enr:%lf",enl, enr);

    aruco_loca->addEncoder(enl, enr);

    /* publish  robot pose */
    geometry_msgs::PoseWithCovarianceStamped pose = aruco_loca->toRosPose(); // pose的协方差在rviz也做了放大
    g_robot_pose_pub.publish(pose);
}

static void parseCameraInfo(const sensor_msgs::CameraInfoConstPtr &cinfo, cv::Mat &matrix, cv::Mat &dist)
{
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            matrix.at<double>(i, j) = cinfo->K[3 * i + j];
    dist = cv::Mat(cinfo->D, true);
}

inline void gettransform(const std::string &target_frame, const std::string &source_frame, geometry_msgs::TransformStamped &transformStamped){
    try
    {
        transformStamped = tfBuffer_.lookupTransform(target_frame, source_frame,
                                                    ros::Time(0),ros::Duration(0.1));
        // Eigen::Affine3d transform_matrix = tf2::transformToEigen(transformStamped);
        // Eigen::Matrix4d T_r_c = transform_matrix.matrix();

    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s", ex.what());
        ros::Duration(1.0).sleep();
    }
}