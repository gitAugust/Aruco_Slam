#ifndef ARUCO_SLAM_H
#define ARUCO_SLAM_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Pose.h>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <tf2_eigen/tf2_eigen.h>

void fillTransform(tf2::Transform &transform_, const cv::Vec3d &rvec, const cv::Vec3d &tvec);

class Observation
{
public:
    Observation() {}
    Observation(const int &aruco_id, const double &x, const double &y, const Eigen::Matrix2d &covariance) : aruco_id_(aruco_id), x_(x), y_(y), covariance_(covariance) {}
    Eigen::Matrix2d covariance_;
    int aruco_id_;
    double x_;
    double y_;
}; // class Observation

class ArucoSlam
{
public:
    ArucoSlam(const cv::Mat &K, const cv::Mat &dist,
                      const double &kl, const double kr, const double &b,
                      const geometry_msgs::TransformStamped &transformStamped_r2c,
                      const double &k, const double &k_r, const double k_phi,
                      const int &markers_dictionary, const double &marker_length);
    /*parameters
    cv::Mat         K
    cv::Mat         dist
    double          kl, kr           小车左右轮的半径
    double          b                小车底盘的一半距离(m)
    Eigen::Matrix4d T_r_c            相机与机器人的相对位姿 机器人->相机
    geometry_msgs::TransformStamped transformStamped_r2c 相机与机器人的相对位姿 机器人->相机
    double          k
    double          k_r             相机观测误差和距离的系数
    double          k_phi           相机观测误差和角度的系数
    int             DICTIONARY_NUM  Opencv aruco 预定以字典的编号
    double          marker_length   Aruco markers的大小(m)
    */
    void addEncoder(const double &el, const double &er); //加入编码器数据进行运动更新
    void addImage(const cv::Mat &img);                   // 加入图像数据进行观测更新
    void loadMap(std::string filename);

    visualization_msgs::MarkerArray toRosMarkers(double scale); //将路标点转换成ROS的marker格式，用于发布显示
    geometry_msgs::PoseWithCovarianceStamped toRosPose();       //将机器人位姿转化成ROS的pose格式，用于发布显示

    Eigen::MatrixXd &mu() { return mu_; }
    Eigen::MatrixXd &sigma() { return sigma_; }
    cv::Mat markedImg() { return marker_img_; }

    visualization_msgs::MarkerArray detectedMAParray_;
    visualization_msgs::MarkerArray get_mapmarkerarray() { return mapmarkerarray_; }
    visualization_msgs::MarkerArray get_detectedmarkerarray() { return detectedmarkerarray_; }
    void setcameraparameters(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs)
    {
        camera_matrix_ = camera_matrix;
        dist_coeffs_ = dist_coeffs;
    }

private:
    int getObservations(const cv::Mat &img, std::vector<Observation> &obs);
    void normAngle(double &angle);
    bool checkLandmark(const int &aruco_id, int &landmark_idx);
    void clearMarkers();
    std::map<int, int> aruco_id_map; //pair<int, int>{aruco_id, position_i}

    bool CYLINDERmarker_generate(int id, double x, double y, double z, visualization_msgs::Marker &marker_, std_msgs::ColorRGBA color, ros::Duration lifetime);
    bool marker_generate(int id, double length, double x, double y, double z, tf2::Quaternion q,
                         visualization_msgs::Marker &marker_, std_msgs::ColorRGBA color, ros::Duration lifetime = ros::Duration(0));
    void addMarker(int id, double length, double x, double y, double z,
                   double yaw, double pitch, double roll);
    void calculate_covariance(const cv::Vec3d &tvec, const cv::Vec3d &rvec, const std::vector<cv::Point2f> &marker_corners, Eigen::Matrix2d &covariance);
    /* 系统状态 */
    bool is_init_;

    /* 系统配置参数 */
    cv::Mat K_, dist_;   //　相机内参数
    double kl_, kr_, b_; // 里程计参数
    // Eigen::Matrix4d T_r_c_;                              // 机器人外参数
    geometry_msgs::TransformStamped transformStamped_r2c_;
    double k_;     // 里程计协方差参数
    double k_r_;   // 观测协方差参数
    double k_phi_; // 观测协方差参数

    int markers_dictionary_;
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Mat marker_img_;

    /* 上一帧的编码器读数 */
    double last_enl_, last_enr_;
    ros::Time last_time_;

    /* 求解的扩展状态 均值 和 协方差 */
    Eigen::MatrixXd mu_;         //均值
    Eigen::MatrixXd sigma_;      //方差
    std::vector<int> aruco_ids_; //对应于每个路标的aruco码id

    // cv::Ptr<cv::aruco::Board> board_;
    std::map<int, tf2::Vector3> myMap_;
    visualization_msgs::MarkerArray mapmarkerarray_;
    visualization_msgs::MarkerArray detectedmarkerarray_;

    cv::Ptr<cv::aruco::DetectorParameters> parameters_;
    cv::Mat camera_matrix_, dist_coeffs_;
    int N_;

    std::vector<cv::Point3f> objectPoints_ = {cv::Vec3f(-marker_length_ / 2.f, marker_length_ / 2.f, 0), cv::Vec3f(marker_length_ / 2.f, marker_length_ / 2.f, 0), cv::Vec3f(marker_length_ / 2.f, -marker_length_ / 2.f, 0), cv::Vec3f(-marker_length_ / 2.f, -marker_length_ / 2.f, 0)};

}; // class ArucoSlam

#endif
