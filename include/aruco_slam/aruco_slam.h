/** \file     	aruco_slam.h
 *  \author   	Yichen Liang (liangyichen666@gmail.com)
 *  \copyright  GNU General Public License (GPL)
 *  \brief   	Universal Synchronous/Asynchronous Receiver/Transmitter
 *  \version	V0.01
 *  \date    	07-OCT-2022
 *  \note
 *  This file is part of Arcuo_Slam.                                            \n
 *  This program is free software; you can redistribute it and/or modify 		\n
 *  it under the terms of the GNU General Public License version 3 as 		    \n
 *  published by the Free Software Foundation.                               	\n
 *  You should have received a copy of the GNU General Public License   		\n
 *  along with OST. If not, see <http://www.gnu.org/licenses/>.       			\n
 *  Unless required by applicable law or agreed to in writing, software       	\n
 *  distributed under the License is distributed on an "AS IS" BASIS,         	\n
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  	\n
 *  See the License for the specific language governing permissions and     	\n
 *  limitations under the License.   											\n
 */
#ifndef ARUCO_SLAM_H
#define ARUCO_SLAM_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <tf2_eigen/tf2_eigen.h>

/** \struct
 *  \brief Date loaded form parameter.yaml file for ArucoSlam class
 */
struct ArucoSlamIniteData
{
    double Q_k,                                           /**< Error coefficient of encoder */
        R_x,                                              /**< Error coefficient of observation x */
        R_y,                                              /**< Error coefficient of observation y */
        R_theta;                                          /**< Error coefficient of observation y */
    double kl,                                            /**< Left wheel radius */
        kr,                                               /**< Right wheel radius */
        b;                                                /**< Half of robot wheelbase */
    int markers_dictionary;                               /**< \enum cv::aruco::PREDEFINED_DICTIONARY_NAME */
    double marker_length;                                 /**< Length of the aruco markers */
    std::string world_frame,                              /**< Galobal frame name */
        camera_frame_optical,                             /**< Image frame name */
        robot_frame_base;                                 /**< Robot base link frame name */
    std::string image_topic_name,                         /**< Subscribed topic for image msg */
        encoder_topic_name;                               /**< Subscribed topic for encoder msg */
    std::string map_f;                                    /**< Direction of map file */
    geometry_msgs::TransformStamped transformStamped_r2c; /**< Calculated transfor matrix for robot base to camera optical */
    float USEFUL_DISTANCE_THRESHOLD=3;

};

/** \class ArucoMarker aruco_slam.h "include/aruco_slam.h"
 *  \brief This is a class for markers.
 *
 *  Every real object has a uniqu aruco_id and aruco_index.
 */
class ArucoMarker
{
public:
    ArucoMarker() {}
    ArucoMarker(const int &aruco_id, const double &x, const double &y, const double &theta, const Eigen::Matrix3d &observe_covariance)
        : aruco_id_(aruco_id), x_(x), y_(y), theta_(theta), observe_covariance_(observe_covariance), aruco_index_(-1) {}

    int aruco_id_;
    int aruco_index_;
    double x_;
    double y_;
    double theta_;
    Eigen::Matrix3d observe_covariance_;
    Eigen::Vector3d last_observation_; /**< Value of last observation contain delta_x, delta_y, delta_theta */
    /**
     * @brief used for the sort of the detected markers, the markers will be sorted according to
     * the sequence of been added to the map
     */
    friend bool operator<(const ArucoMarker &a, const ArucoMarker &b)
    {
        return a.aruco_index_ > b.aruco_index_;
    }

    friend bool operator==(const ArucoMarker &a, const ArucoMarker &b)
    {
        return a.aruco_id_ == b.aruco_id_;
    }
};

/** \class ArucoSlam aruco_slam.h "include/aruco_slam.h"
 *  \brief This is main class for aruco slam.
 *
 *  detailed description
 */
class ArucoSlam
{
public:
    /**
     * @brief Construct a new Aruco Slam object
     *
     * @param inite_data
     */
    ArucoSlam(const struct ArucoSlamIniteData &inite_data);
    /**
     * @brief add encoder data and update
     *
     * @param el
     * @param er
     */
    void addEncoder(const double &el, const double &er);
    /**
     * @brief add image data and update
     *
     * @param img
     */
    void addImage(const cv::Mat &img);
    /**
     * @brief Set the Camera Parameters object
     *
     * set camera projection matrix and distortion coefficients of the lens.
     * @param cameraparameters std::pair<cv::Mat, cv::Mat>
     */
    void setCameraParameters(const std::pair<cv::Mat, cv::Mat> &cameraparameters)
    {
        camera_matrix_ = cameraparameters.first;
        dist_coeffs_ = cameraparameters.second;
    }
    /**
     * @brief get the markers that have been added to the map, for visualization.
     *
     * @return visualization_msgs::MarkerArray
     */
    visualization_msgs::MarkerArray toRosMappedMarkers() { return detected_map_; };
    /**
     * @brief get the markers that been used for the EKF correction, for visualization in rviz.
     *
     * @return visualization_msgs::MarkerArray
     */
    visualization_msgs::MarkerArray toRosDetectedMarkers() { return detected_markers_; };
    /**
     * @brief get current pose of the robot with pose covariance, for visulization.
     *
     * @return geometry_msgs::PoseWithCovarianceStamped
     */
    geometry_msgs::PoseWithCovarianceStamped toRosPose();
    cv::Mat getMarkedImg() { return markered_img_; }

private:
    void fillTransform(tf2::Transform &transform_, const cv::Vec3d &rvec, const cv::Vec3d &tvec);

    Eigen::VectorXd mu() { return mu_; }
    Eigen::MatrixXd sigma() { return sigma_; }

    int getObservations(const cv::Mat &img);
    void normAngle(double &angle);
    bool checkLandmark(const int &aruco_id, int &landmark_idx);
    void clearMarkers();
    std::map<int, int> aruco_id_map; // pair<int, int>{aruco_id, position_i}
    bool GenerateMarker(int id, double length, double x, double y, double z, tf2::Quaternion q,
                        visualization_msgs::Marker &marker_, std_msgs::ColorRGBA color, ros::Duration lifetime = ros::Duration(0));
    void CalculateCovariance(const cv::Vec3d &tvec, const cv::Vec3d &rvec, const std::vector<cv::Point2f> &marker_corners, Eigen::Matrix3d &covariance);
    bool is_init_;
    /** Parameters */
    cv::Mat camera_matrix_, dist_coeffs_; /**< Camera parameters*/
    cv::Mat K_, dist_;
    double kl_, kr_, b_; /**< Robot parameters */
    geometry_msgs::TransformStamped transformStamped_r2c_;
    int markers_dictionary_; /**< Markers parameters*/
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    double Q_k_;                 /**< Error coefficient of encoder */
    double R_x_, R_y_, R_theta_; /**< Error coefficient of observation */
    double last_enl_, last_enr_; /**< Encoder data recoder */
    ros::Time last_time_;
    int N_;                      /**< Number of markers in the map */
    Eigen::VectorXd mu_;         /**< Mean of state 3+n*3 */
    Eigen::MatrixXd sigma_;      /**< Covariance of state */
    std::vector<int> aruco_ids_; /**< Ids of markers in the map */
    cv::Mat markered_img_;       /**< Image with markered detected markers */
    visualization_msgs::MarkerArray detected_map_;
    visualization_msgs::MarkerArray detected_markers_;
    std::vector<ArucoMarker> last_observed_marker_;
    std::vector<cv::Point3f> objectPoints_ = {cv::Vec3f(-marker_length_ / 2.f, marker_length_ / 2.f, 0), cv::Vec3f(marker_length_ / 2.f, marker_length_ / 2.f, 0), cv::Vec3f(marker_length_ / 2.f, -marker_length_ / 2.f, 0), cv::Vec3f(-marker_length_ / 2.f, -marker_length_ / 2.f, 0)};
    std::priority_queue<ArucoMarker> obs_; /**< record detected markers in order of map */
    float USEFUL_DISTANCE_THRESHOLD_;

};

#endif
