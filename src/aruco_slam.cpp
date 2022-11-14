#include "aruco_slam/aruco_slam.h"

ArucoSlam::ArucoSlam(const struct ArucoSlamIniteData &inite_data)
    : kl_(inite_data.kl), kr_(inite_data.kr), b_(inite_data.b), transformStamped_r2c_(inite_data.transformStamped_r2c), Q_k_(inite_data.Q_k),
      R_x_(inite_data.R_x), R_y_(inite_data.R_y), R_theta_(inite_data.R_theta), markers_dictionary_(inite_data.markers_dictionary),
      marker_length_(inite_data.marker_length), USEFUL_DISTANCE_THRESHOLD_(inite_data.USEFUL_DISTANCE_THRESHOLD)
{
    is_init_ = false;
    K_ = cv::Mat(3, 3, 6);
    dist_ = cv::Mat(5, 1, 6);
    dictionary_ = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(markers_dictionary_));
    mu_.resize(3);
    mu_.setZero();
    sigma_.resize(3, 3);
    sigma_.setZero();
    detected_map_.markers.clear();
    last_observed_marker_.clear();
}

void ArucoSlam::addEncoder(const double &wl, const double &wr)
{
    // ROS_INFO_STREAM("Encoder data reveived");
    if (is_init_ == false)
    {
        last_time_ = ros::Time::now();
        is_init_ = true;
        return;
    }

    double dt = (ros::Time::now() - last_time_).toSec();
    last_time_ = ros::Time::now();

    /* calculate change of distance */
    double delta_enl = dt * wl; // change of angle
    double delta_enr = dt * wr;
    double delta_sl = kl_ * delta_enl; // change of distance
    double delta_sr = kr_ * delta_enr;

    double l_ = 2 * b_;
    double delta_theta = (delta_sr - delta_sl) / l_;
    double delta_s = 0.5 * (delta_sr + delta_sl);

    /***** update mean value *****/
    double tmp_th = mu_(2) + 0.5 * delta_theta;
    double cos_tmp_th = cos(tmp_th);
    double sin_tmp_th = sin(tmp_th);

    mu_(0) += delta_s * cos_tmp_th;
    mu_(1) += delta_s * sin_tmp_th;
    mu_(2) += delta_theta;
    normAngle(mu_(2)); // norm

    /***** update covariance *****/
    Eigen::Matrix3d H_xi;
    H_xi << 1.0, 0.0, -delta_s * sin_tmp_th,
        0.0, 1.0, delta_s * cos_tmp_th,
        0.0, 0.0, 1.0;

    Eigen::Matrix<double, 3, 2> wkh;
    wkh << cos_tmp_th, cos_tmp_th, sin_tmp_th, sin_tmp_th, 1 / b_, -1 / b_;
    wkh = (0.5 * kl_ * dt) * wkh;

    int N = mu_.rows();
    Eigen::MatrixXd F(N, 3);
    F.setZero();
    F.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Identity(N, N);
    Hx.block(0, 0, 3, 3) = H_xi;
    Eigen::Matrix2d sigma_u;
    sigma_u << Q_k_ * fabs(wl), 0.0, 0.0, Q_k_ * fabs(wr);
    Eigen::MatrixXd Qk = wkh * sigma_u * wkh.transpose();
    sigma_ = Hx * sigma_ * Hx.transpose() + F * Qk * F.transpose();
}

void ArucoSlam::addImage(const cv::Mat &img)
{
    // ROS_INFO_STREAM("Image data reveived");
    ROS_INFO_STREAM(" \n\n\n\n\n\n\n  Image data reveived " << std::endl
                    // "z_hat:" << z_hat << std::endl
                    // << "mu:" << mu_ << std::endl
                    // << "sigma:" << sigma_ << std::endl
    );
    if (is_init_ == false)
        return;

    getObservations(img);
    Eigen::MatrixXd mu = mu_;
    ROS_INFO_STREAM("obs_.size:" << obs_.size() << std::endl);
    std::vector<ArucoMarker> observed_marker;
    int index = 0;
    while (!obs_.empty())
    {
        ArucoMarker ob = obs_.top();
        obs_.pop();
        ROS_INFO_STREAM("\n\n Image data reveived 1 "
                        << "ob.aruco_index_: " << ob.aruco_index_ << "\n\n");

        /* 计算观测方差 */
        Eigen::Matrix3d Rk = ob.observe_covariance_;
        // double &x = mu_(0);
        // double &y = mu_(1);
        // double &theta = mu_(2);
        // double sintheta = sin(theta);
        // double costheta = cos(theta);
        // Q << k_r_ * k_r_ * fabs(ob.x_ * ob.x_) + 0.1, 0.0, 0.0, k_phi_ * k_phi_ * fabs(ob.y_ * ob.y_) + 0.1;
        // ROS_INFO_STREAM_ONCE("addImage");
        if (ob.aruco_index_ >= 0) // 如果路标已经存在了
        {
            // ROS_INFO_STREAM("\n\n\n\n\n\\n\n  Image data reveived 1 \n\n\n\n\n");

            int N = mu_.rows();
            Eigen::MatrixXd F(6, N);
            F.setZero();
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(3, 3 + 3 * ob.aruco_index_) = Eigen::Matrix3d::Identity();

            /* calculate estimation based on estimated state */
            double &mx = mu(3 + 3 * ob.aruco_index_);
            double &my = mu(4 + 3 * ob.aruco_index_);
            double &mtheta = mu(5 + 3 * ob.aruco_index_);
            double &x = mu(0);
            double &y = mu(1);
            double &theta = mu(2);
            double sintheta = sin(theta);
            double costheta = cos(theta);
            double global_delta_x = mx - x;
            double global_delta_y = my - y;
            double global_delta_theta = mtheta - theta;
            normAngle(global_delta_theta);

            Eigen::Vector3d z_hat(global_delta_x * costheta + global_delta_y * sintheta,
                                  -global_delta_x * sintheta + global_delta_y * costheta,
                                  global_delta_theta);

            Eigen::Vector3d z(ob.x_, ob.y_, ob.theta_);
            Eigen::Vector3d ze = z - z_hat;
            normAngle(ze[2]);

            Eigen::MatrixXd Gxm(3, 6);
            Gxm << -costheta, -sintheta, -global_delta_x * sintheta + global_delta_y * costheta, costheta, sintheta, 0,
                sintheta, -costheta, -global_delta_x * costheta - global_delta_y * sintheta, -sintheta, costheta, 0,
                0, 0, -1, 0, 0, 1;
            Eigen::MatrixXd Gx = Gxm * F;

            Eigen::MatrixXd K = sigma_ * Gx.transpose() * (Gx * sigma_ * Gx.transpose() + Rk).inverse();

            // if(K.norm()>1){
            // ROS_INFO_STREAM("sigma_:" << sigma_ << std::endl
            //                           << "F:" << F << std::endl
            //                           << "Gxm:" << Gxm << std::endl
            //                           << "K:" << K << std::endl);
            // }
            // double phi_hat = atan2(delta_y, delta_x) - theta;
            // normAngle(phi_hat);
            if (ze.norm() >= 1 || K.norm() >= 10)
            {
                // ROS_INFO_ONCE("error of z: %lf", ze.norm());
                // ROS_ERROR_STREAM_ONCE("error of z:" << ze.norm() << std::endl);
                // ROS_INFO_STREAM_ONCE("state:" << mu_ << std::endl);
                ROS_INFO_STREAM("\n\n\n error \n\n\n"
                                //  << "ob.aruco_index_:"<<ob.aruco_index_ << std::endl
                                << "z_hat:" << z_hat << std::endl
                                << "z:" << z << std::endl
                                << "ze:" << ze.norm() << std::endl
                                << "K:" << K.norm() << std::endl
                                << "Rk:" << Rk << std::endl
                                << "Gxm:" << Gxm << std::endl
                                //  << "i:" << ob.aruco_index_ << std::endl
                                << "mu:" << mu_ << std::endl
                                << "sigma:" << sigma_ << std::endl);
                // TODO: Remove map point?
                // mu_.topLeftCorner(3, 0) += (K * ze).topLeftCorner(3, 0);
                // continue;
            }
            // ROS_INFO_STREAM(" "
            //                 // "z_hat:" << z_hat << std::endl
            //                 << "ze:" << ze << std::endl
            //                 << "i:" << ob.aruco_index_ << std::endl
            //                 << "K:" << K << std::endl
            //                 << "Rk:" << Rk << std::endl
            //                 << "Gxm:" << Gxm << std::endl
            //                 << "mu:" << mu_ << std::endl
            //                 << "sigma:" << sigma_ << std::endl);

            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);

            bool up_date_map = false;
            double robot_pose_convariance = sigma_.topLeftCorner(3, 3).norm();
            double map_pose_convariance = sigma_.block(3 + 3 * ob.aruco_index_, 3 + 3 * ob.aruco_index_, 3, 3).norm();

            std::vector<ArucoMarker>::iterator last_observe_ptr = std::find(last_observed_marker_.begin(), last_observed_marker_.end(), ob);
            if (last_observe_ptr != last_observed_marker_.end() && (last_observe_ptr->last_observation_ - z).norm() < 0.01)
            {
                ROS_INFO_STREAM("\n lastobservation_delt:" << (last_observe_ptr->last_observation_ - z).norm() << std::endl);
                mu_.topLeftCorner(3, 0) += (K * ze).topLeftCorner(3, 0);
                // sigma_.topLeftCorner(3, 3) = ((I - K * Gx) * sigma_).topLeftCorner(3, 3);
            }
            else
            {
                // only do correction after the ArucoMarker changes a distance
                ob.last_observation_ = z;
                mu_ += (K * ze);
                sigma_ = ((I - K * Gx) * sigma_);
            }
            // ROS_INFO_STREAM("mu_corrected:" << mu_ << std::endl);
        }
        else // new markers are added to the map
        {
            float sinth = sin(mu(2));
            float costh = cos(mu(2));

            /**** add to the map ****/
            /* updata the mean value */
            int N = mu_.rows();
            Eigen::VectorXd tmp_mu(N + 3);
            tmp_mu.setZero();

            double map_x = mu(0) + costh * ob.x_ - sinth * ob.y_;
            double map_y = mu(1) + sinth * ob.x_ + costh * ob.y_;
            double map_theta = mu(2) + ob.theta_;
            normAngle(map_theta);
            tmp_mu << mu_, map_x, map_y, map_theta;
            mu_.resize(N + 3);
            mu_ = tmp_mu;

            /* update the covariance of the map */
            double deltax = map_x - mu(0);
            double deltay = map_y - mu(1);
            Eigen::Matrix3d sigma_s = sigma_.block(0, 0, 3, 3);
            Eigen::Matrix<double, 3, 3> Gsk;
            Gsk << -costh, -sinth, -sinth * deltax + costh * deltay,
                sinth, -costh, -deltax * costh - deltay * sinth,
                0, 0, -1;
            Eigen::Matrix<double, 3, 3> Gmi;
            Gmi << costh, sinth, 0,
                -sinth, costh, 0,
                0, 0, 1;

            /* calculate variance for new marker*/
            Eigen::Matrix3d sigma_mm = Gmi * (Gsk * sigma_s * Gsk.transpose() + Rk).transpose() * Gmi.transpose();

            /* calculate covariance for new marker and exited markers*/
            Eigen::MatrixXd Gfx = sigma_.topRows(3);
            Eigen::MatrixXd sigma_mx = -Gmi * Gsk * Gfx;
            Eigen::MatrixXd tmp_sigma(N + 3, N + 3);
            tmp_sigma.setZero();
            tmp_sigma.topLeftCorner(N, N) = sigma_;
            tmp_sigma.topRightCorner(N, 3) = sigma_mx.transpose();
            tmp_sigma.bottomLeftCorner(3, N) = sigma_mx;
            tmp_sigma.bottomRightCorner(3, 3) = sigma_mm;
            sigma_.resize(N + 3, N + 3);
            sigma_ = tmp_sigma;

            /***** add new marker's id to the dictionary *****/
            aruco_id_map.insert(std::pair<int, int>{ob.aruco_id_, (mu_.rows() - 3) / 3 - 1});
            // ROS_INFO_STREAM("state:" << mu_ << std::endl
            //                          << "obs:id" << ob.aruco_id_ << "obs:x:" << ob.x_ << "obs:y: " << ob.y_ << "obs:theta:" << ob.theta_ << std::endl
            //                          << "obs.covariance:" << ob.covariance_ << std::endl);
        } // add new landmark
        observed_marker.push_back(ob);
    } // for all observation
    last_observed_marker_ = observed_marker;
    /* visualise the new map marker */
    detected_map_.markers.clear();
    for (int i = 0; i < (mu_.rows() - 3) / 3; i++)
    {
        double map_x = mu_(i * 3 + 3, 0);
        double map_y = mu_(i * 3 + 4, 0);
        double map_theta = mu_(i * 3 + 5, 0);
        tf2::Quaternion q;
        q.setRPY(0, 1.5708, map_theta);
        visualization_msgs::Marker marker;
        std_msgs::ColorRGBA color;
        color.a = 0.5;
        color.b = 1;
        color.g = 0.5;
        color.r = 1;
        GenerateMarker(i, marker_length_, map_x, map_y, 0.3, q, marker, color);
        detected_map_.markers.push_back(marker);
    }
    // ROS_INFO_STREAM("Image data precess finished");
    ROS_INFO_STREAM("Image data precess finished "
                    // "z_hat:" << z_hat << std::endl
                    << "mu:" << mu_ << std::endl
                    << "sigma:" << sigma_ << std::endl);
}

bool ArucoSlam::GenerateMarker(int id, double length, double x, double y, double z, tf2::Quaternion q, visualization_msgs::Marker &marker_, std_msgs::ColorRGBA color, ros::Duration lifetime)
{
    marker_.id = id;
    marker_.header.frame_id = "world";
    marker_.type = visualization_msgs::Marker::CUBE;
    marker_.scale.x = length;
    marker_.scale.y = length;
    marker_.scale.z = 0.01;

    marker_.color = color;
    marker_.pose.position.x = x;
    marker_.pose.position.y = y;
    marker_.pose.position.z = z;
    marker_.pose.orientation = tf2::toMsg(q);
    marker_.lifetime = lifetime;
    return true;
}

int ArucoSlam::getObservations(const cv::Mat &img)
{
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> IDs;
    std::vector<cv::Vec3d> rvs, tvs;
    detected_markers_.markers.clear();
    cv::aruco::detectMarkers(img, dictionary_, marker_corners, IDs);
    cv::aruco::estimatePoseSingleMarkers(marker_corners, marker_length_, camera_matrix_, dist_coeffs_, rvs, tvs);
    // The returned transformation is the one that transforms points from the board coordinate system to the camera coordinate system.

    /* draw all marks */
    markered_img_ = img.clone();
    cv::aruco::drawDetectedMarkers(markered_img_, marker_corners, IDs);
    // for (size_t i = 0; i < IDs.size(); i++)
    //     cv::aruco::drawAxis(marker_img_, K_, dist_, rvs[i], tvs[i], 0.07);

    // const float USEFUL_DISTANCE_THRESHOLD = 2.5; // 3 m

    for (size_t i = 0; i < IDs.size(); i++)
    {
        float dist = cv::norm<double>(tvs[i]);
        // float dist = tvs[i][2];
        if (dist > USEFUL_DISTANCE_THRESHOLD_)
        {
            continue;
            ROS_ERROR_STREAM("USEFUL_DISTANCE_THRESHOLD_:"<<USEFUL_DISTANCE_THRESHOLD_);
        }

        /*visualise used marks */
        visualization_msgs::Marker _marker;
        tf2::Transform _transform;
        fillTransform(_transform, rvs[i], tvs[i]);
        std_msgs::ColorRGBA color;
        color.a = 1;
        color.b = 0;
        color.g = 0;
        color.r = 1;
        GenerateMarker(IDs[i], marker_length_, tvs[i][0], tvs[i][1], tvs[i][2], _transform.getRotation(), _marker, color, ros::Duration(0.1));
        tf2::doTransform(_marker.pose, _marker.pose, transformStamped_r2c_);
        _marker.header.frame_id = "base_link";
        detected_markers_.markers.push_back(_marker);
        // cv::aruco::drawAxis(marker_img_, camera_matrix_, dist_coeffs_, rvs[i], tvs[i], 0.07);

        /* calculate observation mx, my, mtheta */
        cv::Vec3d tvec = tvs[i];
        cv::Vec3d rvec = rvs[i];
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Eigen::Affine3d transform_matrix = tf2::transformToEigen(transformStamped_r2c_);
        // ROS_INFO_STREAM_ONCE("transformStamped_r2c_"<<transformStamped_r2c_<<std::endl<<"tvsc:"<<tvec<<std::endl<<"rvec:"<<rvec<<std::endl);

        double x = tvec[2] + transformStamped_r2c_.transform.translation.x;
        double y = -tvec[0] + transformStamped_r2c_.transform.translation.y;
        double theta = atan2(-R.at<double>(0, 2), R.at<double>(2, 2));
        normAngle(theta);
        int aruco_id = IDs[i];
        Eigen::Matrix3d covariance;
        CalculateCovariance(tvec, rvec, marker_corners[i], covariance);
        /* add to observation vector */
        if (covariance.norm() > 1)
            continue;
        ArucoMarker ob(aruco_id, x, y, theta, covariance);
        int aruco_index;
        checkLandmark(aruco_id, aruco_index);
        ob.aruco_index_ = aruco_index;
        obs_.push(ob);
    } // for all detected markers
    return obs_.size();
}

geometry_msgs::PoseWithCovarianceStamped ArucoSlam::toRosPose()
{
    /* 转换带协方差的机器人位姿 */
    geometry_msgs::PoseWithCovarianceStamped rpose;
    rpose.header.frame_id = "world";
    rpose.pose.pose.position.x = mu_(0);
    rpose.pose.pose.position.y = mu_(1);
    rpose.pose.pose.position.z = 0.1;

    // rpose.pose.pose.orientation = tf2::createQuaternionMsgFromYaw(mu_(2));
    tf2::Quaternion QuaternionMsgFromYaw;
    QuaternionMsgFromYaw.setRPY(0, 0, mu_(2));
    rpose.pose.pose.orientation = tf2::toMsg(QuaternionMsgFromYaw);
    /*rpose.pose.covariance
    xx  xy  0   0   0   xy  0-5
    yx  yy  0   0   0   yy  6-11
    0   0   0   0   0   0   12-17
    0   0   0   0   0   0   18-23
    0   0   0   0   0   0   24-29
    yy  yy  0   0   0   yy  30-35
    */
    rpose.pose.covariance.at(0) = sigma_(0, 0);
    rpose.pose.covariance.at(1) = sigma_(0, 1);
    rpose.pose.covariance.at(5) = sigma_(0, 2);
    rpose.pose.covariance.at(6) = sigma_(1, 0);
    rpose.pose.covariance.at(7) = sigma_(1, 1);
    rpose.pose.covariance.at(11) = sigma_(1, 2);
    rpose.pose.covariance.at(30) = sigma_(2, 0);
    rpose.pose.covariance.at(31) = sigma_(2, 1);
    rpose.pose.covariance.at(35) = sigma_(2, 2);

    return rpose;
}

void ArucoSlam::normAngle(double &angle)
{
    /* swap angle to (-pi-pi)*/
    const static double PI = 3.14159265358979323846;
    const static double Two_PI = 2.0 * PI;
    if (angle >= PI)
        angle -= Two_PI;
    if (angle < -PI)
        angle += Two_PI;
}

bool ArucoSlam::checkLandmark(const int &aruco_id, int &landmark_idx)
{

    if (!aruco_id_map.empty() && aruco_id_map.end() != aruco_id_map.find(aruco_id))
    {
        landmark_idx = aruco_id_map.at(aruco_id);
        // ROS_INFO_STREAM("aruco_id:" << aruco_id << "index:" << landmark_idx);
        return true;
    }
    else
        landmark_idx = -1;
    return false;
}

void ArucoSlam::CalculateCovariance(const cv::Vec3d &tvec, const cv::Vec3d &rvec, const std::vector<cv::Point2f> &marker_corners, Eigen::Matrix3d &covariance)
{

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints_, rvec, tvec, camera_matrix_,
                      dist_coeffs_, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;

    auto dist = [](const cv::Point2f &p1, const cv::Point2f &p2)
    {
        double x1 = p1.x;
        double y1 = p1.y;
        double x2 = p2.x;
        double y2 = p2.y;

        double dx = x1 - x2;
        double dy = y1 - y2;

        return sqrt(dx * dx + dy * dy);
    };

    for (unsigned int i = 0; i < objectPoints_.size(); i++)
    {
        double error = dist(marker_corners[i], projectedPoints[i]);
        totalError += error * error;
    }
    double rmserror = totalError / (double)objectPoints_.size();
    double object_error = (rmserror / dist(marker_corners[0], marker_corners[2])) *
                          (norm(tvec) / marker_length_);
    // covariance << rerror / 100, 0, 0, 0, rerror / 300, 0, 0, 0, rerror / 100;
    // ROS_INFO_STREAM("object_error:"<<object_error);
    covariance << object_error * R_x_ + 1e-2, 0, 0, 0, object_error * R_y_ + 1e-2, 0, 0, 0, object_error * R_theta_ + 1e-3;
}

void ArucoSlam::fillTransform(tf2::Transform &transform_, const cv::Vec3d &rvec, const cv::Vec3d &tvec)
{
    cv::Mat rot(3, 3, CV_64FC1);
    // cv::Mat Rvec64;
    // rvec.convertTo(Rvec64, CV_64FC1);
    cv::Rodrigues(rvec, rot);
    // cv::Mat tran64;
    // tvec.convertTo(tran64, CV_64FC1);

    tf2::Matrix3x3 tf_rot(rot.at<double>(0, 0), rot.at<double>(0, 1), rot.at<double>(0, 2),
                          rot.at<double>(1, 0), rot.at<double>(1, 1), rot.at<double>(1, 2),
                          rot.at<double>(2, 0), rot.at<double>(2, 1), rot.at<double>(2, 2));

    tf2::Vector3 tf_orig(tvec[0], tvec[1], tvec[2]);

    tf2::Transform transform(tf_rot, tf_orig);
    transform_ = transform;
}