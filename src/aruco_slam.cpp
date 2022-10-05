#include "aruco_slam/aruco_slam.h"

ArucoSlam::ArucoSlam(const double &kl, const double kr, const double &b,
                     const geometry_msgs::TransformStamped &transformStamped_r2c,
                     const double &k, const double &k_r, const double k_phi,
                     const int &markers_dictionary, const double &marker_length)
    : kl_(kl), kr_(kr), b_(b), transformStamped_r2c_(transformStamped_r2c), k_(k), k_r_(k_r), k_phi_(k_phi), markers_dictionary_(markers_dictionary), marker_length_(marker_length)
{
    is_init_ = false;
    K_ = cv::Mat(3, 3, 6);
    dist_ = cv::Mat(5, 1, 6);
    dictionary_ = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(markers_dictionary_));
    /* 初始时刻机器人位姿为0，绝对准确, 协方差为0 */
    mu_.resize(3);
    mu_.setZero();
    sigma_.resize(3, 3);
    sigma_.setZero();
}

ArucoSlam::ArucoSlam(const struct ArucoSlamIniteData &inite_data)
    : kl_(inite_data.kl), kr_(inite_data.kr), b_(inite_data.b), transformStamped_r2c_(inite_data.transformStamped_r2c), k_(inite_data.k),
      k_r_(inite_data.k_r), k_phi_(inite_data.k_phi), markers_dictionary_(inite_data.markers_dictionary), marker_length_(inite_data.marker_length)
{
    is_init_ = false;
    K_ = cv::Mat(3, 3, 6);
    dist_ = cv::Mat(5, 1, 6);
    dictionary_ = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(markers_dictionary_));
    /* State and Covariance */
    mu_.resize(3);
    mu_.setZero();
    sigma_.resize(3, 3);
    sigma_.setZero();
}

void ArucoSlam::addEncoder(const double &wl, const double &wr)
{   
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
    Eigen::Matrix3d G_xi;
    G_xi << 1.0, 0.0, -delta_s * sin_tmp_th,
        0.0, 1.0, delta_s * cos_tmp_th,
        0.0, 0.0, 1.0;

    Eigen::Matrix<double, 3, 2> Gup;
    Gup << cos_tmp_th, cos_tmp_th, sin_tmp_th, sin_tmp_th, 1 / b_, -1 / b_;
    Gup = (0.5 * kl_ * dt) * Gup;
    int N = mu_.rows();
    Eigen::MatrixXd F(N, 3);
    F.setZero();
    F.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd Gt = Eigen::MatrixXd::Identity(N, N);
    Gt.block(0, 0, 3, 3) = G_xi;
    Eigen::Matrix2d sigma_u;
    sigma_u << k_ * fabs(wl), 0.0, 0.0, k_ * fabs(wr);

    sigma_ = Gt * sigma_ * Gt.transpose() + F * Gup * sigma_u * Gup.transpose() * F.transpose();
}

void ArucoSlam::addImage(const cv::Mat &img)
{   
    if (is_init_ == false)
        return;
    std::vector<Observation> obs;
    getObservations(img, obs);
    for (Observation ob : obs)
    {
        /* 计算观测方差 */
        Eigen::Matrix3d Q = ob.covariance_;
        // Q << k_r_ * k_r_ * fabs(ob.x_ * ob.x_) + 0.1, 0.0, 0.0, k_phi_ * k_phi_ * fabs(ob.y_ * ob.y_) + 0.1;
        ROS_INFO_STREAM_ONCE("addImage");
        int i;                              // 第 i 个路标
        if (checkLandmark(ob.aruco_id_, i)) // 如果路标已经存在了
        {
            // int N = mu_.rows();
            // Eigen::MatrixXd F(5, N);
            // F.setZero();
            // F.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
            // F(3, 3 + 2 * i) = 1;
            // F(4, 4 + 2 * i) = 1;

            // double &mx = mu_(3 + 2 * i);
            // double &my = mu_(4 + 2 * i);
            // double &x = mu_(0);
            // double &y = mu_(1);
            // double &theta = mu_(2);
            // double delta_x = mx - x;
            // double delta_y = my - y;

            // double sintheta = sin(theta);
            // double costheta = cos(theta);
            // Eigen::MatrixXd Hv(2, 5);
            // Eigen::Vector2d z_hat(delta_x, delta_y);

            // Eigen::Vector2d z(ob.x_, ob.y_);
            // Eigen::Vector2d ze = z - z_hat;
            // if (ze.norm() >= 1)
            // {
            //     ROS_INFO_ONCE("error of z: %lf", ze.norm());
            //     ROS_ERROR_STREAM_ONCE("error of z:" << ze.norm() << std::endl);
            //     ROS_INFO_STREAM_ONCE("state:" << mu_ << std::endl);
            //     continue;
            // }

            // Hv << 1, 0, -mx * sintheta - my * costheta, costheta, -sintheta,
            //     0, 1, mx * costheta - my * sintheta, sintheta, costheta;
            // Eigen::MatrixXd Ht = Hv * F;
            // Eigen::MatrixXd K = sigma_ * Ht.transpose() * (Ht * sigma_ * Ht.transpose() + Q).inverse();
            // // if(K.norm()>1){
            // // ROS_INFO_STREAM("sigma_:" << sigma_ << std::endl
            // //                           << "Ht:" << Ht << std::endl
            // //                           << "Q:" << Q << std::endl
            // //                           << "i:" << i << std::endl);
            // // }
            // // double phi_hat = atan2(delta_y, delta_x) - theta;
            // // normAngle(phi_hat);

            // // ROS_INFO_STREAM("z_hat:" << z_hat << std::endl
            // //                          << "z:" << z << std::endl
            // //                          << "k:" << K << std::endl
            // //                          << "mu:" << mu_ << std::endl);

            // mu_ = mu_ + K * (z - z_hat);
            // Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
            // sigma_ = (I - K * Ht) * sigma_;
            // ROS_INFO_STREAM("mu_corrected:" << mu_ << std::endl);
        }
        else // new markers are added to the map
        {
            float sinth = sin(mu_(2));
            float costh = cos(mu_(2));

            /**** add to the map ****/
            /* updata the mean value */
            int N = mu_.rows();
            Eigen::VectorXd tmp_mu(N + 3);
            tmp_mu.setZero();

            tmp_mu << mu_, mu_(0) + costh * ob.x_ - sinth * ob.y_, mu_(1) + sinth * ob.x_ + costh * ob.y_, mu_(2) + ob.theta_;
            mu_.resize(N + 3);
            mu_ = tmp_mu;

            /* update the covariance of the map */
            Eigen::Matrix3d sigma_s = sigma_.block(0, 0, 3, 3);
            Eigen::Matrix<double, 3, 3> Gsk;
            Gsk << -costh, 0, ob.x_ * sinth,
                0, -sinth, -ob.y_ * costh,
                0, 0, -1;
            Eigen::Matrix<double, 3, 3> Gmi;
            Gmi << 1, 0, -ob.x_ * sinth,
                0, 1, ob.y_ * costh,
                0, 0, 1;

            /* calculate variance for new marker*/
            Eigen::Matrix3d sigma_mm = Gmi * (Gsk * sigma_s * Gsk.transpose() + Q).transpose() * Gmi.transpose();

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
            aruco_id_map.insert(std::pair<int, int>{ob.aruco_id_, (mu_.rows() - 3) / 2 - 1});
            ROS_INFO_STREAM("state:" << mu_ << std::endl
                                     << "obs:id" << ob.aruco_id_ << "obs:x,y" << ob.x_ << " " << ob.y_ << std::endl
                                     << "obs.covariance:" << ob.covariance_ << std::endl);
        } // add new landmark
    }     // for all observation

    // get_detected_map().markers.clear();
    // for (int i = 0; i < (mu_.rows() - 3) / 2; i++)
    // {

    //     double x = mu_(i * 2 + 3, 0);
    //     double y = mu_(i * 2 + 4, 0);
    //     visualization_msgs::Marker marker;
    //     std_msgs::ColorRGBA color;
    //     color.a = 0.5;
    //     color.b = 1;
    //     color.g = 0.5;
    //     color.r = 1;
    //     ROS_INFO_STREAM("x:" << x << "y:" << y << std::endl);
    //     CylinderMarkerGenerate(i, x, y, 0.3, color, ros::Duration(1), marker);
    //     get_detected_map().markers.push_back(marker);
    // }
}

void ArucoSlam::loadMap(std::string filename)
{
    ROS_INFO("\n loading from: %s", filename.c_str());
    std::ifstream f(filename);
    std::string line;
    clearMarkers();
    if (!f.good())
    {
        ROS_ERROR("%s - %s", strerror(errno), filename.c_str());
        return;
    }
    N_ = 3;
    while (std::getline(f, line))
    {
        int id;
        double length, x, y, z, yaw, pitch, roll;

        std::istringstream s(line);
        // Read first character to see whether it's a comment
        char first = 0;
        if (!(s >> first))
        {
            // No non-whitespace characters, must be a blank line
            continue;
        }

        if (first == '#')
        {
            ROS_DEBUG("Skipping line as a comment: %s", line.c_str());
            continue;
        }
        else if (isdigit(first))
        {
            // Put the digit back into the stream
            // Note that this is a non-modifying putback, so this should work with istreams
            // (see https://en.cppreference.com/w/cpp/io/basic_istream/putback)
            s.putback(first);
        }
        else
        {
            // Probably garbage data; inform user and throw an exception, possibly killing nodelet
            ROS_ERROR("Malformed input: %s", line.c_str());
            clearMarkers();
            return;
        }

        if (!(s >> id >> length >> x >> y))
        {
            ROS_ERROR("Not enough data in line: %s; "
                      "Each marker must have at least id, length, x, y fields",
                      line.c_str());
            continue;
        }
        // Be less strict about z, yaw, pitch roll
        if (!(s >> z))
        {
            ROS_DEBUG("No z coordinate provided for marker %d, assuming 0", id);
            z = 0;
        }
        if (!(s >> roll))
        {
            ROS_DEBUG("No yaw provided for marker %d, assuming 0", id);
            yaw = 0;
        }
        if (!(s >> pitch))
        {
            ROS_DEBUG("No pitch provided for marker %d, assuming 0", id);
            pitch = 0;
        }
        if (!(s >> yaw))
        {
            ROS_DEBUG("No roll provided for marker %d, assuming 0", id);
            roll = 0;
        }
        // ROS_INFO("id, length, x, y, z, yaw, pitch, roll:%d, %lf, %lf %lf %lf %lf %lf %lf ",
        // id, length, x, y, z, yaw, pitch, roll);
        addMarker(id, length, x, y, z, yaw, pitch, roll);
        N_ += 1;
    }

    ROS_INFO("loading %s complete (%d markers)", filename.c_str(), static_cast<int>(myMap_.size()));
}

void ArucoSlam::clearMarkers()
{
    myMap_.clear();
    real_map_.markers.clear();
}

void ArucoSlam::addMarker(int id, double length, double x, double y, double z,
                          double yaw, double pitch, double roll)
{
    // Create transform
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    tf2::Vector3 tvs(x, y, z);
    tf2::Transform transform(q, tvs);
    myMap_.insert(std::pair<int, tf2::Vector3>(id, tvs));
    /* marker's corners:
        y
        ^
    0	|	1
    ----|------->x
    3	|	2
    */
    // double halflen = length / 2;
    // tf2::Vector3 p0(-halflen, halflen, 0);
    // tf2::Vector3 p1(halflen, halflen, 0);
    // tf2::Vector3 p2(halflen, -halflen, 0);
    // tf2::Vector3 p3(-halflen, -halflen, 0);
    // p0 = transform * p0;
    // p1 = transform * p1;
    // p2 = transform * p2;
    // p3 = transform * p3;

    // std::vector<cv::Point3f> obj_points = {
    //     cv::Point3f(p0.x(), p0.y(), p0.z()),
    //     cv::Point3f(p1.x(), p1.y(), p1.z()),
    //     cv::Point3f(p2.x(), p2.y(), p2.z()),
    //     cv::Point3f(p3.x(), p3.y(), p3.z())};

    // board_->ids.push_back(id);
    // board_->objPoints.push_back(obj_points);

    // Add marker to array
    visualization_msgs::Marker marker;
    std_msgs::ColorRGBA color;
    color.a = 1;
    color.b = 1;
    color.g = 1;
    color.r = 1;
    marker_generate(id, length, x, y, z, q, marker, color);
    real_map_.markers.push_back(marker);
}

bool ArucoSlam::marker_generate(int id, double length, double x, double y, double z, tf2::Quaternion q, visualization_msgs::Marker &marker_, std_msgs::ColorRGBA color, ros::Duration lifetime)
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

bool ArucoSlam::CylinderMarkerGenerate(const int &id, const double &x, const double &y, const double &z, const std_msgs::ColorRGBA &color, const ros::Duration &lifetime, visualization_msgs::Marker &marker_)
{
    marker_.id = id;
    marker_.header.frame_id = "world";
    marker_.type = visualization_msgs::Marker::CYLINDER;
    marker_.scale.x = 0.1;
    marker_.scale.y = 0.1;
    marker_.scale.z = 0.2;

    marker_.color = color;
    marker_.pose.position.x = x;
    marker_.pose.position.y = y;
    marker_.pose.position.z = z;

    marker_.pose.orientation.w = 1;
    marker_.pose.orientation.x = 0;
    marker_.pose.orientation.y = 0;
    marker_.pose.orientation.z = 0;

    marker_.lifetime = lifetime;
    return true;
}

int ArucoSlam::getObservations(const cv::Mat &img, std::vector<Observation> &obs)
{
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected;
    std::vector<int> IDs;
    std::vector<cv::Vec3d> rvs, tvs;
    detected_markers_.markers.clear();
    cv::aruco::detectMarkers(img, dictionary_, marker_corners, IDs);
    cv::aruco::estimatePoseSingleMarkers(marker_corners, marker_length_, camera_matrix_, dist_coeffs_, rvs, tvs);
    // The returned transformation is the one that transforms points from the board coordinate system to the camera coordinate system.

    /* draw all marks */
    marker_img_ = img.clone();
    cv::aruco::drawDetectedMarkers(marker_img_, marker_corners, IDs);
    // for (size_t i = 0; i < IDs.size(); i++)
    //     cv::aruco::drawAxis(marker_img_, K_, dist_, rvs[i], tvs[i], 0.07);

    const float USEFUL_DISTANCE_THRESHOLD = 2.5; // 3 m

    for (size_t i = 0; i < IDs.size(); i++)
    {
        // float dist = cv::norm<double>(tvs[i]);
        float dist = tvs[i][2];
        if (dist > USEFUL_DISTANCE_THRESHOLD)
        {
            continue;
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
        marker_generate(IDs[i], marker_length_, tvs[i][0], tvs[i][1], tvs[i][2], _transform.getRotation(), _marker, color, ros::Duration(0.1));
        tf2::doTransform(_marker.pose, _marker.pose, transformStamped_r2c_);
        _marker.header.frame_id = "base_link";
        detected_markers_.markers.push_back(_marker);
        // cv::aruco::drawAxis(marker_img_, camera_matrix_, dist_coeffs_, rvs[i], tvs[i], 0.07);

        /* calculate observation mx, my, mtheta */
        cv::Vec3d tvec = tvs[i];
        cv::Vec3d rvec = rvs[i];
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        Eigen::Affine3d transform_matrix = tf2::transformToEigen(transformStamped_r2c_);
        // ROS_INFO_STREAM_ONCE("transformStamped_r2c_"<<transformStamped_r2c_<<std::endl<<"tvsc:"<<tvec<<std::endl<<"rvec:"<<rvec<<std::endl);
        Eigen::Matrix3d covariance;

        double x = tvec[2] + transform_matrix(0, 3);
        double y = -tvec[0] + transform_matrix(1, 3);
        double theta = atan2(-R.at<double>(0, 2), R.at<double>(2, 2));
        int aruco_id = IDs[i];
        CalculateCovariance(tvec, rvec, marker_corners[i], covariance);
        /* add to observation vector */
        if (covariance.norm() > 1)
            continue;
        obs.push_back(Observation(aruco_id, x, y, theta, covariance));
    } // for all detected markers
    return obs.size();
}

visualization_msgs::MarkerArray ArucoSlam::toRosMarkers(double scale)
{

    visualization_msgs::MarkerArray markers;
    int N = 0;
    for (int i = 4; i < mu_.rows(); i += 2)
    {
        double &mx = mu_(i - 1);
        double &my = mu_(i);

        /* 计算地图点的协方差椭圆角度以及轴长 */
        Eigen::Matrix3d sigma_m = sigma_.block(i - 1, i - 1, 2, 2); //协方差
        cv::Mat cvsigma_m = (cv::Mat_<double>(2, 2) << sigma_m(0, 0), sigma_m(0, 1), sigma_m(1, 0), sigma_m(1, 1));
        cv::Mat eigen_value, eigen_vector;
        cv::eigen(cvsigma_m, eigen_value, eigen_vector);
        double angle = atan2(eigen_vector.at<double>(0, 1), eigen_vector.at<double>(0, 0));
        double x_len = 2 * sqrt(eigen_value.at<double>(0, 0) * 5.991);
        double y_len = 2 * sqrt(eigen_value.at<double>(1, 0) * 5.991);

        /* 构造marker */
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time();
        marker.ns = "ekf_slam";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = mx;
        marker.pose.position.y = my;
        marker.pose.position.z = 0;
        tf2::Quaternion _q;
        _q.setRPY(0, 0, angle);
        marker.pose.orientation = tf2::toMsg(_q);
        marker.scale.x = scale * x_len;
        marker.scale.y = scale * y_len;
        marker.scale.z = 0.1 * scale * (x_len + y_len);
        marker.color.a = 0.8; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        markers.markers.push_back(marker);
    } // for all mpts

    return markers;
}

geometry_msgs::PoseWithCovarianceStamped ArucoSlam::toRosPose()
{
    /* 转换带协方差的机器人位姿 */
    geometry_msgs::PoseWithCovarianceStamped rpose;
    rpose.header.frame_id = "world";
    rpose.pose.pose.position.x = mu_(0);
    rpose.pose.pose.position.y = mu_(1);

    // rpose.pose.pose.orientation = tf2::createQuaternionMsgFromYaw(mu_(2));
    tf2::Quaternion QuaternionMsgFromYaw;
    QuaternionMsgFromYaw.setRPY(0, 0, mu_(2));
    rpose.pose.pose.orientation = tf2::toMsg(QuaternionMsgFromYaw);

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
    const static double PI = 3.1415926;
    static double Two_PI = 2.0 * PI;
    if (angle >= PI)
        angle -= Two_PI;
    if (angle < -PI)
        angle += Two_PI;
}

// 暴力搜素
bool ArucoSlam::checkLandmark(const int &aruco_id, int &landmark_idx)
{
    if (!aruco_id_map.empty() && aruco_id_map.end() != aruco_id_map.find(aruco_id))
    {
        landmark_idx = aruco_id_map.at(aruco_id);
        // ROS_INFO_STREAM("map:"<<aruco_id_map.at(aruco_id));
        return true;
    }
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
    double rerror = totalError / (double)objectPoints_.size();
    rerror = rerror * 2;
    covariance << rerror + 0.01, 0, 0, 0, rerror + 0.01, 0, 0, 0, rerror;
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