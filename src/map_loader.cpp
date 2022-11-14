#include "aruco_slam/map_loader.h"

MapLoader::MapLoader(const string &file_path){
    loadMap(file_path);
}

void MapLoader::loadMap(const string &file_path)
{
    ROS_INFO("\n loading from: %s", file_path.c_str());
    std::ifstream f(file_path);
    std::string line;
    real_map_.markers.clear();
    if (!f.good())
    {
        ROS_ERROR("%s - %s", strerror(errno), file_path.c_str());
        return;
    }
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
            // Probably garbage data; inform user and throw an exception
            ROS_ERROR("Malformed input: %s", line.c_str());
            real_map_.markers.clear();
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
        // map file formte: id, length, x, y, z, yaw, pitch, roll;
        addMarker(id, length, x, y, z, yaw, pitch, roll);
    }
    ROS_INFO("loading %s complete (%d markers)", file_path.c_str(), static_cast<int>(real_map_.markers.size()));
}

void MapLoader::addMarker(const int &id, const double &length, const double &x, const double &y, const double &z,
                   const double &yaw, const double &pitch, const double &roll)
{
    // Create transform
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);

    // Add marker to array
    real_map_.markers.push_back(generateMarker(id, length, x, y, z, q));
}

visualization_msgs::Marker MapLoader::generateMarker(const int &id, const double &length,
                                                         const double &x, const double &y, const double &z, const tf2::Quaternion &q)
{
    visualization_msgs::Marker marker;
    marker.id = id;
    marker.header.frame_id = "world";
    marker.type = visualization_msgs::Marker::CUBE;
    marker.scale.x = length;
    marker.scale.y = length;
    marker.scale.z = 0.01;

    marker.color.a = 0.5;
    marker.color.b = 1;
    marker.color.g = 1;
    marker.color.r = 1;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation = tf2::toMsg(q);
    marker.lifetime = ros::Duration(0);
    return marker;
}