/** \file        map_loader.h
 *  \author   	Yichen Liang (liangyichen666@gmail.com)
 *  \copyright   GNU General Public License (GPL)
 *  \brief   	Load the real accurate markers map
 *  \version	    V0.01
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
#ifndef MAP_LOADER_H
#define MAP_LOADER_H

#include <fstream>
#include <string>

#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <visualization_msgs/MarkerArray.h>

using namespace std;

class MapLoader
{
public:
    MapLoader() = delete;
    /**
     * @brief Construct a new Map Loader object
     *
     * The map will be parse from the file in the filepath to the class member real_map_
     * when the object be instantiated
     * @param file_path the path of the map file
     */
    MapLoader(const string &file_path);
    /**
     * @brief get the markers that have been added to the map, for visualization.
     *
     * @return visualization_msgs::MarkerArray
     */
    visualization_msgs::MarkerArray toRosRealMapMarkers() { return real_map_; };

private:
    visualization_msgs::MarkerArray real_map_;
    void loadMap(const string &file_path);
    void addMarker(const int &id, const double &length, const double &x, const double &y, const double &z,
                   const double &yaw, const double &pitch, const double &roll);
    visualization_msgs::Marker generateMarker(const int &id, const double &length,
                                              const double &x, const double &y, const double &z, const tf2::Quaternion &q);
};
#endif