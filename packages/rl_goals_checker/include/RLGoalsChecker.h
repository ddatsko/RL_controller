#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>
#include <cstdlib>
#include <functional>

#include <mrs_lib/transformer.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <string>


namespace rl_goals_checker {
    class Goal {
    public:
        Eigen::Vector3d position;
        Eigen::Vector3d direction;
        double tolerance;

        Goal() = default;

        Goal(const Eigen::Vector3d &position, const Eigen::Vector3d &direction, double tolerance) :
                position{position}, direction{direction}, tolerance{tolerance} {};

        [[nodiscard]] bool reached(const Eigen::Vector3d &prev_pos, const Eigen::Vector3d &current_pos) const;
    };

    /*!
     * Read goals from environment file. In case of error returns an empty vector
     * @param filename File path to the environment configuration file
     * @return Goals read from file ot empty vector in case of any error
     */
    std::vector<Goal> read_yaml_config_file(const std::string &filename);

    class RLGoalsChecker : public nodelet::Nodelet {
    public:
        void onInit() override;

    private:
        std::string m_goal_frame_id;
        std::string m_uav_name;
        size_t m_current_goal = 0;
        std::vector<Goal> m_goals_to_visit;

        ros::ServiceClient m_set_goal_service_client;
        ros::Subscriber m_odometry_subscriber;

        ros::Publisher m_current_goal_publisher;
        ros::Publisher m_pub_all_goals;

        Eigen::Vector3d m_previous_position;
        bool m_previous_position_received = false;

        void m_odometry_callback(const nav_msgs::Odometry &odom_msg);

        void send_current_goal();

        ros::Timer m_tim_goals;
        void tim_markers_pb([[maybe_unused]] const ros::TimerEvent &ev);

        mrs_lib::Transformer m_transformer;
    };

}
