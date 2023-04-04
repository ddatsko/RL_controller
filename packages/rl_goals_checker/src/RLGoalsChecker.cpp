#include <RLGoalsChecker.h>
#include <pluginlib/class_list_macros.h>
#include <mrs_lib/param_loader.h>
#include <mrs_lib/transformer.h>
#include <yaml-cpp/yaml.h>
#include <mrs_msgs/ReferenceStampedSrv.h>


namespace rl_goals_checker {

    std::vector<Goal> read_yaml_config_file(const std::string &filename) {
        YAML::Node root_node;
        try {
            root_node = YAML::LoadFile(filename);
        } catch (const YAML::Exception &e) {
            ROS_ERROR_STREAM("[RLGoalsChecker]: Could not read environment config file: " << filename);
            return {};
        }
        if (root_node["environment_configuration"] && root_node["environment_configuration"]["goals"]) {
            std::vector<Goal> read_goals;
            auto goals = root_node["environment_configuration"]["goals"];
            for (auto &&goal: goals) {
                try {
                    Goal g;
                    g.tolerance = goal["tolerance"].as<float>();
                    g.position.x() = goal["position"][0].as<float>();
                    g.position.y() = goal["position"][1].as<float>();
                    g.position.z() = goal["position"][2].as<float>();

                    g.direction.x() = goal["direction"][0].as<float>();
                    g.direction.y() = goal["direction"][1].as<float>();
                    g.direction.z() = goal["direction"][2].as<float>();
                    g.direction.normalize();

                    read_goals.push_back(g);

                } catch (const std::runtime_error &e) {
                    ROS_ERROR_STREAM("[RLGoalsChecker]: Error while reading configuration file: " << e.what());
                    return {};
                }
            }
            return read_goals;
        } else {
            ROS_ERROR_STREAM("[RLGoalsChecker]: Error. No environment_configuration/goals in file " << filename);
            return {};
        }
        return {};
    }


    bool Goal::reached(const Eigen::Vector3d &prev_pos, const Eigen::Vector3d &current_pos) const {
        Eigen::Vector3d v = current_pos - prev_pos;  // step movement vector
        Eigen::Vector3d n = direction;           // goal direction vector
        double vTn = v.dot(n);          // dot product of the vector

        // either runs in parallel or passed in other direction
        if (vTn <= 0.0) return false;

        Eigen::Vector3d a = prev_pos - position;  // vector from initial position to goal position
        // how many times the current would be taken to pass the goal (can be
        // negative)
        double t = -(a.dot(n) / vTn);
        if (t >= 0 && t <= 1) {  // only these values mean agent passed by the goal in
            // the current time-step
            Eigen::Vector3d intersect = a + t * v;  // intersection point
            if (intersect.norm() <= tolerance)  // is the intersection point within tolerance
                return true;

        }
        return false;
    }

    void RLGoalsChecker::onInit() {
        /* obtain node handle */
        ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

        /* waits for the ROS to publish clock */
        ros::Time::waitForValid();


        std::string environment_config_filename;
        // | ------------------- load ros parameters ------------------ |
        mrs_lib::ParamLoader pl(nh, "RLGoalsChecker");

        pl.loadParam("UAV_NAME", m_uav_name);
        pl.loadParam("environment_configuration_file", environment_config_filename);
        pl.loadParam("goal_frame_id", m_goal_frame_id);

        if (!pl.loadedSuccessfully()) {
            ROS_ERROR("[RLGoalsChecker]: failed to load non-optional parameters!");
            ros::shutdown();
        } else {
            ROS_INFO_ONCE("[RLGoalsChecker]: loaded parameters");
        }

        // Read environment configuration
        m_goals_to_visit = read_yaml_config_file(environment_config_filename);
        if (m_goals_to_visit.empty()) {
            ROS_ERROR_STREAM("[RLGoalsChecker]: Error. No goals read from file " << environment_config_filename <<
                                                                                 ". Nodelet will not post any goals");
            ros::shutdown();

        } else {
            ROS_INFO_STREAM("[RLGoalsChecker]: Successfully read " << m_goals_to_visit.size() << " goals from file");
        }

        m_current_goal_publisher = nh.advertise<geometry_msgs::PoseStamped>("goals_out", 10);

        m_set_goal_service_client = nh.serviceClient<mrs_msgs::ReferenceStampedSrv>(
                "/" + m_uav_name + "/control_manager/rl_controller/set_goal");
        ROS_INFO_STREAM("[RLGoalsChecker]: Waiting for existence of service server...");
        m_set_goal_service_client.waitForExistence();
        ROS_INFO_STREAM("[RLGoalsChecker]: Service server is present. Continuing");
        send_current_goal();

        m_odometry_subscriber = nh.subscribe("odometry_in", 10, &RLGoalsChecker::m_odometry_callback,
                                             this);

        m_transformer = mrs_lib::Transformer("RLGoalsChecker");

        ROS_INFO_ONCE("[RLGoalsChecker]: initialized");
    }

    void RLGoalsChecker::m_odometry_callback(const nav_msgs::Odometry &odom_msg) {
        auto odom_position = m_transformer.transformSingle(odom_msg, m_goal_frame_id);
        if (!odom_position.has_value()) {
            ROS_ERROR_STREAM("[RLGoalsChecker]: Could not convert odometry to frame " << m_goal_frame_id << ". Skipping measurement");
            return;
        }

        Eigen::Vector3d position = {odom_position->pose.pose.position.x,
                                    odom_position->pose.pose.position.y,
                                    odom_position->pose.pose.position.z};

        if (!m_previous_position_received) {
            m_previous_position_received = true;
            m_previous_position = position;
            return;
        }

        if (m_goals_to_visit[m_current_goal].reached(m_previous_position, position)) {
            ROS_INFO_STREAM("[RLGoalsChecker]: UAV reached goal: \n" << m_goals_to_visit[m_current_goal].position);

            // TODO: change the behaviour when last goal reached. Maybe, change controller to MPC or so
            m_current_goal = std::min(m_current_goal + 1, m_goals_to_visit.size() - 1);

            // If the goal is reached -- send a new one to the controller
            send_current_goal();
        }
        m_previous_position = position;
    }

    void RLGoalsChecker::send_current_goal() {
        mrs_msgs::ReferenceStampedSrv set_goal_request;
        set_goal_request.request.header.frame_id = m_goal_frame_id;

        set_goal_request.request.reference.position.x = m_goals_to_visit[m_current_goal].position.x();
        set_goal_request.request.reference.position.y = m_goals_to_visit[m_current_goal].position.y();
        set_goal_request.request.reference.position.z = m_goals_to_visit[m_current_goal].position.z();

        geometry_msgs::PoseStamped goal;
        goal.header = set_goal_request.request.header;
        goal.pose.position = set_goal_request.request.reference.position;

        // Call the service as soon as possible
        m_set_goal_service_client.call(set_goal_request);

        Eigen::Vector3d initial_rotation{1, 0, 0};
        auto orientation = Eigen::Quaterniond().setFromTwoVectors(initial_rotation, m_goals_to_visit[m_current_goal].direction);
        goal.pose.orientation.x = orientation.x();
        goal.pose.orientation.y = orientation.y();
        goal.pose.orientation.z = orientation.z();
        goal.pose.orientation.w = orientation.w();

        m_current_goal_publisher.publish(goal);
    }


}  // namespace ro_goals_checker  

/* every nodelet must export its class as nodelet plugin */
PLUGINLIB_EXPORT_CLASS(rl_goals_checker::RLGoalsChecker, nodelet::Nodelet)
