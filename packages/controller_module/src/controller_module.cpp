/* includes //{ */

#include <ros/ros.h>

#include <nav_msgs/Odometry.h>

#include <random>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscribe_handler.h>
#include <geometry_msgs/PoseStamped.h>

#include <dynamic_reconfigure/server.h>
#include <controller_module/controller_paramsConfig.h>

#include <eigen3/Eigen/Eigen>

#include <mrs_uav_managers/controller.h>
#include <torch/torch.h>
#include <torch/script.h>

//}

namespace controller_module {

/* class ControllerModule //{ */

    class ControllerModule : public mrs_uav_managers::Controller {

        class Goal {
        public:
            Eigen::Vector3d position;
            Eigen::Vector3d direction;
            double tolerance;

            Goal(const Eigen::Vector3d &position, const Eigen::Vector3d &direction, double tolerance) :
                    position{position}, direction{direction}, tolerance{tolerance} {};

            [[nodiscard]] bool reached(const Eigen::Vector3d &prev_pos, const Eigen::Vector3d &current_pos) const {
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

        };


    public:
        ~ControllerModule() override = default;

        void initialize(const ros::NodeHandle &parent_nh, std::string name, std::string name_space,
                        double uav_mass,
                        std::shared_ptr<mrs_uav_managers::CommonHandlers_t> common_handlers) override;

        bool activate(const mrs_msgs::AttitudeCommand::ConstPtr &last_attitude_cmd) override;

        void deactivate() override;

        const mrs_msgs::AttitudeCommand::ConstPtr update(const mrs_msgs::UavState::ConstPtr &uav_state,
                                                         const mrs_msgs::PositionCommand::ConstPtr &control_reference) override;

        const mrs_msgs::ControllerStatus getStatus() override;

        void switchOdometrySource(const mrs_msgs::UavState::ConstPtr &new_uav_state) override;

        void resetDisturbanceEstimators() override;

        const mrs_msgs::DynamicsConstraintsSrvResponse::ConstPtr
        setConstraints(const mrs_msgs::DynamicsConstraintsSrvRequest::ConstPtr &cmd);

    private:
        bool is_initialized_ = false;
        bool is_active_ = false;

        std::shared_ptr<mrs_uav_managers::CommonHandlers_t> common_handlers_;

        std::vector<Goal> m_goals_to_reach;

        // Just default values to not leave fields uninitialized. All will be replaced in initialize() function
        double _uav_mass_ = 3.5;
        double hover_thrust_ = 0;
        double m_uav_mass = 3.5;
        size_t m_current_goal_index = 0;
//        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> points_to_visit;


        // | --------------- dynamic reconfigure server --------------- |

        boost::recursive_mutex mutex_drs_;
        typedef controller_module::controller_paramsConfig DrsParams_t;
        typedef dynamic_reconfigure::Server<DrsParams_t> Drs_t;
        boost::shared_ptr<Drs_t> drs_;

        void callbackDrs(controller_module::controller_paramsConfig &params, uint32_t level);

        DrsParams_t params_;
        std::mutex mutex_params_;


        std::mutex m_uav_state_mutex;
        std::mutex m_model_mutex;
        torch::jit::script::Module m_policy_module;
        Eigen::Vector3d m_prev_pos;

//        static bool goal_passed(in)
    };

//}

// --------------------------------------------------------------
// |                   controller's interface                   |
// --------------------------------------------------------------

/* initialize() //{ */

    void ControllerModule::initialize(const ros::NodeHandle &parent_nh, [[maybe_unused]] const std::string name,
                                      const std::string name_space,
                                      const double uav_mass,
                                      std::shared_ptr<mrs_uav_managers::CommonHandlers_t> common_handlers) {

        ros::NodeHandle nh_(parent_nh, name_space);

        common_handlers_ = common_handlers;
        _uav_mass_ = uav_mass;

        ros::Time::waitForValid();

        // | ------------------- loading parameters ------------------- |

        mrs_lib::ParamLoader param_loader(nh_, "ControllerModule");

        /* TODO: move this to parameters inside a normal namespace. Current problem is that
        if working with ParamLoader, a custom prefix should be set or control_manager.launch file should be changed */
        std::string policy_filename = "/home/mrs/flightsim/test.pt";

        // TODO: Read this from some parameter
        m_uav_mass = 3.5;

        // TODO: read this from some config file of ROS parameter too
        m_current_goal_index = 0;
        m_prev_pos = {0, 0, 0};
        m_goals_to_reach.emplace_back(Eigen::Vector3d{2.0, 2.0, 2.0}, Eigen::Vector3d{1.0, 1.0, 1.0}, 0.5);
        m_goals_to_reach.emplace_back(Eigen::Vector3d{3.0, 3.0, 4.0}, Eigen::Vector3d{0.0, 0.0, 1.0}, 0.5);
        m_goals_to_reach.emplace_back(Eigen::Vector3d{0.0, 0.0, 5.0}, Eigen::Vector3d{-1.0, -1.0, 0.0}, 0.5);
        m_goals_to_reach.emplace_back(Eigen::Vector3d{-2.0, 0.0, 3.0}, Eigen::Vector3d{0.0, 0.0, -1.0}, 0.5);


        if (!param_loader.loadedSuccessfully()) {
            ROS_ERROR("[ControllerModule]: Could not load all parameters!");
            ros::shutdown();
        }

        try {
            m_policy_module = torch::jit::load(policy_filename);
        } catch (const c10::Error &e) {
            ROS_ERROR_STREAM("[ControllerModule]: failed to load jit script! Error: " << e.what());
            ros::shutdown();
        }
        ROS_INFO("[ControllerModule]: policy file loaded successfully!");


        // | ----------- calculate the default hover thrust ----------- |
        hover_thrust_ = mrs_lib::quadratic_thrust_model::forceToThrust(common_handlers_->motor_params,
                                                                       _uav_mass_ * common_handlers_->g);

        // | --------------------------- drs -------------------------- |

        drs_.reset(new Drs_t(mutex_drs_, nh_));
        Drs_t::CallbackType f = boost::bind(&ControllerModule::callbackDrs, this, _1, _2);
        drs_->setCallback(f);

        // | ----------------------- finish init ---------------------- |

        ROS_INFO("[ControllerModule]: initialized");

        is_initialized_ = true;
    }

//}

/* activate() //{ */

    bool ControllerModule::activate(const mrs_msgs::AttitudeCommand::ConstPtr &last_attitude_cmd) {

        if (last_attitude_cmd == mrs_msgs::AttitudeCommand::Ptr()) {

            ROS_WARN("[ControllerModule]: activated without getting the last controller's command");

            return false;
        }

        is_active_ = true;

        return true;
    }

//}

/* deactivate() //{ */

    void ControllerModule::deactivate(void) {

        is_active_ = false;

        ROS_INFO("[ControllerModule]: deactivated");
    }

//}

/* update() //{ */

    const mrs_msgs::AttitudeCommand::ConstPtr
    ControllerModule::update([[maybe_unused]] const mrs_msgs::UavState::ConstPtr &uav_state_p,
                             [[maybe_unused]] const mrs_msgs::PositionCommand::ConstPtr &control_reference) {

//        ROS_INFO_ONCE("[ControllerModule]: update()");
        if (!is_active_) {
            return mrs_msgs::AttitudeCommand::ConstPtr();
        }


        mrs_msgs::UavState uav_state;
        {
            std::scoped_lock l{m_uav_state_mutex};
            uav_state = *uav_state_p;
        }

        // Check if the UAV crossed the next waypoint and if it has, update the waypoint to the next one
        Eigen::Vector3d current_position{uav_state.pose.position.x, uav_state.pose.position.y,
                                         uav_state.pose.position.z};
        if (m_goals_to_reach[m_current_goal_index].reached(m_prev_pos, current_position)) {
            ROS_INFO_STREAM(
                    "[ControllerModule]: Reached the waypoint: " << m_goals_to_reach[m_current_goal_index].position);
            m_current_goal_index = std::min(m_current_goal_index + 1, m_goals_to_reach.size() - 1);
            ROS_INFO_STREAM(
                    "[ControllerModule]: Reached the waypoint. Next one to follow is number " << m_current_goal_index);
        }
        m_prev_pos = current_position;

        // Get rotation matrix from orientation quaternion
        Eigen::Quaterniond orientation_q;
        orientation_q.x() = uav_state.pose.orientation.x;
        orientation_q.y() = uav_state.pose.orientation.y;
        orientation_q.z() = uav_state.pose.orientation.z;
        orientation_q.w() = uav_state.pose.orientation.w;
        Eigen::Matrix3d R = orientation_q.normalized().toRotationMatrix();

        // Fill in model inputs. TODO: check if these are correct values
        float model_input[18];
        model_input[0] = static_cast<float>(uav_state.pose.position.x);
        model_input[1] = static_cast<float>(uav_state.pose.position.y);
        model_input[2] = static_cast<float>(uav_state.pose.position.z);

        model_input[3] = static_cast<float>(uav_state.velocity.linear.x);
        model_input[4] = static_cast<float>(uav_state.velocity.linear.y);
        model_input[5] = static_cast<float>(uav_state.velocity.linear.z);

        // Fill in orientation matrix values
        double *R_data = R.data();
        for (int i = 6; i < 6 + 9; ++i) {
            model_input[i] = static_cast<float>(R_data[i - 6]);
        }

        model_input[15] = static_cast<float>(m_goals_to_reach[m_current_goal_index].position.x() -
                                             uav_state.pose.position.x);
        model_input[16] = static_cast<float>(m_goals_to_reach[m_current_goal_index].position.y() -
                                             uav_state.pose.position.y);
        model_input[17] = static_cast<float>(m_goals_to_reach[m_current_goal_index].position.z() -
                                             uav_state.pose.position.z);

        torch::Tensor input_tensor = torch::from_blob(model_input, {1, 18});
        torch::IValue input_ivalue{input_tensor};

        torch::Tensor model_output;
        {
            std::scoped_lock l{m_model_mutex};
            model_output = m_policy_module.forward({input_ivalue}).toTensor();
        }

        mrs_msgs::AttitudeCommand::Ptr output_command(new mrs_msgs::AttitudeCommand);
        output_command->header.stamp = ros::Time::now();

        double model_thrust = mrs_lib::quadratic_thrust_model::forceToThrust(common_handlers_->motor_params,
                                                                             model_output[0][0].item<float>() *
                                                                             m_uav_mass);
        output_command->thrust = model_thrust;
        output_command->mode_mask = output_command->MODE_ATTITUDE_RATE;

        output_command->mass_difference = 0;
        output_command->total_mass = m_uav_mass;

        output_command->attitude_rate.x = model_output[0][1].item<float>();
        output_command->attitude_rate.y = model_output[0][2].item<float>();
        output_command->attitude_rate.z = model_output[0][3].item<float>();

        ROS_DEBUG_STREAM("[ControllerModule]: UAV position: \n" << uav_state.pose.position << "\n" <<
                                                                "UAV rotation: " << R << "\n" <<
                                                                "Current reference point: "
                                                                << m_goals_to_reach[m_current_goal_index].position <<
                                                                "Desired body rates: "
                                                                << model_output[0][1].item<float>() << ", "
                                                                << model_output[0][2].item<float>() << ", "
                                                                << model_output[0][3].item<float>() << "\n" <<
                                                                "Desired thrust: " << output_command->thrust << "\n\n");

        output_command->controller_enforcing_constraints = false;

        output_command->controller = "ControllerModule";

        return output_command;

    }

//}

// | ------------------- DO NOT MODIFY BELOW ------------------ |

/* //{ callbackDrs() */

    void
    ControllerModule::callbackDrs(controller_module::controller_paramsConfig &params, [[maybe_unused]] uint32_t level) {

        mrs_lib::set_mutexed(mutex_params_, params, params_);

        ROS_INFO("[ControllerModule]: DRS updated");
    }

//}

/* getStatus() //{ */

    const mrs_msgs::ControllerStatus ControllerModule::getStatus() {

        mrs_msgs::ControllerStatus controller_status;

        controller_status.active = is_active_;

        return controller_status;
    }

//}

/* switchOdometrySource() //{ */

    void ControllerModule::switchOdometrySource([[maybe_unused]] const mrs_msgs::UavState::ConstPtr &new_uav_state) {
    }

//}

/* resetDisturbanceEstimators() //{ */

    void ControllerModule::resetDisturbanceEstimators(void) {
    }

//}

/* setConstraints() //{ */

    const mrs_msgs::DynamicsConstraintsSrvResponse::ConstPtr ControllerModule::setConstraints([
                                                                                              [maybe_unused]] const mrs_msgs::DynamicsConstraintsSrvRequest::ConstPtr &constraints) {

        return mrs_msgs::DynamicsConstraintsSrvResponse::ConstPtr(new mrs_msgs::DynamicsConstraintsSrvResponse());
    }

//}

}  // namespace controller_module

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(controller_module::ControllerModule, mrs_uav_managers::Controller)
