/* includes //{ */

#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <mrs_msgs/ReferenceSrv.h>

#include <random>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscribe_handler.h>
#include <geometry_msgs/PoseStamped.h>

#include <dynamic_reconfigure/server.h>
#include <rl_controller/controller_paramsConfig.h>

#include <eigen3/Eigen/Eigen>

#include <mrs_uav_managers/controller.h>
#include <torch/torch.h>
#include <torch/script.h>

//}

namespace rl_controller {


    class RLController : public mrs_uav_managers::Controller {

        ros::ServiceServer goal_service_server;

    public:
        ~RLController() override = default;

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
        // Just default values to not leave fields uninitialized. All will be replaced in initialize() function
        double _uav_mass_ = 3.5;
        double hover_thrust_ = 0;
        double m_uav_mass = 3.5;

        Eigen::Vector3d m_current_goal;

        // | --------------- dynamic reconfigure server --------------- |

        boost::recursive_mutex mutex_drs_;
        typedef rl_controller::controller_paramsConfig DrsParams_t;
        typedef dynamic_reconfigure::Server<DrsParams_t> Drs_t;
        boost::shared_ptr<Drs_t> drs_;

        void callbackDrs(rl_controller::controller_paramsConfig &params, uint32_t level);

        DrsParams_t params_;
        std::mutex mutex_params_;


        std::mutex m_uav_state_mutex;
        std::mutex m_model_mutex;
        torch::jit::script::Module m_policy_module;

        ros::ServiceServer m_set_goal_service_server;

        bool callback_set_goal(mrs_msgs::ReferenceSrv::Request &req, mrs_msgs::ReferenceSrv::Response &res);

//        static bool goal_passed(in)
    };

//}

// --------------------------------------------------------------
// |                   controller's interface                   |
// --------------------------------------------------------------

/* initialize() //{ */

    void RLController::initialize(const ros::NodeHandle &parent_nh, [[maybe_unused]] const std::string name,
                                  const std::string name_space,
                                  const double uav_mass,
                                  std::shared_ptr<mrs_uav_managers::CommonHandlers_t> common_handlers) {

        ros::NodeHandle nh_(parent_nh, name_space);

        common_handlers_ = common_handlers;
        _uav_mass_ = uav_mass;

        ros::Time::waitForValid();

        // | ------------------- loading parameters ------------------- |

        mrs_lib::ParamLoader param_loader(nh_, "RLController");

        /* TODO: move this to parameters inside a normal namespace. Current problem is that
        if working with ParamLoader, a custom prefix should be set or control_manager.launch file should be changed */
        std::string policy_filename = "/home/mrs/flightsim/test.pt";

        // TODO: Read this from some parameter
        m_uav_mass = 3.5;

        if (!param_loader.loadedSuccessfully()) {
            ROS_ERROR("[RLController]: Could not load all parameters!");
            ros::shutdown();
        }

        m_set_goal_service_server = nh_.advertiseService("set_goal", &RLController::callback_set_goal, this);

        try {
            m_policy_module = torch::jit::load(policy_filename);
        } catch (const c10::Error &e) {
            ROS_ERROR_STREAM("[RLController]: failed to load jit script! Error: " << e.what());
            ros::shutdown();
        }
        ROS_INFO("[RLController]: policy file loaded successfully!");


        // | ----------- calculate the default hover thrust ----------- |
        hover_thrust_ = mrs_lib::quadratic_thrust_model::forceToThrust(common_handlers_->motor_params,
                                                                       _uav_mass_ * common_handlers_->g);

        // | --------------------------- drs -------------------------- |

        drs_.reset(new Drs_t(mutex_drs_, nh_));
        Drs_t::CallbackType f = boost::bind(&RLController::callbackDrs, this, _1, _2);
        drs_->setCallback(f);

        // | ----------------------- finish init ---------------------- |

        ROS_INFO("[RLController]: initialized");

        is_initialized_ = true;
    }

//}

    bool RLController::callback_set_goal(mrs_msgs::ReferenceSrv::Request &req, mrs_msgs::ReferenceSrv::Response &res) {
        m_current_goal.x() = req.reference.position.x;
        m_current_goal.y() = req.reference.position.y;
        m_current_goal.z() = req.reference.position.z;

        ROS_INFO_STREAM("[RLController]: New goal set: \n" << m_current_goal);

        res.message = "New goal set";
        res.success = true;
        return true;
    }


/* activate() //{ */

    bool RLController::activate(const mrs_msgs::AttitudeCommand::ConstPtr &last_attitude_cmd) {

        if (last_attitude_cmd == mrs_msgs::AttitudeCommand::Ptr()) {

            ROS_WARN("[RLController]: activated without getting the last controller's command");

            return false;
        }

        is_active_ = true;

        return true;
    }

//}

/* deactivate() //{ */

    void RLController::deactivate(void) {

        is_active_ = false;

        ROS_INFO("[RLController]: deactivated");
    }

//}

/* update() //{ */

    const mrs_msgs::AttitudeCommand::ConstPtr
    RLController::update([[maybe_unused]] const mrs_msgs::UavState::ConstPtr &uav_state_p,
                         [[maybe_unused]] const mrs_msgs::PositionCommand::ConstPtr &control_reference) {

//        ROS_INFO_ONCE("[RLController]: update()");
        if (!is_active_) {
            return mrs_msgs::AttitudeCommand::ConstPtr();
        }


        mrs_msgs::UavState uav_state;
        {
            std::scoped_lock l{m_uav_state_mutex};
            uav_state = *uav_state_p;
        }


        // Get rotation matrix from orientation quaternion
        Eigen::Quaterniond orientation_q;
        orientation_q.x() = uav_state.pose.orientation.x;
        orientation_q.y() = uav_state.pose.orientation.y;
        orientation_q.z() = uav_state.pose.orientation.z;
        orientation_q.w() = uav_state.pose.orientation.w;
        Eigen::Matrix3d R = orientation_q.normalized().toRotationMatrix();

        // Fill in model inputs
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

        model_input[15] = static_cast<float>(m_current_goal.x() - uav_state.pose.position.x);
        model_input[16] = static_cast<float>(m_current_goal.y() - uav_state.pose.position.y);
        model_input[17] = static_cast<float>(m_current_goal.z() - uav_state.pose.position.z);

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

        ROS_DEBUG_STREAM("[RLController]: UAV position: \n" << uav_state.pose.position << "\n" <<
                                                            "UAV rotation: " << R << "\n" <<
                                                            "Current reference point: "
                                                            << m_current_goal <<
                                                            "Desired body rates: "
                                                            << model_output[0][1].item<float>() << ", "
                                                            << model_output[0][2].item<float>() << ", "
                                                            << model_output[0][3].item<float>() << "\n" <<
                                                            "Desired thrust: " << output_command->thrust << "\n\n");

        output_command->controller_enforcing_constraints = false;

        output_command->controller = "RLController";

        return output_command;

    }

//}

// | ------------------- DO NOT MODIFY BELOW ------------------ |

/* //{ callbackDrs() */

    void
    RLController::callbackDrs(rl_controller::controller_paramsConfig &params, [[maybe_unused]] uint32_t level) {

        mrs_lib::set_mutexed(mutex_params_, params, params_);

        ROS_INFO("[RLController]: DRS updated");
    }

//}

/* getStatus() //{ */

    const mrs_msgs::ControllerStatus RLController::getStatus() {

        mrs_msgs::ControllerStatus controller_status;

        controller_status.active = is_active_;

        return controller_status;
    }

//}

/* switchOdometrySource() //{ */

    void RLController::switchOdometrySource([[maybe_unused]] const mrs_msgs::UavState::ConstPtr &new_uav_state) {
    }

//}

/* resetDisturbanceEstimators() //{ */

    void RLController::resetDisturbanceEstimators(void) {
    }

//}

/* setConstraints() //{ */

    const mrs_msgs::DynamicsConstraintsSrvResponse::ConstPtr RLController::setConstraints([
                                                                                          [maybe_unused]] const mrs_msgs::DynamicsConstraintsSrvRequest::ConstPtr &constraints) {

        return mrs_msgs::DynamicsConstraintsSrvResponse::ConstPtr(new mrs_msgs::DynamicsConstraintsSrvResponse());
    }

//}

}  // namespace controller_module

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(rl_controller::RLController, mrs_uav_managers::Controller)
