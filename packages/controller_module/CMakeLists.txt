cmake_minimum_required(VERSION 3.1.2)
project(controller_module)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        cmake_modules
        mrs_msgs
        nav_msgs
        std_srvs
        nodelet
        mrs_uav_managers
        mrs_lib
        dynamic_reconfigure
        )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

# TODO: replace this with some general path (maybe from env) for now, everyone after cloning needs to replace this with own path
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)
set(CMAKE_PREFIX_PATH /usr/local/lib/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("Torch libraries found at: ${TORCH_LIBRARIES}")

generate_dynamic_reconfigure_options(
        cfg/controller_params.cfg
)

catkin_package(
        CATKIN_DEPENDS roscpp nodelet mrs_msgs mrs_uav_managers mrs_lib std_srvs dynamic_reconfigure
        DEPENDS Eigen
        LIBRARIES ControllerModule
)

include_directories(
        /home/mrs/workspace/devel/include
        ${catkin_INCLUDE_DIRS}
        ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
)

# wrapper

add_library(ControllerModule
        src/controller_module.cpp
        )

add_dependencies(ControllerModule
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )

target_link_libraries(ControllerModule
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
        )
