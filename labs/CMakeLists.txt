cmake_minimum_required(VERSION 2.8.3)
project(prob_rob_labs)

find_package(catkin REQUIRED COMPONENTS
  opencv_apps
  roslaunch
)

catkin_package()

roslaunch_add_file_check(launch/jackal_and_door.launch)

catkin_python_setup()

install(DIRECTORY launch worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)



catkin_install_python(PROGRAMS bin/ekf_error_node
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/ekf_landmark_loc
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/ekf_localization
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/measurement_model
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/correspondence_matcher
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/measure_predictor
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/odometry_error_node
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/ekf_odometry_node
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/ground_truth_extraction
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/find_thres_noise
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/door_control_node
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/flaky_door_opener
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS bin/image_mean_feature_x
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
