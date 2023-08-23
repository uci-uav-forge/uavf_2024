cmake_minimum_required(VERSION 3.5)
project(px4_offboard_mpc)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(uavf_ros2_msgs REQUIRED)
# find_package(<dependency> REQUIRED)


include_directories(include)
# Create Cpp executable
# add_executable(cpp_executable src/cpp_node.cpp)
# ament_target_dependencies(cpp_executable rclcpp)

# install( TARGETS
#  cpp_executable
#  DESTINATION lib/${PROJECT_NAME}
#)

# Install Python executables
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/commander_node.py
  scripts/trajectory_planner_node.py
  scripts/waypoint_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(_pytest_tests
    # tests/test.py
    # Add other test files here
  )
  foreach(_test_path ${_pytest_tests})
    get_filename_component(_test_name ${_test_path} NAME_WE)
    ament_add_pytest_test(${_test_name} ${_test_path}
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
      TIMEOUT 60
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  endforeach()
endif()


ament_package()
