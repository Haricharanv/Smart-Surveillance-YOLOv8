cmake_minimum_required(VERSION 3.10)
project(YOLOv8SurveillanceInference)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package (required for image structures and DNN module)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the C++ executable
add_executable(inference_demo inference_demo.cpp)

# Link the executable with OpenCV libraries
target_link_libraries(inference_demo ${OpenCV_LIBS})

# Print configuration info
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
