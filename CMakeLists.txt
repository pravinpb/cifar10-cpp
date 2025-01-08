# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.16)

# Set the project name and version
project(CIFAR10_CPP VERSION 1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Define the paths for OpenCV and vcpkg
set(OpenCV_DIR "/opt/homebrew/share/OpenCV")

# Include OpenCV and vcpkg headers
include_directories(
    /opt/homebrew/include/opencv4
    /System/Volumes/Data/Users/pravinpb/vcpkg/installed/arm64-osx/include
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/include
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/include
)

# Set the source files
set(SRCS
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/conv2d.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/maxpooling.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/dense.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/relu.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/softmax.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/src/loadbinfile.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/src/loadjsonfile.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/src/readimage.cpp
    /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/src/main.cpp
)

# Create an executable
add_executable(compiled ${SRCS})

# Link OpenCV libraries
find_package(OpenCV REQUIRED)
target_link_libraries(compiled ${OpenCV_LIBS})

# Specify additional compiler flags
target_compile_options(compiled PRIVATE -target arm64-apple-macos)
