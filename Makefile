# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -I/opt/homebrew/include/opencv4 -I/System/Volumes/Data/Users/pravinpb/vcpkg/installed/arm64-osx/include -I/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/include -I/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/include -target arm64-apple-macos

Source files and object files
SRCS = /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/conv2d.cpp \
       /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/maxpooling.cpp \
       /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/dense.cpp \
	   /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/relu.cpp \
	   /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Operators/src/softmax.cpp \
       /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/src/loadbinfile.cpp \
	   /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/utilities/src/loadjsonfile.cpp \
       /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/src/main.cpp

# SRCS = /Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/toTest/test4.cpp


# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = compiled

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

# Rule to compile each .cpp file into a .o file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to run the program
run: $(EXEC)
	./$(EXEC)

# Rule to clean the build
clean:
	rm -f $(OBJS) $(EXEC)

.PHONY: all run clean
