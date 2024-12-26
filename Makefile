# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -I/System/Volumes/Data/Users/pravinpb/vcpkg/installed/arm64-osx/include

# Source file and executable
SRC = toTest/test.cpp
EXEC = toTest/compiled

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC)

# Rule to run the program
run: $(EXEC)
	./$(EXEC)

# Rule to clean the build
clean:
	rm -f $(EXEC)

.PHONY: all run clean
