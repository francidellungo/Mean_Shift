# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/francesca/Programmi/clion-2019.3.5/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/francesca/Programmi/clion-2019.3.5/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/francesca/CLionProjects/Mean_Shift

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/francesca/CLionProjects/Mean_Shift/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Mean_Shift.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Mean_Shift.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Mean_Shift.dir/flags.make

CMakeFiles/Mean_Shift.dir/genPoints.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/genPoints.cpp.o: ../genPoints.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Mean_Shift.dir/genPoints.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/genPoints.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/genPoints.cpp

CMakeFiles/Mean_Shift.dir/genPoints.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/genPoints.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/genPoints.cpp > CMakeFiles/Mean_Shift.dir/genPoints.cpp.i

CMakeFiles/Mean_Shift.dir/genPoints.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/genPoints.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/genPoints.cpp -o CMakeFiles/Mean_Shift.dir/genPoints.cpp.s

CMakeFiles/Mean_Shift.dir/Point.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/Point.cpp.o: ../Point.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Mean_Shift.dir/Point.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/Point.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/Point.cpp

CMakeFiles/Mean_Shift.dir/Point.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/Point.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/Point.cpp > CMakeFiles/Mean_Shift.dir/Point.cpp.i

CMakeFiles/Mean_Shift.dir/Point.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/Point.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/Point.cpp -o CMakeFiles/Mean_Shift.dir/Point.cpp.s

CMakeFiles/Mean_Shift.dir/main.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Mean_Shift.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/main.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/main.cpp

CMakeFiles/Mean_Shift.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/main.cpp > CMakeFiles/Mean_Shift.dir/main.cpp.i

CMakeFiles/Mean_Shift.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/main.cpp -o CMakeFiles/Mean_Shift.dir/main.cpp.s

CMakeFiles/Mean_Shift.dir/Cluster.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/Cluster.cpp.o: ../Cluster.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Mean_Shift.dir/Cluster.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/Cluster.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/Cluster.cpp

CMakeFiles/Mean_Shift.dir/Cluster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/Cluster.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/Cluster.cpp > CMakeFiles/Mean_Shift.dir/Cluster.cpp.i

CMakeFiles/Mean_Shift.dir/Cluster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/Cluster.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/Cluster.cpp -o CMakeFiles/Mean_Shift.dir/Cluster.cpp.s

CMakeFiles/Mean_Shift.dir/Utils.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/Utils.cpp.o: ../Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Mean_Shift.dir/Utils.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/Utils.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/Utils.cpp

CMakeFiles/Mean_Shift.dir/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/Utils.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/Utils.cpp > CMakeFiles/Mean_Shift.dir/Utils.cpp.i

CMakeFiles/Mean_Shift.dir/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/Utils.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/Utils.cpp -o CMakeFiles/Mean_Shift.dir/Utils.cpp.s

CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o: ../MeanShift.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/MeanShift.cpp

CMakeFiles/Mean_Shift.dir/MeanShift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/MeanShift.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/MeanShift.cpp > CMakeFiles/Mean_Shift.dir/MeanShift.cpp.i

CMakeFiles/Mean_Shift.dir/MeanShift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/MeanShift.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/MeanShift.cpp -o CMakeFiles/Mean_Shift.dir/MeanShift.cpp.s

CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o: CMakeFiles/Mean_Shift.dir/flags.make
CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o: ../MeanShiftUtils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o -c /home/francesca/CLionProjects/Mean_Shift/MeanShiftUtils.cpp

CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesca/CLionProjects/Mean_Shift/MeanShiftUtils.cpp > CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.i

CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesca/CLionProjects/Mean_Shift/MeanShiftUtils.cpp -o CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.s

# Object files for target Mean_Shift
Mean_Shift_OBJECTS = \
"CMakeFiles/Mean_Shift.dir/genPoints.cpp.o" \
"CMakeFiles/Mean_Shift.dir/Point.cpp.o" \
"CMakeFiles/Mean_Shift.dir/main.cpp.o" \
"CMakeFiles/Mean_Shift.dir/Cluster.cpp.o" \
"CMakeFiles/Mean_Shift.dir/Utils.cpp.o" \
"CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o" \
"CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o"

# External object files for target Mean_Shift
Mean_Shift_EXTERNAL_OBJECTS =

Mean_Shift: CMakeFiles/Mean_Shift.dir/genPoints.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/Point.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/main.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/Cluster.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/Utils.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/MeanShift.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/MeanShiftUtils.cpp.o
Mean_Shift: CMakeFiles/Mean_Shift.dir/build.make
Mean_Shift: CMakeFiles/Mean_Shift.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable Mean_Shift"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Mean_Shift.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Mean_Shift.dir/build: Mean_Shift

.PHONY : CMakeFiles/Mean_Shift.dir/build

CMakeFiles/Mean_Shift.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Mean_Shift.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Mean_Shift.dir/clean

CMakeFiles/Mean_Shift.dir/depend:
	cd /home/francesca/CLionProjects/Mean_Shift/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/francesca/CLionProjects/Mean_Shift /home/francesca/CLionProjects/Mean_Shift /home/francesca/CLionProjects/Mean_Shift/cmake-build-debug /home/francesca/CLionProjects/Mean_Shift/cmake-build-debug /home/francesca/CLionProjects/Mean_Shift/cmake-build-debug/CMakeFiles/Mean_Shift.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Mean_Shift.dir/depend

