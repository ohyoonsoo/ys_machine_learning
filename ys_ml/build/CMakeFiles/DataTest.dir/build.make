# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build

# Include any dependencies generated for this target.
include CMakeFiles/DataTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DataTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DataTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DataTest.dir/flags.make

CMakeFiles/DataTest.dir/dataTest.cpp.o: CMakeFiles/DataTest.dir/flags.make
CMakeFiles/DataTest.dir/dataTest.cpp.o: /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/dataTest.cpp
CMakeFiles/DataTest.dir/dataTest.cpp.o: CMakeFiles/DataTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DataTest.dir/dataTest.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DataTest.dir/dataTest.cpp.o -MF CMakeFiles/DataTest.dir/dataTest.cpp.o.d -o CMakeFiles/DataTest.dir/dataTest.cpp.o -c /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/dataTest.cpp

CMakeFiles/DataTest.dir/dataTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/DataTest.dir/dataTest.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/dataTest.cpp > CMakeFiles/DataTest.dir/dataTest.cpp.i

CMakeFiles/DataTest.dir/dataTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/DataTest.dir/dataTest.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/dataTest.cpp -o CMakeFiles/DataTest.dir/dataTest.cpp.s

# Object files for target DataTest
DataTest_OBJECTS = \
"CMakeFiles/DataTest.dir/dataTest.cpp.o"

# External object files for target DataTest
DataTest_EXTERNAL_OBJECTS =

DataTest: CMakeFiles/DataTest.dir/dataTest.cpp.o
DataTest: CMakeFiles/DataTest.dir/build.make
DataTest: data_handler/libdata.a
DataTest: exception/libysException.a
DataTest: CMakeFiles/DataTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DataTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DataTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DataTest.dir/build: DataTest
.PHONY : CMakeFiles/DataTest.dir/build

CMakeFiles/DataTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DataTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DataTest.dir/clean

CMakeFiles/DataTest.dir/depend:
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/build/CMakeFiles/DataTest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/DataTest.dir/depend
