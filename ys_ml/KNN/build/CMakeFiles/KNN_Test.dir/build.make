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
CMAKE_SOURCE_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build

# Include any dependencies generated for this target.
include CMakeFiles/KNN_Test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/KNN_Test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/KNN_Test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KNN_Test.dir/flags.make

CMakeFiles/KNN_Test.dir/knnTest.cpp.o: CMakeFiles/KNN_Test.dir/flags.make
CMakeFiles/KNN_Test.dir/knnTest.cpp.o: /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/knnTest.cpp
CMakeFiles/KNN_Test.dir/knnTest.cpp.o: CMakeFiles/KNN_Test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KNN_Test.dir/knnTest.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/KNN_Test.dir/knnTest.cpp.o -MF CMakeFiles/KNN_Test.dir/knnTest.cpp.o.d -o CMakeFiles/KNN_Test.dir/knnTest.cpp.o -c /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/knnTest.cpp

CMakeFiles/KNN_Test.dir/knnTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KNN_Test.dir/knnTest.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/knnTest.cpp > CMakeFiles/KNN_Test.dir/knnTest.cpp.i

CMakeFiles/KNN_Test.dir/knnTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KNN_Test.dir/knnTest.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/knnTest.cpp -o CMakeFiles/KNN_Test.dir/knnTest.cpp.s

# Object files for target KNN_Test
KNN_Test_OBJECTS = \
"CMakeFiles/KNN_Test.dir/knnTest.cpp.o"

# External object files for target KNN_Test
KNN_Test_EXTERNAL_OBJECTS =

KNN_Test: CMakeFiles/KNN_Test.dir/knnTest.cpp.o
KNN_Test: CMakeFiles/KNN_Test.dir/build.make
KNN_Test: common_data/libcommon_data.a
KNN_Test: libknn.a
KNN_Test: data_handler/libdata.a
KNN_Test: exception/libysException.a
KNN_Test: CMakeFiles/KNN_Test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable KNN_Test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KNN_Test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KNN_Test.dir/build: KNN_Test
.PHONY : CMakeFiles/KNN_Test.dir/build

CMakeFiles/KNN_Test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/KNN_Test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/KNN_Test.dir/clean

CMakeFiles/KNN_Test.dir/depend:
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KNN/build/CMakeFiles/KNN_Test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/KNN_Test.dir/depend

