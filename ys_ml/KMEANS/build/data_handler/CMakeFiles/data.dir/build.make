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
CMAKE_SOURCE_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build

# Include any dependencies generated for this target.
include data_handler/CMakeFiles/data.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include data_handler/CMakeFiles/data.dir/compiler_depend.make

# Include the progress variables for this target.
include data_handler/CMakeFiles/data.dir/progress.make

# Include the compile flags for this target's objects.
include data_handler/CMakeFiles/data.dir/flags.make

data_handler/CMakeFiles/data.dir/data.cpp.o: data_handler/CMakeFiles/data.dir/flags.make
data_handler/CMakeFiles/data.dir/data.cpp.o: /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data.cpp
data_handler/CMakeFiles/data.dir/data.cpp.o: data_handler/CMakeFiles/data.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object data_handler/CMakeFiles/data.dir/data.cpp.o"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT data_handler/CMakeFiles/data.dir/data.cpp.o -MF CMakeFiles/data.dir/data.cpp.o.d -o CMakeFiles/data.dir/data.cpp.o -c /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data.cpp

data_handler/CMakeFiles/data.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/data.dir/data.cpp.i"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data.cpp > CMakeFiles/data.dir/data.cpp.i

data_handler/CMakeFiles/data.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/data.dir/data.cpp.s"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data.cpp -o CMakeFiles/data.dir/data.cpp.s

data_handler/CMakeFiles/data.dir/data_handler.cpp.o: data_handler/CMakeFiles/data.dir/flags.make
data_handler/CMakeFiles/data.dir/data_handler.cpp.o: /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data_handler.cpp
data_handler/CMakeFiles/data.dir/data_handler.cpp.o: data_handler/CMakeFiles/data.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object data_handler/CMakeFiles/data.dir/data_handler.cpp.o"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT data_handler/CMakeFiles/data.dir/data_handler.cpp.o -MF CMakeFiles/data.dir/data_handler.cpp.o.d -o CMakeFiles/data.dir/data_handler.cpp.o -c /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data_handler.cpp

data_handler/CMakeFiles/data.dir/data_handler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/data.dir/data_handler.cpp.i"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data_handler.cpp > CMakeFiles/data.dir/data_handler.cpp.i

data_handler/CMakeFiles/data.dir/data_handler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/data.dir/data_handler.cpp.s"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler/data_handler.cpp -o CMakeFiles/data.dir/data_handler.cpp.s

# Object files for target data
data_OBJECTS = \
"CMakeFiles/data.dir/data.cpp.o" \
"CMakeFiles/data.dir/data_handler.cpp.o"

# External object files for target data
data_EXTERNAL_OBJECTS =

data_handler/libdata.a: data_handler/CMakeFiles/data.dir/data.cpp.o
data_handler/libdata.a: data_handler/CMakeFiles/data.dir/data_handler.cpp.o
data_handler/libdata.a: data_handler/CMakeFiles/data.dir/build.make
data_handler/libdata.a: data_handler/CMakeFiles/data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libdata.a"
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && $(CMAKE_COMMAND) -P CMakeFiles/data.dir/cmake_clean_target.cmake
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/data.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
data_handler/CMakeFiles/data.dir/build: data_handler/libdata.a
.PHONY : data_handler/CMakeFiles/data.dir/build

data_handler/CMakeFiles/data.dir/clean:
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler && $(CMAKE_COMMAND) -P CMakeFiles/data.dir/cmake_clean.cmake
.PHONY : data_handler/CMakeFiles/data.dir/clean

data_handler/CMakeFiles/data.dir/depend:
	cd /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/data_handler /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler /Users/ohyoonsoo/Documents/ys_machine_learning/ys_ml/KMEANS/build/data_handler/CMakeFiles/data.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : data_handler/CMakeFiles/data.dir/depend

