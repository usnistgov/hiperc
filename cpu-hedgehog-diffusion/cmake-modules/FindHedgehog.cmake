# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# - Find Hedgehog includes and required compiler flags and library dependencies
# Dependencies: C++11 support and threading library
#
# The Hedgehog_CXX_FLAGS should be added to the CMAKE_CXX_FLAGS
#
# This module defines
#  Hedgehog_INCLUDE_DIRS
#  Hedgehog_LIBRARIES
#  Hedgehog_CXX_FLAGS
#  Hedgehog_FOUND
#


# Ensure C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif (MSVC)

# Try to found Hedgehog
SET(Hedgehog_FOUND ON)

FIND_PATH(Hedgehog_INCLUDE_DIR hedgehog.h
		/usr/include/hedgehog
		/usr/local/include/hedgehog
		)

IF (NOT Hedgehog_INCLUDE_DIR)
	SET(Hedgehog_FOUND OFF)
	MESSAGE(STATUS "Could not find Hedgehog includes. Hedgehog_FOUND now off")
ELSE(NOT Hedgehog_INCLUDE_DIR)
	list(APPEND Hedgehog_INCLUDE_DIRS ${Hedgehog_INCLUDE_DIR})
ENDIF ()

IF (Hedgehog_FOUND)
	IF (NOT Hedgehog_FIND_QUIETLY)
		MESSAGE(STATUS "Found Hedgehog include: ${Hedgehog_INCLUDE_DIR}, CXX_FLAGS: ${Hedgehog_CXX_FLAGS}, Libs: ${Hedgehog_LIBRARIES}")
	ENDIF (NOT Hedgehog_FIND_QUIETLY)
ELSE (Hedgehog_FOUND)
	IF (Hedgehog_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could not find Hedgehog header files, please set the cmake variable Hedgehog_INCLUDE_DIR")
	ENDIF (Hedgehog_FIND_REQUIRED)
ENDIF (Hedgehog_FOUND)

MARK_AS_ADVANCED(Hedgehog_INCLUDE_DIR)

# Find other libraries
find_package(Threads QUIET)
if (Threads_FOUND)
	if (CMAKE_USE_PTHREADS_INIT)
		list(APPEND Hedgehog_CXX_FLAGS "-pthread")
	endif (CMAKE_USE_PTHREADS_INIT)
	list(APPEND Hedgehog_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
else ()
	if (Hedgehog_FIND_REQUIRED)
		message(FATAL_ERROR "Unable to find threads. Hedgehog must have a threading library i.e. pthreads.")
	else ()
		message(STATUS "Unable to find threads. Hedgehog must have a threading library i.e. pthreads.")
	endif ()
	SET(Hedgehog_FOUND OFF)
endif ()

#find_package(CUDAToolkit QUIET)
#if (CUDAToolkit_FOUND)
#	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#	set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
#	add_definitions(-DHH_USE_CUDA)
#	list(APPEND Hedgehog_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
#else ()
#	message(STATUS "Unable to find CUDA. All features won't be available.")
#endif ()

# Set Hedgehog_CXX_FLAGS
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
	set(CMAKE_COMPILER_IS_CLANGXX 1)
	set(use-libclang ON)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")

# Set specific flags for filesystem [experimental] library
if (CMAKE_COMPILER_IS_CLANGXX)
	string(REGEX REPLACE bin.* "" CLANG_FOLDER ${CMAKE_CXX_COMPILER})
	string(CONCAT FS_LIB_PATH ${CLANG_FOLDER} "lib/")
	link_directories(${FS_LIB_PATH})
	if (NOT APPLE)
		list(APPEND Hedgehog_LIBRARIES "stdc++fs")
	endif (NOT APPLE)
elseif (NOT MSVC)
	list(APPEND Hedgehog_LIBRARIES "stdc++fs")
endif (CMAKE_COMPILER_IS_CLANGXX)

