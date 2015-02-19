###############################################################################
#
# Build script for project
#
###############################################################################

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

# Add include/lib paths for local cmake builds
#include_directories(${CMAKE_SOURCE_DIR}/../../include)
#link_directories(${CMAKE_SOURCE_DIR}/../../lib/)

###############################################################################
#  Here list the project's code files that make up the model
###############################################################################

set(CCFILES )

set(CUFILES
	PredPrey.cu
  )

set(HFILES
	PredPrey.h)

###############################################################################
#  include folder locations for CUDA, CUDPP, and the project
###############################################################################

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	INCLUDE_DIRECTORIES(
		/Developer/NVIDIA/CUDA-6.0/samples/common/inc
		/Simulations/cudpp-2.1/include
		./Inc)
else()
	INCLUDE_DIRECTORIES(
		/home/johank/Simulations/NVIDIA_CUDA-6.0_Samples/common/inc
		/home/johank/Simulations/cudpp-2.1/include
		./Inc)
endif()

###############################################################################
#  Settings needed to run in Mac OS X, works only for Mavericks 
###############################################################################

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  message(STATUS "Detected Darwin kernel version: ${DARWIN_VERSION}")
  if ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
    set(CUDA_HOST_COMPILER /usr/bin/clang)
    set(APPLE_MAVERICKS_OR_LATER 1 INTERNAL)
    set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
    set(CMAKE_C_FLAGS -stdlib=libstdc++)
  endif ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
  # current CUDA on Mavericks uses libstdc++, not libc++
  # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# debug options
# set(CUDA_NVCC_FLAGS_DEBUG "-G;-g")
# set(CUDA_NVCC_FLAGS_RELEASE "-O2")

###############################################################################
#  Settings needed to run under Microsoft Visual C++ 
###############################################################################

if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

# set runtime library for Microsoft Visual C++
if (MSVC)
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} ";-Xcompiler;/MTd")
  set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} " /MTd")
  
  set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} ";-Xcompiler;/MT")
  set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} " /MT")
endif()

###############################################################################
#  Computer-specific settings with regard to compute capability, remove if new
###############################################################################

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
else()
	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
endif()

###############################################################################
#  Settings specific to the project
###############################################################################

cuda_add_executable(PredPrey ${CCFILES} ${CUFILES})

target_link_libraries(PredPrey cudpp ${CUDA_curand_LIBRARY})

# Generating the data folder where the model stores its results
file(MAKE_DIRECTORY ./build/Data)

# Defining the "run" command (use as: make run) that runs the model
add_custom_target(run ./PredPrey)

# Defining the “destroy: command (use as: make wipe) that clears the build folder
add_custom_target(destroy rm -rf ../build/*
                  COMMENT “Clears_the_build_directory”)
