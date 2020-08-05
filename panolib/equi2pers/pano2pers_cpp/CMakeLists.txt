set(PROJECT_NAME pano2perspective)
project(${PROJECT_NAME})

#**************************************************************************************************
# General CMake settings
#**************************************************************************************************

cmake_minimum_required(VERSION 3.5)

#**************************************************************************************************
# Make Configuration
#**************************************************************************************************

# Find dependencies
find_package(CUDA QUIET REQUIRED)
# find_package(Eigen3 REQUIRED)
# work around...
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
find_package(OpenCV REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(DEPENDENCIES ${OpenCV_LIBS} ${EIGEN3_LIBS})

set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

# CUDA
cuda_add_library(cuda_func SHARED src/cuda_func.cu OPTIONS -std=c++11)
set_target_properties(cuda_func PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Linking
add_subdirectory("./pybind11" "./pybind11/build")
pybind11_add_module(${PROJECT_NAME} src/pano2perspective.hh src/pano2perspective.cc src/wrap.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES} cuda_func ) #-fsanitize=address -static-libasan)
