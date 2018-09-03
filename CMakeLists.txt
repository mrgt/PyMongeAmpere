project(PyMongeAmpere)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/Boost.NumPy/libs/numpy/cmake ${CMAKE_MODULE_PATH})


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO") # RELEASE
ENDIF()

# find required python packages
find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# find boost
# 
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON) 
# set(Boost_USE_STATIC_RUNTIME ON)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

message( STATUS "found boost:"
  "\nINCLUDE: ${Boost_INCLUDE_DIRS}"
  "\nLIB: ${Boost_LIBRARIES}"
)

# enable C++11
ADD_DEFINITIONS(-std=c++11 -pg)

# monge-ampere
LIST(APPEND MONGE_AMPERE_CHECK_INCLUDE_DIRS
  $ENV{MA_DIR}
  ${PyMongeAmpere_SOURCE_DIR}/../MongeAmpere
  )

find_path(MONGE_AMPERE_DIR NAMES include/MA/quadrature.hpp
  PATHS ${MONGE_AMPERE_CHECK_INCLUDE_DIRS})
IF (EXISTS ${MONGE_AMPERE_DIR})
  MESSAGE(STATUS "Found MongeAmpere++ header in: ${MONGE_AMPERE_DIR}")
ELSE()
  MESSAGE(FATAL_ERROR "MongeAmpere++ not found (MA_DIR=${MA_DIR}).")
ENDIF()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MONGE_AMPERE_DIR}/cmake)
INCLUDE_DIRECTORIES(${MONGE_AMPERE_DIR}/include)

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# boost numpy
ADD_SUBDIRECTORY(thirdparty/Boost.NumPy)
INCLUDE_DIRECTORIES(${PyMongeAmpere_SOURCE_DIR}/thirdparty/Boost.NumPy)
LINK_LIBRARIES(boost_numpy)

# CGAL and its components
find_package( CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
LINK_LIBRARIES(${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES} )

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

FIND_PACKAGE ( X11 REQUIRED )
INCLUDE_DIRECTORIES ( ${X11_INCLUDE_DIR} )

# # SuiteSparse (optional)
# find_package(SuiteSparse)
# include_directories(${SUITESPARSE_INCLUDE_DIRS})
# link_libraries(${SUITESPARSE_LIBRARIES})

# if(SUITESPARSE_FOUND)
#   add_definitions(-DMA_USE_SUITESPARSE)
# endif()
# if(SUITESPARSEQR_FOUND)
#   add_definitions(-DMA_USE_SUITESPARSE_QR)
# endif()

# Intel TBB (optional)
# find_package(TBB)
# if(TBB_FOUND)
#   include_directories(${TBB_INCLUDE_DIRS})
#   link_libraries(${TBB_LIBRARIES})
# endif()

ADD_LIBRARY(MongeAmperePP MODULE MongeAmpere.cpp)
SET_TARGET_PROPERTIES(MongeAmperePP PROPERTIES PREFIX "")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(MongeAmpere)
