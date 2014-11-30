project(PyMongeAmpere)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO") # RELEASE
ENDIF()

# Boost + Boost::Python
FIND_PACKAGE(Boost 1.54.0 COMPONENTS python REQUIRED)
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
link_libraries(${Boost_LIBRARIES})
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")

# enable C++11
ADD_DEFINITIONS(-std=c++11 -pg)

# monge-ampere
find_path(MONGE_AMPERE_DIR include/MA/voronoi_triangulation_intersection.hpp
          HINTS ${PyMongeAmpere_SOURCE_DIR}/../MongeAmpere)
message(${MONGE_AMPERE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MONGE_AMPERE_DIR}/cmake)
INCLUDE_DIRECTORIES(${MONGE_AMPERE_DIR}/include)

# python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES})

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# boost numpy
ADD_SUBDIRECTORY(thirdparty/Boost.NumPy)
INCLUDE_DIRECTORIES(${PyMongeAmpere_SOURCE_DIR}/thirdparty/Boost.NumPy)
LINK_LIBRARIES(boost_numpy)

# CGAL and its components
find_package( CGAL QUIET COMPONENTS)
include( ${CGAL_USE_FILE} )
LINK_LIBRARIES(${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES} )

# SuiteSparse (optional)
find_package(SuiteSparse)
include_directories(${SUITESPARSE_INCLUDE_DIRS})
link_libraries(${SUITESPARSE_LIBRARIES})
if(SUITESPARSEQR_FOUND)
  add_definitions(-DMA_USE_SUITESPARSE_QR)
endif()

# Intel TBB (optional)
find_package(TBB)
if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  link_libraries(${TBB_LIBRARIES})
endif()

ADD_LIBRARY(MongeAmperePP MODULE MongeAmpere.cpp)
SET_TARGET_PROPERTIES(MongeAmperePP PROPERTIES PREFIX "")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(MongeAmpere)
