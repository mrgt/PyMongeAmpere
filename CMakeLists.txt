project(PyMongeAmpere)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO") # RELEASE
ENDIF()

# find required python packages
set (PYBIND11_PYTHON_VERSION 2.7)
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory("${PROJECT_SOURCE_DIR}/thirdparty/pybind11")

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
# TODO: better way of detecting MongeAmpere
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

pybind11_add_module(MongeAmperePP SHARED MongeAmpere.cpp)

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(MongeAmpere)
