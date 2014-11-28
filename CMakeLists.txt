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
set(MONGE_AMPERE_DIR ${PyMongeAmpere_SOURCE_DIR}/../voronoi_intersection)
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

function (create_symlinks)
    # Do nothing if building in-source
    if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()

    foreach (path_file ${ARGN})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${folder}")

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${path_file}" link)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${path_file}" target)

        if (UNIX)
            set(command ln -s ${target} ${link})
        else()
            set(command cmd.exe /c mklink ${link} ${target})
        endif()

        execute_process(COMMAND ${command} 
                        RESULT_VARIABLE result
                        ERROR_VARIABLE output)

        if (NOT ${result} EQUAL 0)
            message(FATAL_ERROR "Could not create symbolic link for: ${target} --> ${output}")
        endif()

    endforeach(path_file)
endfunction(create_symlinks)

set(python_files test.py)
create_symlinks(${python_files})

ADD_LIBRARY(MongeAmpere MODULE MongeAmpere.cpp)
SET_TARGET_PROPERTIES(MongeAmpere PROPERTIES PREFIX "")
