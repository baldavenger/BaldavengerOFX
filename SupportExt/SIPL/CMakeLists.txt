cmake_minimum_required(VERSION 2.8)
project(SIPL)


find_package(OpenMP)
if(OPENMP_FOUND)
    message("-- OpenMP was detected. Using OpenMP to speed up MIP calculations.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()



add_library (SIPL Core.cpp Types.cpp Visualization.cpp)
option(sipl_use_gtk "Use GTK for visualization and loading images in SIPL" ON)
if(sipl_use_gtk)
	message("-- Compiling SIPL with GTK")
	find_package (PkgConfig REQUIRED)
	pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)
	include_directories (${GTK2_INCLUDE_DIRS})
	set(SIPL_INCLUDE_DIRS ${GTK2_INCLUDE_DIRS} PARENT_SCOPE)
	link_directories (${GTK2_LIBRARY_DIRS})
	add_definitions (${GTK2_CFLAGS_OTHER})
	target_link_libraries (SIPL ${GTK2_LIBRARIES})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D USE_GTK")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D USE_GTK")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D USE_GTK" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D USE_GTK" PARENT_SCOPE)
endif()


option(compile_sipl_tests "Compile tests for SIPL" OFF) # cmake option to turn on compilation of tests
find_package(GTest)
if(GTEST_FOUND)
if(compile_sipl_tests)
	message("-- Google test framework found. Compiling tests ...")
	add_subdirectory(tests)
endif()
endif()

#------------------------------------------------------------------------------
# Configure file for find_package module 
#------------------------------------------------------------------------------
set( SIPL_INCLUDE_DIRS
    ${SIPL_SOURCE_DIR}
    ) 

set(SIPL_LIBRARY_DIRS
    ${SIPL_BINARY_DIR}
    )    

configure_file (
    "${PROJECT_SOURCE_DIR}/CMake/SIPLConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/SIPLConfig.cmake"
    )

