#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------

# GTK
if(SIPL_USE_GTK)
    find_package (PkgConfig REQUIRED)
    pkg_check_modules (GTK2 REQUIRED gtk+-2.0 gthread-2.0)
endif()

#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories( ${SIPL_INCLUDE_DIRS} ${GTK2_INCLUDE_DIRS})
link_directories (${SIPL_LIBRARY_DIRS} ${GTK2_LIBRARY_DIRS})

