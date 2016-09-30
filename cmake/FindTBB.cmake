# Find the TBB libraries
#
# Options:
#
#   TBB_USE_SINGLE_DYNAMIC_LIBRARY  : use single dynamic library interface
#   TBB_USE_STATIC_LIBS             : use static libraries
#
# This module defines the following variables:
#
#   TBB_FOUND            : True tbb is found
#   TBB_INCLUDE_DIR      : unclude directory
#   TBB_LIB_DIR          : lib directory
#   TBB_LIBRARIES        : the libraries to link against.


# ---[ Root folders
set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")
find_path(TBB_ROOT include/tbb/tbb.h PATHS $ENV{TBB_ROOT} ${INTEL_ROOT}/tbb /usr
  DOC "Folder contains TBB")


#message(STATUS "TBB_ROOT: " "${TBB_ROOT}")

# ---[ Find include dir
find_path(TBB_INCLUDE_DIR tbb/tbb.h PATHS ${TBB_ROOT} PATH_SUFFIXES include)
set(__looked_for TBB_INCLUDE_DIR)

# ---[ Find lib dir
find_path(TBB_LIB_DIR libtbb.so PATHS ${TBB_ROOT} PATH_SUFFIXES lib/intel64/gcc4.4/ lib/)
set(__looked_for TBB_LIB_DIR)

#message(STATUS "TBB_LIB_DIR: " "${TBB_LIB_DIR}")

# ---[ Find libraries
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(__path_suffixes lib lib/ia32/gcc4.4)
else()
  set(__path_suffixes lib lib/intel64/gcc4.4)
endif()

set(__tbb_libs "")

set(TBB_TBB_LIBRARY       tbb)

if(EXISTS ${TBB_LIB_DIR}/libtbb_debug.so)
  set(TBB_TBB_DEBUG_LIBRARY tbb_debug)
else()
  set(TBB_TBB_DEBUG_LIBRARY tbb)
endif()

set(TBB_TBBMALLOC_LIBRARY       tbbmalloc)

if(EXISTS ${TBB_LIB_DIR}/libtbbmalloc_debug.so)
  set(TBB_TBBMALLOC_DEBUG_LIBRARY tbbmalloc_debug)
else()
  set(TBB_TBBMALLOC_DEBUG_LIBRARY tbbmalloc_debug)
endif()

if(NDEBUG)
  list(APPEND __tbb_libs ${TBB_TBB_LIBRARY} ${TBB_TBBMALLOC_LIBRARY})
else()
  list(APPEND __tbb_libs ${TBB_TBB_DEBUG_LIBRARY} ${TBB_TBBMALLOC_DEBUG_LIBRARY})
endif()

foreach (__lib ${__tbb_libs})
  set(__tbb_lib "tbb_${__lib}")
  string(TOUPPER ${__tbb_lib} __tbb_lib_upper)

  find_library(${__tbb_lib_upper}_LIBRARY
    NAMES ${__tbb_lib}
    PATHS ${TBB_ROOT} "${TBB_INCLUDE_DIR}/.."
    PATH_SUFFIXES ${__path_suffixes}
    DOC "The path to Intel(R) TBB ${__tbb_lib} library")
  mark_as_advanced(${__tbb_lib_upper}_LIBRARY)

  list(APPEND __looked_for ${__tbb_lib_upper}_LIBRARY)
  list(APPEND TBB_LIBRARIES ${${__tbb_lib_upper}_LIBRARY})
endforeach()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG ${__looked_for})

if(TBB_FOUND)
  message(STATUS "Found TBB (include: ${TBB_INCLUDE_DIR}, lib: ${TBB_LIBRARIES}, libdir: ${TBB_LIB_DIR}")
endif()


