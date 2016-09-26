cmake_minimum_required(VERSION 2.8.7)

project(fastmf C CXX)

################################################################################################
# Command alias for debugging messages
# Usage:
#   dbprint(<message>)
function(dbmsg)
  message(STATUS ${ARGN})
endfunction()

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(ExternalProject)

set(fastmf_LINKER_LIBS_L "" )
set(fastmf_INCLUDE_DIR_L "" )

add_subdirectory("dmlc-core")
include_directories("dmlc-core/include")

# --[ MKL
include(${PROJECT_SOURCE_DIR}/cmake/FindMKL.cmake)

if(MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR})
endif()

# --[ TBB
include(${PROJECT_SOURCE_DIR}/cmake/FindTBB.cmake)

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIR})
  link_directories(${TBB_LIB_DIR})
endif()

# ---[ Swig
find_package(SWIG)

# ---[ Protobuf
include(${PROJECT_SOURCE_DIR}/cmake/ProtoBuf.cmake)

# ---[ Google Test
if(NOT GTEST_ROOT)
  if(EXISTS "${PROJECT_SOURCE_DIR}/googletest")
    add_subdirectory("${PROJECT_SOURCE_DIR}/googletest")
    set(GTEST_ROOT "${PROJECT_SOURCE_DIR}/googletest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  elseif(EXISTS "${PROJECT_SOURCE_DIR}/gtest")
    add_subdirectory("${PROJECT_SOURCE_DIR}/gtest")
    set(GTEST_ROOT "${PROJECT_SOURCE_DIR}/gtest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  else()
    enable_testing()
    find_package(GTest)
    dbmsg("GTEST_BOTH_LIBRARIES: ${GTEST_BOTH_LIBRARIES}")
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)

# generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/src")
file(GLOB_RECURSE proto_files "src/*.proto")
fastmf_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python "${PROJECT_SOURCE_DIR}" "src" ${proto_files})
include_directories(fastmf "${PROJECT_SOURCE_DIR}/include/")
include_directories(fastmf "${PROJECT_BINARY_DIR}/src/")
list(APPEND fastmf_INCLUDE_DIR_L "${PROJECT_BINARY_DIR}/include/")

if(MSVC)
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /bigobj")
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"   SUPPORT_CXX11)
  check_cxx_compiler_flag("-msse2"       SUPPORT_MSSE2)
  check_cxx_compiler_flag("-fopenmp"     SUPPORT_OPENMP)
  set(CMAKE_C_FLAGS "-Wall -Wno-unknown-pragmas -Wno-unused-result -fPIC")
  if(NDEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb3")
  endif()
  if(SUPPORT_MSSE2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2")
  endif()
  if(SUPPORT_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
  if(SUPPORT_OPENMP)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")
endif()

#FILE(COPY DIRECTORY "${PROJECT_BINARY_DIR}/include" DESTINATION "${PROJECT_SOURCE_DIR}/" FILES_MATCHING PATTERN "*.pb.h")
FILE(GLOB SOURCE "src/*.cc" "src/*.h")
FILE(GLOB TEST_SOURCE "tests/*.cc")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  list(APPEND fastmf_LINKER_LIBS_L "ipHlpApi.lib" "ws2_32.lib")
  foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endif()

dbmsg(${MKL_LIBRARIES})
set(fastmf_LINKER_LIBS ${fastmf_LINKER_LIBS} ${MKL_LIBRARIES})
set(fastmf_LINKER_LIBS ${fastmf_LINKER_LIBS} ${TBB_LIBRARIES})
set(fastmf_LINKER_LIBS ${fastmf_LINKER_LIBS} ${PROTOBUF_LIBRARIES})
set(fastmf_LINKER_LIBS ${fastmf_LINKER_LIBS} dmlccore)

list(APPEND SOURCE ${proto_srcs})
add_library(fastmf ${SOURCE})
target_link_libraries(fastmf ${fastmf_LINKER_LIBS})

add_executable(getdata data/getdata.cc)
target_link_libraries(getdata fastmf ${fastmf_LINKER_LIBS})

add_executable(mf tools/mf.cc)
target_link_libraries(mf fastmf ${fastmf_LINKER_LIBS})

add_executable(mf_unit_tests ${TEST_SOURCE})
target_link_libraries(mf_unit_tests
  fastmf
  ${fastmf_LINKER_LIBS}
  ${GTEST_BOTH_LIBRARIES}
  )
add_test(AllTestsInMfUnitTests mf_unit_tests)

list(APPEND fastmf_LINKER_LIBS_L ${PROTOBUF_LIBRARIES})
list(APPEND fastmf_INCLUDE_DIR_L "${PROJECT_SOURCE_DIR}/include")
list(APPEND fastmf_INCLUDE_DIR_L ${PROTOBUF_INCLUDE_DIR})

#set(fastmf_LINKER_LIBS ${fastmf_LINKER_LIBS_L} PARENT_SCOPE)
#set(fastmf_INCLUDE_DIR ${fastmf_INCLUDE_DIR_L} PARENT_SCOPE)
