#cmake file for project lol                     #表示注释  
#author:>---myg 
#created:>--2018/07/18 
 
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)    #cmake 最低版本要求，低于2.6 构建过程会被终止。  
 
PROJECT(lol_filter)                          #定义工程名称 


SET(CMAKE_BUILD_TYPE "Release")                      #指定编译类型
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wall")          #指定编译器 
add_definitions(-msse4.1  -std=c++11)


file( GLOB LIB_SOURCES_GEMM ${PROJECT_SOURCE_DIR}/src/*.cc )

file( GLOB LIB_HEADER_GEMM ${PROJECT_SOURCE_DIR}/src/*.h )

set(DEMO_FILE 
    ${PROJECT_SOURCE_DIR}/demo.cc 
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)



add_executable(gemm_demo ${DEMO_FILE} ${LIB_SOURCES_GEMM} ${LIB_HEADER_GEMM})
target_link_libraries(gemm_demo   -lpthread -lm -lstdc++)