# 项目信息
PROJECT(DFLAC)

# 最低版本号要求
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# include路径
include_directories(DFLAS)

# 设置可执行文件生成路径
set(EXECUTABLE_OUTPUT_PATH ../build)

FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
message("OPENMP FOUND")  
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif() 

# 生成debug版本
SET(CMAKE_BUILD_TYPE "release")
if (CMAKE_BUILD_TYPE STREQUAL debug)
    add_definitions(-D_DEBUG)
endif ()
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++14")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++14")

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
# aux_source_directory(. DIR_SRCS)

# 指定生成目标的文件
 set(DIR_SRCS KGCompletion/main.cpp KGCompletion/trie_map.hpp KGCompletion/kgcrl.hpp KGCompletion/utilitys.hpp)


# 指定生成目标
add_executable(kgc KGCompletion/main.cpp)

# 添加子目录


# 指定目标编译语言
#SET_TARGET_PROPERTIES(dflas PROPERTIES LINKER_LANGUAGE C++)




# cmake .. -DENC=UTF-8
# IF (DEFINED ENC)
#     ADD_DEFINITIONS(-DDFLAC_${ENC})
# ENDIF()
