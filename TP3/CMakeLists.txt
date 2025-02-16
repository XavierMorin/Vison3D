
cmake_minimum_required(VERSION 3.20)

project( simple )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( SGBM  sgbm.cpp)
target_link_libraries( SGBM  ${OpenCV_LIBS} )



