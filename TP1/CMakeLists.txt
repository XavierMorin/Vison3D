
cmake_minimum_required(VERSION 3.20)

project( simple )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( homographie homographie.cpp)
target_link_libraries( homographie ${OpenCV_LIBS} )



