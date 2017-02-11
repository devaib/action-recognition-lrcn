CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(PROJECT_NAME "optical_flow")
PROJECT("${PROJECT_NAME}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenCV_DIR "/home/binghao/software/opencv/build")

# add opencv package to the project
FIND_PACKAGE( OpenCV REQUIRED )

SET(OpenCV_INCLUDE_DIRS "/home/binghao/software/opencv/include;/home/binghao/software/opencv/include/opencv" )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

MESSAGE(STATUS "Project Name: ${PROJECT_NAME}")
MESSAGE(STATUS "    OpenCV version: ${OpenCV_VERSION} ")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

FILE(GLOB files
  "*.h"
  "*.hpp"
  "*.cpp"
)
FILE(GLOB hidden
	".*"
)
FILE(GLOB resources
  "*.*"
)

ADD_EXECUTABLE(${PROJECT_NAME} ${files} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${OpenCV_LIBS} )

#LIST(REMOVE_ITEM resources ${files} ${hidden} "${CMAKE_SOURCE_DIR}/CMakeLists.txt")
#FILE(COPY ${resources} DESTINATION "Debug")
#FILE(COPY ${resources} DESTINATION "Release")
