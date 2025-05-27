cmake_minimum_required(VERSION 3.17)
project(facesolution)

<<<<<<< HEAD
set(CMAKE_CXX_STANDARD 17)
=======
set(CMAKE_CXX_STANDARD 14)
>>>>>>> dcdbd17cb815f4e84cbf8404f64e88ff9844b640
set(ONNX "/opt/homebrew/cellar/onnxruntime/1.17.1/")
set(SOURCES
    main.cpp
    face_ali.cpp
    face_det.cpp
    face_rec.cpp
    face.cpp
    postprocessor.cpp
    preprocessor.cpp
    utility.cpp
)
<<<<<<< HEAD
find_package(OpenCV)
=======


find_package(OpenCV)

>>>>>>> dcdbd17cb815f4e84cbf8404f64e88ff9844b640
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(../include)
include_directories(${ONNX}/include/onnxruntime)     
link_directories(${ONNX}/lib)

# 指定为C++17 版本
set(CMAKE_CXX_STANDARD 17)

add_executable(facesolution ${SOURCES})

target_link_libraries(facesolution ${OpenCV_LIBS}) 
target_link_libraries(facesolution ${ONNX}/lib/libonnxruntime.1.17.1.dylib) 