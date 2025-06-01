# retainface-arcface
## path
```
.
├── build
├── model
│   ├── MFN.onnx
│   └── mobilenet0.25_Final.onnx
└── src
    ├── face_ali.cpp
    ├── face_ali.h
    ├── face_det.cpp
    ├── face_det.h
    ├── face_rec.cpp
    ├── face_rec.h
    ├── face.cpp
    ├── face.h
    ├── main.cpp
    ├── postprocessor.cpp
    ├── postprocessor.h
    ├── preprocessor.cpp
    ├── preprocessor.h
    ├── processor.h
    ├── ThreadPool.h
    ├── utility.cpp
    └── utility.h
```
## quick start
```bash
cmake -S src -B build 
cmake --build build
build/facesolution 10 ./video/video.mp4  #根目录运行
``` 
## model
使用了arcface用于人脸识别,retainface用于人脸检测