# retainface-arcface
## path
```
.
├── build
├── img
├── model
└── src
```
## quick start
```bash
cmake -S src -B build 
cmake --build build
build/facesolution 10 ./video/video.mp4  #根目录运行
``` 
## model
使用了arcface用于人脸识别,retainface用于人脸检测