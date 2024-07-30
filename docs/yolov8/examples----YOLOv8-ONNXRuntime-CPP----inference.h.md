# `.\yolov8\examples\YOLOv8-ONNXRuntime-CPP\inference.h`

```py
#pragma once
#pragma once 指令：确保头文件只被编译一次，避免重复定义

#define    RET_OK nullptr
RET_OK 宏定义：表示一个空指针常量，用于表示操作成功时的返回值

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif
#ifdef _WIN32 条件编译：仅在 Windows 平台下包含特定的系统头文件

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif
普通头文件包含：包含 C++ 标准库、OpenCV、ONNX Runtime 的 C++ API 头文件，并根据 USE_CUDA 宏条件包含 CUDA FP16 头文件

enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};
MODEL_TYPE 枚举类型：定义了多个模型类型，包括 FLOAT32 和 FLOAT16 的 YOLO 模型

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int    keyPointsNum = 2;//Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;
_DL_INIT_PARAM 结构体：定义了初始化深度学习模型所需的各种参数，包括模型路径、类型、图像尺寸、阈值等

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;
_DL_RESULT 结构体：定义了深度学习模型运行后的输出结果，包括类别 ID、置信度、边界框、关键点等信息

class YOLO_V8
{
public:
    YOLO_V8();
    构造函数：用于初始化 YOLO_V8 类的实例

    ~YOLO_V8();
    析构函数：用于释放 YOLO_V8 类的实例

public:
    char* CreateSession(DL_INIT_PARAM& iParams);
    方法声明：创建深度学习模型会话

    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    方法声明：运行深度学习模型会话并返回结果

    char* WarmUpSession();
    方法声明：预热深度学习模型会话

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);
    模板方法声明：对输入图像进行张量处理并返回处理结果

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    方法声明：预处理输入图像，将其调整为指定尺寸

    std::vector<std::string> classes{};
    类成员变量：用于存储模型输出类别的名称

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    类私有成员变量：存储 ONNX Runtime 执行环境、会话、CUDA 加速状态、运行选项、输入节点名称、输出节点名称等信息

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
    类私有成员变量：存储模型类型、图像尺寸、置信度阈值、IoU 阈值、调整比例等信息
};
```