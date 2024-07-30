# `.\yolov8\examples\YOLOv8-CPP-Inference\inference.h`

```py
#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>          // 文件流操作
#include <vector>           // 向量容器
#include <string>           // 字符串操作
#include <random>           // 随机数生成

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>   // 图像处理功能
#include <opencv2/opencv.hpp>    // OpenCV 核心功能
#include <opencv2/dnn.hpp>       // OpenCV 深度神经网络模块

// 结构体用于表示检测结果
struct Detection
{
    int class_id{0};            // 类别 ID
    std::string className{};    // 类别名称
    float confidence{0.0};      // 置信度
    cv::Scalar color{};         // 框的颜色
    cv::Rect box{};             // 框的位置信息
};

// 推理类声明
class Inference
{
public:
    // 构造函数，初始化模型路径、输入形状、类别文件路径及是否使用 CUDA
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
    
    // 执行推理的方法，输入图像并返回检测结果
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    // 从文件加载类别信息
    void loadClassesFromFile();
    
    // 加载 ONNX 模型
    void loadOnnxNetwork();
    
    // 将图像格式化为正方形
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};            // 模型文件路径
    std::string classesPath{};          // 类别文件路径
    bool cudaEnabled{};                 // 是否启用 CUDA 加速

    std::vector<std::string> classes{   // 默认类别列表
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Size2f modelShape{};            // 模型输入形状

    float modelConfidenceThreshold {0.25};   // 置信度阈值
    float modelScoreThreshold      {0.45};   // 分数阈值
    float modelNMSThreshold        {0.50};   // 非最大抑制阈值

    bool letterBoxForSquare = true;     // 是否使用 letterbox 方法来调整图像为正方形

    cv::dnn::Net net;                   // OpenCV DNN 网络对象
};

#endif // INFERENCE_H
```