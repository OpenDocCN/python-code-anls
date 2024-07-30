# `.\yolov8\examples\YOLOv8-OpenVINO-CPP-Inference\inference.h`

```py
#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>                               // 包含字符串处理的标准库
#include <vector>                               // 包含向量处理的标准库
#include <opencv2/imgproc.hpp>                  // 包含 OpenCV 图像处理模块
#include <openvino/openvino.hpp>                // 包含 OpenVINO 框架

namespace yolo {

struct Detection {
    short class_id;                            // 检测到的对象类别ID
    float confidence;                          // 检测到的对象置信度
    cv::Rect box;                              // 检测到的对象边界框
};

class Inference {
 public:
    Inference() {}                             // 默认构造函数
    // 使用默认输入形状初始化模型的构造函数
    Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);
    // 使用指定输入形状初始化模型的构造函数
    Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

    void RunInference(cv::Mat &frame);          // 执行推理过程的方法

 private:
    void InitializeModel(const std::string &model_path);  // 初始化模型的私有方法
    void Preprocessing(const cv::Mat &frame);             // 图像预处理方法
    void PostProcessing(cv::Mat &frame);                  // 后处理方法
    cv::Rect GetBoundingBox(const cv::Rect &src) const;   // 获取边界框方法
    void DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;  // 绘制检测到的对象方法

    cv::Point2f scale_factor_;                // 输入帧的缩放因子
    cv::Size2f model_input_shape_;            // 模型的输入形状
    cv::Size model_output_shape_;             // 模型的输出形状

    ov::InferRequest inference_request_;      // OpenVINO 推理请求
    ov::CompiledModel compiled_model_;        // OpenVINO 编译后的模型

    float model_confidence_threshold_;        // 检测置信度阈值
    float model_NMS_threshold_;               // 非极大值抑制阈值

    std::vector<std::string> classes_ {       // 检测模型所涉及的对象类别列表
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_
```