# `.\yolov8\examples\YOLOv8-CPP-Inference\main.cpp`

```py
// 引入必要的头文件
#include <iostream>
#include <vector>
#include <getopt.h>

// 引入 OpenCV 库
#include <opencv2/opencv.hpp>

// 引入自定义的推理类头文件
#include "inference.h"

// 使用 std 命名空间，方便使用标准库函数
using namespace std;
using namespace cv;

// 主函数入口
int main(int argc, char **argv)
{
    // 设置 Ultralytics 项目的基础路径
    std::string projectBasePath = "/home/user/ultralytics"; // Set your ultralytics base path

    // 是否在 GPU 上运行推理
    bool runOnGPU = true;

    //
    // 选择要使用的 ONNX 模型文件:
    //
    // "yolov8s.onnx" 或 "yolov5s.onnx"
    //
    // 用于运行 yolov8/yolov5 的推理
    //

    // 注意：在此示例中类别信息是硬编码的，'classes.txt' 是一个占位符。
    // 创建推理对象，加载指定的 ONNX 模型文件、设置输入图像尺寸和类别文件名
    Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);

    // 设置要处理的图像文件名列表
    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    // 遍历图像列表
    for (int i = 0; i < imageNames.size(); ++i)
    {
        // 读取图像文件
        cv::Mat frame = cv::imread(imageNames[i]);

        // 推理开始...
        // 执行推理，获取检测结果
        std::vector<Detection> output = inf.runInference(frame);

        // 统计检测到的物体数量
        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        // 遍历每个检测结果
        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            // 获取检测框和颜色
            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // 在图像上绘制检测框
            cv::rectangle(frame, box, color, 2);

            // 绘制检测框上的文本
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // 推理结束...

        // 仅用于预览目的，缩放图像并显示
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1); // 等待用户按键退出
    }
}
```