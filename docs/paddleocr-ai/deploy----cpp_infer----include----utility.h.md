# `.\PaddleOCR\deploy\cpp_infer\include\utility.h`

```
// 版权声明，告知代码版权归属及使用许可
// 使用 Apache 许可证 2.0 版本，详细内容可在指定网址查看
// 根据适用法律或书面同意，软件按"原样"分发，不提供任何担保或条件
// 查看许可证以了解特定语言的权限和限制
#pragma once
// 包含必要的头文件
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// 命名空间 PaddleOCR，包含 OCR 相关的结构和类
namespace PaddleOCR {

// OCR 预测结果结构体
struct OCRPredictResult {
  std::vector<std::vector<int>> box; // 文本框坐标
  std::string text; // 文本内容
  float score = -1.0; // 文本置信度
  float cls_score; // 类别置信度
  int cls_label = -1; // 类别标签
};

// 结构化预测结果结构体
struct StructurePredictResult {
  std::vector<float> box; // 结构框坐标
  std::vector<std::vector<int>> cell_box; // 单元格框坐标
  std::string type; // 结构类型
  std::vector<OCRPredictResult> text_res; // 文本识别结果
  std::string html; // HTML 内容
  float html_score = -1; // HTML 置信度
  float confidence; // 结果置信度
};

// 实用工具类
class Utility {
public:
  // 读取字典文件
  static std::vector<std::string> ReadDict(const std::string &path);

  // 可视化文本框
  static void VisualizeBboxes(const cv::Mat &srcimg,
                              const std::vector<OCRPredictResult> &ocr_result,
                              const std::string &save_path);

  // 可视化结构框
  static void VisualizeBboxes(const cv::Mat &srcimg,
                              const StructurePredictResult &structure_result,
                              const std::string &save_path);

  // 模板函数，返回迭代器范围内的最大值索引
  template <class ForwardIterator>
  inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
  // 返回[first, last)范围内最大元素与first之间的距离
  return std::distance(first, std::max_element(first, last));
}

// 获取指定目录下的所有文件名
static void GetAllFiles(const char *dir_name,
                        std::vector<std::string> &all_inputs);

// 获取旋转裁剪后的图像
static cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                                  std::vector<std::vector<int>> box);

// 对数组进行排序并返回排序后的索引
static std::vector<int> argsort(const std::vector<float> &array);

// 返回路径中的基本文件名部分
static std::string basename(const std::string &filename);

// 检查路径是否存在
static bool PathExists(const std::string &path);

// 创建目录
static void CreateDir(const std::string &path);

// 打印OCR预测结果
static void print_result(const std::vector<OCRPredictResult> &ocr_result);

// 裁剪图像
static cv::Mat crop_image(cv::Mat &img, const std::vector<int> &area);
static cv::Mat crop_image(cv::Mat &img, const std::vector<float> &area);

// 对OCR预测结果进行排序
static void sorted_boxes(std::vector<OCRPredictResult> &ocr_result);

// 将8个坐标值转换为4个坐标值
static std::vector<int> xyxyxyxy2xyxy(std::vector<std::vector<int>> &box);
static std::vector<int> xyxyxyxy2xyxy(std::vector<int> &box);

// 快速计算指数函数
static float fast_exp(float x);

// softmax激活函数
static std::vector<float> activation_function_softmax(std::vector<float> &src);

// 计算两个矩形框的IoU（交并比）
static float iou(std::vector<int> &box1, std::vector<int> &box2);
static float iou(std::vector<float> &box1, std::vector<float> &box2);
// 定义一个私有静态方法，用于比较两个 OCR 预测结果的框的位置
private:
  static bool comparison_box(const OCRPredictResult &result1,
                             const OCRPredictResult &result2) {
    // 如果第一个结果框的顶点 Y 坐标小于第二个结果框的顶点 Y 坐标，则返回 true
    if (result1.box[0][1] < result2.box[0][1]) {
      return true;
    } 
    // 如果两个结果框的顶点 Y 坐标相等，则比较顶点 X 坐标
    else if (result1.box[0][1] == result2.box[0][1]) {
      return result1.box[0][0] < result2.box[0][0];
    } 
    // 其他情况返回 false
    else {
      return false;
    }
  }
};

} // namespace PaddleOCR
```