# `.\PaddleOCR\deploy\cpp_infer\include\ocr_cls.h`

```
// 版权声明，告知代码版权信息
// 根据 Apache 许可证 2.0 版本规定使用此文件
// 获取许可证的链接
// 根据适用法律或书面协议，分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
// 查看许可证以了解特定语言的权限和限制
#pragma once
// 包含 PaddlePaddle 的 API 头文件
#include "paddle_api.h"
// 包含 Paddle 推理 API 头文件
#include "paddle_inference_api.h"
// 包含预处理操作的头文件
#include <include/preprocess_op.h>
// 包含实用工具的头文件
#include <include/utility.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 定义分类器类
class Classifier {
public:
  // 构造函数，初始化分类器对象
  explicit Classifier(const std::string &model_dir, const bool &use_gpu,
                      const int &gpu_id, const int &gpu_mem,
                      const int &cpu_math_library_num_threads,
                      const bool &use_mkldnn, const double &cls_thresh,
                      const bool &use_tensorrt, const std::string &precision,
                      const int &cls_batch_num) {
    // 初始化 GPU 使用标志
    this->use_gpu_ = use_gpu;
    // 初始化 GPU ID
    this->gpu_id_ = gpu_id;
    // 初始化 GPU 内存
    this->gpu_mem_ = gpu_mem;
    // 初始化 CPU 数学库线程数
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    // 初始化是否使用 MKL-DNN
    this->use_mkldnn_ = use_mkldnn;

    // 初始化分类阈值
    this->cls_thresh = cls_thresh;
    // 初始化是否使用 TensorRT
    this->use_tensorrt_ = use_tensorrt;
    // 初始化精度
    this->precision_ = precision;
    // 初始化分类批次数
    this->cls_batch_num_ = cls_batch_num;

    // 加载模型
    LoadModel(model_dir);
  }
  // 默认分类阈值
  double cls_thresh = 0.9;

  // 加载 Paddle 推理模型
  void LoadModel(const std::string &model_dir);

  // 运行分类器
  void Run(std::vector<cv::Mat> img_list, std::vector<int> &cls_labels,
           std::vector<float> &cls_scores, std::vector<double> &times);
// 私有成员变量，用于存储预测器指针
private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  // 是否使用 GPU，默认为 false
  bool use_gpu_ = false;
  // GPU 设备 ID，默认为 0
  int gpu_id_ = 0;
  // GPU 内存大小，默认为 4000
  int gpu_mem_ = 4000;
  // CPU 数学库线程数，默认为 4
  int cpu_math_library_num_threads_ = 4;
  // 是否使用 MKL-DNN，默认为 false
  bool use_mkldnn_ = false;

  // 图像预处理参数：均值
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  // 图像预处理参数：缩放
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  // 是否进行缩放，默认为 true
  bool is_scale_ = true;
  // 是否使用 TensorRT，默认为 false
  bool use_tensorrt_ = false;
  // 精度设置，默认为 "fp32"
  std::string precision_ = "fp32";
  // 分类批次数，默认为 1
  int cls_batch_num_ = 1;
  
  // 预处理操作：调整图像大小
  ClsResizeImg resize_op_;
  // 预处理操作：归一化
  Normalize normalize_op_;
  // 预处理操作：批次置换
  PermuteBatch permute_op_;

}; // 类 Classifier

} // 命名空间 PaddleOCR
```