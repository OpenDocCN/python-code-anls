# `.\PaddleOCR\deploy\cpp_infer\include\structure_layout.h`

```py
// 版权声明，告知代码版权归属及使用许可
// 根据 Apache 许可证 2.0 版本使用本文件
// 获取许可证的链接
// 根据适用法律或书面协议，分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制
#pragma once

// 引入 PaddlePaddle 的 API 头文件
#include "paddle_api.h"
#include "paddle_inference_api.h"

// 引入自定义的后处理操作头文件
#include <include/postprocess_op.h>
// 引入自定义的预处理操作头文件
#include <include/preprocess_op.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 结构化布局识别器类
class StructureLayoutRecognizer {
public:
  // 构造函数，初始化结构化布局识别器
  explicit StructureLayoutRecognizer(
      const std::string &model_dir, const bool &use_gpu, const int &gpu_id,
      const int &gpu_mem, const int &cpu_math_library_num_threads,
      const bool &use_mkldnn, const std::string &label_path,
      const bool &use_tensorrt, const std::string &precision,
      const double &layout_score_threshold,
      const double &layout_nms_threshold) {
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
    // 初始化是否使用 TensorRT
    this->use_tensorrt_ = use_tensorrt;
    // 初始化精度
    this->precision_ = precision;

    // 初始化后处理器
    this->post_processor_.init(label_path, layout_score_threshold,
                               layout_nms_threshold);
    // 加载模型
    LoadModel(model_dir);
  }

  // 加载 Paddle 推理模型
  void LoadModel(const std::string &model_dir);

  // 运行结构化布局识别器
  void Run(cv::Mat img, std::vector<StructurePredictResult> &result,
           std::vector<double> &times);
// 私有成员变量，用于存储 PaddlePaddle 预测器
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

  // 图像预处理参数
  // 均值，默认为 {0.485, 0.456, 0.406}
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  // 缩放比例，默认为 {1/0.229, 1/0.224, 1/0.225}
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  // 是否进行缩放，默认为 true
  bool is_scale_ = true;

  // 是否使用 TensorRT，默认为 false
  bool use_tensorrt_ = false;
  // 精度设置，默认为 "fp32"
  std::string precision_ = "fp32";

  // 预处理操作
  // 图像缩放操作
  Resize resize_op_;
  // 图像归一化操作
  Normalize normalize_op_;
  // 图像维度变换操作
  Permute permute_op_;

  // 后处理操作
  // Picodet 后处理器
  PicodetPostProcessor post_processor_;
};

// 命名空间结束
} // namespace PaddleOCR
```