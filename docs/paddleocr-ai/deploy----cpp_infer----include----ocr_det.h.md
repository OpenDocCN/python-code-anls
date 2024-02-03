# `.\PaddleOCR\deploy\cpp_infer\include\ocr_det.h`

```py
// 版权声明，告知代码版权归属
// 根据 Apache License, Version 2.0 许可证使用代码
// 获取许可证的链接
// 根据适用法律或书面同意，分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
// 查看许可证以了解特定语言的权限和限制

// 包含必要的头文件
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <include/postprocess_op.h>
#include <include/preprocess_op.h>

// 命名空间定义
namespace PaddleOCR {

// DBDetector 类的构造函数，初始化各个参数
class DBDetector {
public:
  explicit DBDetector(const std::string &model_dir, const bool &use_gpu,
                      const int &gpu_id, const int &gpu_mem,
                      const int &cpu_math_library_num_threads,
                      const bool &use_mkldnn, const std::string &limit_type,
                      const int &limit_side_len, const double &det_db_thresh,
                      const double &det_db_box_thresh,
                      const double &det_db_unclip_ratio,
                      const std::string &det_db_score_mode,
                      const bool &use_dilation, const bool &use_tensorrt,
                      const std::string &precision) {
    // 初始化 GPU 相关参数
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    this->use_mkldnn_ = use_mkldnn;

    // 初始化限制类型和边长参数
    this->limit_type_ = limit_type;
    this->limit_side_len_ = limit_side_len;

    // 初始化检测参数
    this->det_db_thresh_ = det_db_thresh;
    this->det_db_box_thresh_ = det_db_box_thresh;
    this->det_db_unclip_ratio_ = det_db_unclip_ratio;
    this->det_db_score_mode_ = det_db_score_mode;
    this->use_dilation_ = use_dilation;
    // 设置是否使用 TensorRT
    this->use_tensorrt_ = use_tensorrt;
    // 设置模型精度
    this->precision_ = precision;

    // 载入 Paddle 推理模型
    LoadModel(model_dir);
  }

  // 载入 Paddle 推理模型
  void LoadModel(const std::string &model_dir);

  // 运行预测器
  void Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes,
           std::vector<double> &times);
// 定义私有成员变量，用于存储 PaddlePaddle 预测器
private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  // 是否使用 GPU，默认为 false
  bool use_gpu_ = false;
  // GPU 设备 ID，默认为 0
  int gpu_id_ = 0;
  // GPU 内存大小，默认为 4000 MB
  int gpu_mem_ = 4000;
  // CPU 数学库线程数，默认为 4
  int cpu_math_library_num_threads_ = 4;
  // 是否使用 MKL-DNN，默认为 false
  bool use_mkldnn_ = false;

  // 限制类型，默认为 "max"
  std::string limit_type_ = "max";
  // 限制边长，默认为 960
  int limit_side_len_ = 960;

  // 目标检测阈值，默认为 0.3
  double det_db_thresh_ = 0.3;
  // 目标检测框阈值，默认为 0.5
  double det_db_box_thresh_ = 0.5;
  // 目标检测框解压比例，默认为 2.0
  double det_db_unclip_ratio_ = 2.0;
  // 目标检测得分模式，默认为 "slow"
  std::string det_db_score_mode_ = "slow";
  // 是否使用膨胀，默认为 false
  bool use_dilation_ = false;

  // 是否可视化结果，默认为 true
  bool visualize_ = true;
  // 是否使用 TensorRT，默认为 false
  bool use_tensorrt_ = false;
  // 精度类型，默认为 "fp32"
  std::string precision_ = "fp32";

  // 均值数组，默认为 {0.485, 0.456, 0.406}
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  // 缩放数组，默认为 {1/0.229, 1/0.224, 1/0.225}
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  // 是否进行缩放，默认为 true
  bool is_scale_ = true;

  // 预处理操作对象
  // 图像大小调整操作
  ResizeImgType0 resize_op_;
  // 归一化操作
  Normalize normalize_op_;
  // 排列操作
  Permute permute_op_;

  // 后处理操作对象
  // DB 后处理器
  DBPostProcessor post_processor_;
};

// 命名空间结束标记
} // namespace PaddleOCR
```