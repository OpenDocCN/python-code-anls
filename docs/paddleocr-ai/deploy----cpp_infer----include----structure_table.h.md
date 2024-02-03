# `.\PaddleOCR\deploy\cpp_infer\include\structure_table.h`

```py
// 版权声明，告知代码版权归属
// 根据 Apache 许可证 2.0 版本规定使用此文件
// 获取许可证的链接
// 根据适用法律或书面同意，分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
// 查看许可证以了解特定语言的权限和限制
#pragma once

// 引入 PaddlePaddle 的 API 头文件
#include "paddle_api.h"
// 引入 Paddle 推理 API 头文件
#include "paddle_inference_api.h"

// 引入后处理操作的头文件
#include <include/postprocess_op.h>
// 引入预处理操作的头文件
#include <include/preprocess_op.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 定义 StructureTableRecognizer 类
class StructureTableRecognizer {
public:
  // 构造函数，初始化模型目录、是否使用 GPU、GPU ID、GPU 内存、CPU 线程数、是否使用 MKL-DNN、标签路径、是否使用 TensorRT、精度、表格批次数、表格最大长度、是否合并无跨度结构
  explicit StructureTableRecognizer(
      const std::string &model_dir, const bool &use_gpu, const int &gpu_id,
      const int &gpu_mem, const int &cpu_math_library_num_threads,
      const bool &use_mkldnn, const std::string &label_path,
      const bool &use_tensorrt, const std::string &precision,
      const int &table_batch_num, const int &table_max_len,
      const bool &merge_no_span_structure) {
    // 设置是否使用 GPU
    this->use_gpu_ = use_gpu;
    // 设置 GPU ID
    this->gpu_id_ = gpu_id;
    // 设置 GPU 内存
    this->gpu_mem_ = gpu_mem;
    // 设置 CPU 线程数
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    // 设置是否使用 MKL-DNN
    this->use_mkldnn_ = use_mkldnn;
    // 设置是否使用 TensorRT
    this->use_tensorrt_ = use_tensorrt;
    // 设置精度
    this->precision_ = precision;
    // 设置表格批次数
    this->table_batch_num_ = table_batch_num;
    // 设置表格最大长度
    this->table_max_len_ = table_max_len;

    // 初始化后处理器，传入标签路径和是否合并无跨度结构
    this->post_processor_.init(label_path, merge_no_span_structure);
    // 调用 LoadModel 函数，加载 Paddle 推理模型
    LoadModel(model_dir);
    
    // 加载 Paddle 推理模型
    void LoadModel(const std::string &model_dir);
    
    // 运行推理模型，传入图像列表，返回识别的 HTML 标签、得分、边界框和推理时间
    void Run(std::vector<cv::Mat> img_list,
             std::vector<std::vector<std::string>> &rec_html_tags,
             std::vector<float> &rec_scores,
             std::vector<std::vector<std::vector<int>>> &rec_boxes,
             std::vector<double> &times);
// 私有成员变量，用于存储预测器指针
std::shared_ptr<paddle_infer::Predictor> predictor_;

// 是否使用 GPU，默认为 false
bool use_gpu_ = false;
// GPU 的设备 ID，默认为 0
int gpu_id_ = 0;
// GPU 内存大小，默认为 4000
int gpu_mem_ = 4000;
// CPU 数学库线程数，默认为 4
int cpu_math_library_num_threads_ = 4;
// 是否使用 MKL-DNN，默认为 false
bool use_mkldnn_ = false;
// 表格最大长度，默认为 488
int table_max_len_ = 488;

// 均值数组，用于图像预处理，默认为 {0.485f, 0.456f, 0.406f}
std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
// 缩放数组，用于图像预处理，默认为 {1 / 0.229f, 1 / 0.224f, 1 / 0.225f}
std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
// 是否进行缩放，默认为 true
bool is_scale_ = true;

// 是否使用 TensorRT，默认为 false
bool use_tensorrt_ = false;
// 精度设置，默认为 "fp32"
std::string precision_ = "fp32";
// 表格批次数，默认为 1
int table_batch_num_ = 1;

// 预处理操作对象
TableResizeImg resize_op_;
Normalize normalize_op_;
PermuteBatch permute_op_;
TablePadImg pad_op_;

// 后处理操作对象
TablePostProcessor post_processor_;

}; // class StructureTableRecognizer

} // namespace PaddleOCR
```