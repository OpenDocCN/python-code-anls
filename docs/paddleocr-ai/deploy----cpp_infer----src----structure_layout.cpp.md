# `.\PaddleOCR\deploy\cpp_infer\src\structure_layout.cpp`

```py
// 版权声明，声明代码版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证版本 2.0 进行许可;
// 除非符合许可证的规定，否则您不得使用此文件。
// 您可以在以下网址获取许可证的副本:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 均基于“原样”分发，没有任何明示或暗示的保证或条件。
// 请查看许可证以获取特定语言的权限和限制。
//
// 包含结构布局的头文件
#include <include/structure_layout.h>

// 使用 PaddleOCR 命名空间
namespace PaddleOCR {
// 运行结构布局识别器，对输入图像进行处理并返回结果和时间
void StructureLayoutRecognizer::Run(cv::Mat img,
                                    std::vector<StructurePredictResult> &result,
                                    std::vector<double> &times) {
  // 计算预处理时间
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  // 计算推理时间
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  // 计算后处理时间
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  // 预处理
  auto preprocess_start = std::chrono::steady_clock::now();

  // 复制输入图像
  cv::Mat srcimg;
  img.copyTo(srcimg);
  // 调整图像大小
  cv::Mat resize_img;
  this->resize_op_.Run(srcimg, resize_img, 800, 608);
  // 归一化处理
  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  // 创建输入向量
  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();
  preprocess_diff += preprocess_end - preprocess_start;

  // 推理
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  auto inference_start = std::chrono::steady_clock::now();
  input_t->CopyFromCpu(input.data());

  this->predictor_->Run();

  // 获取输出张量
  std::vector<std::vector<float>> out_tensor_list;
  std::vector<std::vector<int>> output_shape_list;
  auto output_names = this->predictor_->GetOutputNames();
  for (int j = 0; j < output_names.size(); j++) {
    auto output_tensor = this->predictor_->GetOutputHandle(output_names[j]);
    std::vector<int> output_shape = output_tensor->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    output_shape_list.push_back(output_shape);
    // 创建一个存储浮点数的向量 out_data，并设置其大小为 out_num
    std::vector<float> out_data;
    out_data.resize(out_num);
    // 将输出张量的数据复制到 CPU 中的 out_data 中
    output_tensor->CopyToCpu(out_data.data());
    // 将 out_data 添加到 out_tensor_list 中
    out_tensor_list.push_back(out_data);
    
    // 记录推理结束时间点
    auto inference_end = std::chrono::steady_clock::now();
    // 计算推理时间
    inference_diff += inference_end - inference_start;
    
    // 进行后处理
    auto postprocess_start = std::chrono::steady_clock::now();
    
    // 创建一个存储整数的向量 bbox_num
    std::vector<int> bbox_num;
    // 初始化 reg_max 为 0
    int reg_max = 0;
    // 遍历 out_tensor_list
    for (int i = 0; i < out_tensor_list.size(); i++) {
      // 如果 i 等于 post_processor_.fpn_stride_.size()
      if (i == this->post_processor_.fpn_stride_.size()) {
        // 计算 reg_max
        reg_max = output_shape_list[i][2] / 4;
        // 退出循环
        break;
      }
    }
    // 创建存储整数的向量 ori_shape 和 resize_shape
    std::vector<int> ori_shape = {srcimg.rows, srcimg.cols};
    std::vector<int> resize_shape = {resize_img.rows, resize_img.cols};
    // 运行后处理器的 Run 方法
    this->post_processor_.Run(result, out_tensor_list, ori_shape, resize_shape, reg_max);
    // 将 result 的大小添加到 bbox_num 中
    bbox_num.push_back(result.size());
    
    // 记录后处理结束时间点
    auto postprocess_end = std::chrono::steady_clock::now();
    // 计算后处理时间
    postprocess_diff += postprocess_end - postprocess_start;
    // 将预处理时间、推理时间和后处理时间添加到 times 中
    times.push_back(double(preprocess_diff.count() * 1000));
    times.push_back(double(inference_diff.count() * 1000));
    times.push_back(double(postprocess_diff.count() * 1000));
// 结构布局识别器类的加载模型函数，用于加载模型文件
void StructureLayoutRecognizer::LoadModel(const std::string &model_dir) {
  // 创建配置对象
  paddle_infer::Config config;
  // 检查是否存在推理模型文件和参数文件
  if (Utility::PathExists(model_dir + "/inference.pdmodel") &&
      Utility::PathExists(model_dir + "/inference.pdiparams")) {
    // 设置推理模型和参数文件路径
    config.SetModel(model_dir + "/inference.pdmodel",
                    model_dir + "/inference.pdiparams");
  } else if (Utility::PathExists(model_dir + "/model.pdmodel") &&
             Utility::PathExists(model_dir + "/model.pdiparams")) {
    // 设置模型和参数文件路径
    config.SetModel(model_dir + "/model.pdmodel",
                    model_dir + "/model.pdiparams");
  } else {
    // 输出错误信息并退出程序
    std::cerr << "[ERROR] not find model.pdiparams or inference.pdiparams in "
              << model_dir << std::endl;
    exit(1);
  }

  // 如果使用 GPU
  if (this->use_gpu_) {
    // 启用 GPU，并设置 GPU 内存和 GPU ID
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    // 如果使用 TensorRT
    if (this->use_tensorrt_) {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (this->precision_ == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      if (this->precision_ == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      // 启用 TensorRT 引擎
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      // 如果不存在 trt_layout_shape.txt 文件，则收集形状范围信息
      if (!Utility::PathExists("./trt_layout_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_layout_shape.txt");
      } else {
        // 启用调整后的 TensorRT 动态形状
        config.EnableTunedTensorRtDynamicShape("./trt_layout_shape.txt", true);
      }
    }
  } else {
    // 禁用 GPU
    config.DisableGpu();
    // 如果使用 MKLDNN
    if (this->use_mkldnn_) {
      // 启用 MKLDNN
      config.EnableMKLDNN();
    }
    // 设置 CPU 数学库线程数
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // 关闭零拷贝张量
  config.SwitchUseFeedFetchOps(false);
  // 启用指定输入名称
  config.SwitchSpecifyInputNames(true);

  // 启用 IR 优化
  config.SwitchIrOptim(true);

  // 启用内存优化
  config.EnableMemoryOptim();
  // 禁用 Glog 信息
  config.DisableGlogInfo();

  // 创建预测器对象
  this->predictor_ = paddle_infer::CreatePredictor(config);
}
} // namespace PaddleOCR
```