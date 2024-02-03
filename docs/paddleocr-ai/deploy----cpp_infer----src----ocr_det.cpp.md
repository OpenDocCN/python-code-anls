# `.\PaddleOCR\deploy\cpp_infer\src\ocr_det.cpp`

```
// 包含 PaddleOCR 的头文件 ocr_det.h
#include <include/ocr_det.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 加载模型的方法
void DBDetector::LoadModel(const std::string &model_dir) {
  // 创建 PaddlePaddle 推理配置对象
  paddle_infer::Config config;
  // 设置模型文件路径和参数文件路径
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  // 如果使用 GPU
  if (this->use_gpu_) {
    // 启用 GPU，并设置 GPU 内存和 GPU ID
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    // 如果使用 TensorRT
    if (this->use_tensorrt_) {
      // 设置精度为 float32
      auto precision = paddle_infer::Config::Precision::kFloat32;
      // 如果精度为 fp16
      if (this->precision_ == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      // 如果精度为 int8
      if (this->precision_ == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      // 启用 TensorRT 引擎
      config.EnableTensorRtEngine(1 << 30, 1, 20, precision, false, false);
      // 如果不存在 trt_det_shape.txt 文件
      if (!Utility::PathExists("./trt_det_shape.txt")) {
        // 收集形状范围信息到 trt_det_shape.txt 文件
        config.CollectShapeRangeInfo("./trt_det_shape.txt");
      } else {
        // 启用调整后的 TensorRT 动态形状
        config.EnableTunedTensorRtDynamicShape("./trt_det_shape.txt", true);
      }
    }
  } else {
    // 禁用 GPU
    config.DisableGpu();
    // 如果使用 MKLDNN
    if (this->use_mkldnn_) {
      // 启用 MKLDNN
      config.EnableMKLDNN();
      // 缓存 10 种不同的形状以避免内存泄漏
      config.SetMkldnnCacheCapacity(10);
    }
    // 设置 CPU 数学库的线程数
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }
  // 将 zero_copy_run 设置为默认值
  config.SwitchUseFeedFetchOps(false);
  // 设置为 true 表示有多个输入
  config.SwitchSpecifyInputNames(true);

  // 打开 IR 优化
  config.SwitchIrOptim(true);

  // 启用内存优化
  config.EnableMemoryOptim();
  // 禁用 Glog 信息输出
  // config.DisableGlogInfo();

  // 创建预测器对象
  this->predictor_ = paddle_infer::CreatePredictor(config);
// 运行 DBDetector 类的 Run 方法，对输入图像进行文字检测
void DBDetector::Run(cv::Mat &img,
                     std::vector<std::vector<std::vector<int>>> &boxes,
                     std::vector<double> &times) {
  // 定义高度和宽度的比例
  float ratio_h{};
  float ratio_w{};

  // 复制输入图像到 srcimg
  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);

  // 记录预处理开始时间
  auto preprocess_start = std::chrono::steady_clock::now();
  // 运行 resize_op_ 对象的 Run 方法，对图像进行缩放和裁剪
  this->resize_op_.Run(img, resize_img, this->limit_type_,
                       this->limit_side_len_, ratio_h, ratio_w,
                       this->use_tensorrt_);

  // 运行 normalize_op_ 对象的 Run 方法，对 resize_img 进行归一化
  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  // 创建输入数据 input，大小为 1 * 3 * resize_img.rows * resize_img.cols
  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  // 运行 permute_op_ 对象的 Run 方法，对 resize_img 进行排列
  this->permute_op_.Run(&resize_img, input.data());
  // 记录预处理结束时间
  auto preprocess_end = std::chrono::steady_clock::now();

  // 推理过程
  // 获取输入名称
  auto input_names = this->predictor_->GetInputNames();
  // 获取输入数据句柄
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  // 重新设置输入数据形状
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  // 记录推理开始时间
  auto inference_start = std::chrono::steady_clock::now();
  // 将输入数据从 CPU 复制到 input_t
  input_t->CopyFromCpu(input.data());

  // 运行预测器
  this->predictor_->Run();

  // 获取输出数据
  std::vector<float> out_data;
  // 获取输出名称
  auto output_names = this->predictor_->GetOutputNames();
  // 获取输出数据句柄
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  // 获取输出数据形状
  std::vector<int> output_shape = output_t->shape();
  // 计算输出数据的元素个数
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  // 调整 out_data 的大小
  out_data.resize(out_num);
  // 将输出数据从 output_t 复制到 CPU
  output_t->CopyToCpu(out_data.data());
  // 记录推理结束时间
  auto inference_end = std::chrono::steady_clock::now();

  // 后处理过程
  auto postprocess_start = std::chrono::steady_clock::now();
  // 获取输出数据的第三维和第四维大小
  int n2 = output_shape[2];
  int n3 = output_shape[3];
  int n = n2 * n3;

  // 创建预测数据 pred 和字符缓冲区 cbuf
  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  // 遍历输出数据，将其转换为浮点数并存储在 pred 中
  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
  // 将浮点数数据乘以255转换为无符号字符类型，并存储在cbuf数组中
  cbuf[i] = (unsigned char)((out_data[i]) * 255);
}

// 创建CV_8UC1类型的Mat对象cbuf_map，用于存储cbuf数组的数据
cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
// 创建CV_32F类型的Mat对象pred_map，用于存储pred数组的数据
cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

// 设置阈值和最大值
const double threshold = this->det_db_thresh_ * 255;
const double maxvalue = 255;
// 创建bit_map对象，用于存储阈值化后的数据
cv::Mat bit_map;
// 对cbuf_map进行阈值化处理，结果存储在bit_map中
cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

// 如果使用膨胀操作
if (this->use_dilation_) {
  // 创建膨胀操作的结构元素
  cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
  // 对bit_map进行膨胀操作
  cv::dilate(bit_map, bit_map, dila_ele);
}

// 从预测数据和二值化数据中获取边界框
boxes = post_processor_.BoxesFromBitmap(
    pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
    this->det_db_score_mode_);

// 过滤边界框检测结果
boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
auto postprocess_end = std::chrono::steady_clock::now();

// 计算预处理时间
std::chrono::duration<float> preprocess_diff =
    preprocess_end - preprocess_start;
times.push_back(double(preprocess_diff.count() * 1000));
// 计算推理时间
std::chrono::duration<float> inference_diff = inference_end - inference_start;
times.push_back(double(inference_diff.count() * 1000));
// 计算后处理时间
std::chrono::duration<float> postprocess_diff =
    postprocess_end - postprocess_start;
times.push_back(double(postprocess_diff.count() * 1000));
}
``` 


} // namespace PaddleOCR
```