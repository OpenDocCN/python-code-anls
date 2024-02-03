# `.\PaddleOCR\deploy\cpp_infer\src\ocr_cls.cpp`

```py
// 包含版权声明和许可信息
// 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
// 根据 Apache 许可证 2.0 版本许可
// 除非符合许可证的规定，否则不得使用此文件
// 您可以在以下网址获取许可证的副本
//     http://www.apache.org/licenses/LICENSE-2.0
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 基于“按原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

#include <include/ocr_cls.h> // 包含 OCR 分类器的头文件

namespace PaddleOCR { // 命名空间 PaddleOCR

void Classifier::Run(std::vector<cv::Mat> img_list, // 运行分类器的函数，输入图像列表
                     std::vector<int> &cls_labels, // 分类标签的输出向量
                     std::vector<float> &cls_scores, // 分类得分的输出向量
                     std::vector<double> &times) { // 时间信息的输出向量
  std::chrono::duration<float> preprocess_diff = // 初始化预处理时间差
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff = // 初始化推理时间差
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff = // 初始化后处理时间差
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size(); // 获取图像列表的大小
  std::vector<int> cls_image_shape = {3, 48, 192}; // 定义分类器输入图像的形状
  for (int beg_img_no = 0; beg_img_no < img_num; // 循环处理每张图像
       beg_img_no += this->cls_batch_num_) { // 每次处理一批图像
    auto preprocess_start = std::chrono::steady_clock::now(); // 记录预处理开始时间
    int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_); // 计算当前批次的结束图像编号
    int batch_num = end_img_no - beg_img_no; // 计算当前批次的图像数量
    // preprocess
    std::vector<cv::Mat> norm_img_batch; // 初始化归一化后的图像批次
    // 遍历图像序号范围，进行预处理操作
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      // 从图像列表中复制图像到 srcimg
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      // 调用 resize_op_ 对象的 Run 方法，对图像进行缩放操作
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, this->use_tensorrt_,
                           cls_image_shape);

      // 调用 normalize_op_ 对象的 Run 方法，对 resize_img 进行归一化操作
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      // 如果 resize_img 的列数小于 cls_image_shape[2]，则进行边界填充
      if (resize_img.cols < cls_image_shape[2]) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                           cls_image_shape[2] - resize_img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      }
      // 将处理后的图像加入 norm_img_batch
      norm_img_batch.push_back(resize_img);
    }
    // 创建用于存储输入数据的 input 向量
    std::vector<float> input(batch_num * cls_image_shape[0] *
                                 cls_image_shape[1] * cls_image_shape[2],
                             0.0f);
    // 调用 permute_op_ 对象的 Run 方法，对 norm_img_batch 进行排列操作
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;

    // 推理过程
    // 获取输入名称
    auto input_names = this->predictor_->GetInputNames();
    // 获取输入数据句柄
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    // 重新调整输入数据的形状
    input_t->Reshape({batch_num, cls_image_shape[0], cls_image_shape[1],
                      cls_image_shape[2]});
    auto inference_start = std::chrono::steady_clock::now();
    // 将输入数据从 CPU 复制到 input_t
    input_t->CopyFromCpu(input.data());
    // 运行预测
    this->predictor_->Run();

    // 获取预测结果
    std::vector<float> predict_batch;
    // 获取输出名称
    auto output_names = this->predictor_->GetOutputNames();
    // 获取输出数据句柄
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    // 获取预测结果的形状
    auto predict_shape = output_t->shape();

    // 计算输出数据的元素个数
    int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1,
                                  std::multiplies<int>());
    predict_batch.resize(out_num);

    // 将输出数据从 output_t 复制到 CPU
    output_t->CopyToCpu(predict_batch.data());
    auto inference_end = std::chrono::steady_clock::now();
    inference_diff += inference_end - inference_start;

    // 后处理
    // 记录后处理开始时间
    auto postprocess_start = std::chrono::steady_clock::now();
    // 遍历每个批次的预测结果
    for (int batch_idx = 0; batch_idx < predict_shape[0]; batch_idx++) {
      // 获取预测结果中概率最大的类别标签
      int label = int(
          Utility::argmax(&predict_batch[batch_idx * predict_shape[1]],
                          &predict_batch[(batch_idx + 1) * predict_shape[1]));
      // 获取预测结果中概率最大的类别得分
      float score = float(*std::max_element(
          &predict_batch[batch_idx * predict_shape[1]],
          &predict_batch[(batch_idx + 1) * predict_shape[1]));
      // 将类别标签和得分存储到对应的数组中
      cls_labels[beg_img_no + batch_idx] = label;
      cls_scores[beg_img_no + batch_idx] = score;
    }
    // 记录后处理结束时间
    auto postprocess_end = std::chrono::steady_clock::now();
    // 累加后处理时间到总时间中
    postprocess_diff += postprocess_end - postprocess_start;
  }
  // 将预处理时间、推理时间和后处理时间转换为毫秒并存储到时间数组中
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
// 加载模型的方法，参数为模型目录的路径
void Classifier::LoadModel(const std::string &model_dir) {
  // 创建配置对象
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
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      // 如果不存在 trt_cls_shape.txt 文件
      if (!Utility::PathExists("./trt_cls_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_cls_shape.txt");
      } else {
        // 启用调整后的 TensorRT 动态形状
        config.EnableTunedTensorRtDynamicShape("./trt_cls_shape.txt", true);
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
    // 设置 CPU 数学库的线程数
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // 关闭零拷贝张量
  config.SwitchUseFeedFetchOps(false);
  // 启用多输入
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