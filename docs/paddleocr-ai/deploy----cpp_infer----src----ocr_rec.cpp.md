# `.\PaddleOCR\deploy\cpp_infer\src\ocr_rec.cpp`

```py
// 包含 OCR 识别器的头文件
#include <include/ocr_rec.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 运行 CRNN 识别器，对输入图像列表进行识别，返回识别文本、置信度、时间等信息
void CRNNRecognizer::Run(std::vector<cv::Mat> img_list,
                         std::vector<std::string> &rec_texts,
                         std::vector<float> &rec_text_scores,
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

  // 获取图像数量
  int img_num = img_list.size();
  // 存储图像宽高比
  std::vector<float> width_list;
  // 计算每张图像的宽高比
  for (int i = 0; i < img_num; i++) {
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  // 对宽高比进行排序，返回排序后的索引
  std::vector<int> indices = Utility::argsort(width_list);

  // 对每个批次的图像进行处理
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->rec_batch_num_) {
    // 记录预处理开始时间
    auto preprocess_start = std::chrono::steady_clock::now();
    // 计算结束图像索引
    int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
    // 计算批次大小
    int batch_num = end_img_no - beg_img_no;
    // 获取图像高度和宽度
    int imgH = this->rec_image_shape_[1];
    int imgW = this->rec_image_shape_[2];
    // 计算最大宽高比
    float max_wh_ratio = imgW * 1.0 / imgH;
    // 遍历指定范围内的图像列表
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      // 获取当前图像的高度和宽度
      int h = img_list[indices[ino]].rows;
      int w = img_list[indices[ino]].cols;
      // 计算当前图像的宽高比
      float wh_ratio = w * 1.0 / h;
      // 更新最大宽高比
      max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
    }

    // 初始化批处理宽度为图像宽度
    int batch_width = imgW;
    // 存储归一化后的图像批次
    std::vector<cv::Mat> norm_img_batch;
    // 遍历指定范围内的图像列表
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      // 复制当前图像到新的 Mat 对象
      cv::Mat srcimg;
      img_list[indices[ino]].copyTo(srcimg);
      // 调用 resize_op_ 对象的 Run 方法进行图像缩放和裁剪
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                           this->use_tensorrt_, this->rec_image_shape_);
      // 调用 normalize_op_ 对象的 Run 方法进行图像归一化
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      // 将归一化后的图像添加到批次中
      norm_img_batch.push_back(resize_img);
      // 更新批处理宽度为最大的图像宽度
      batch_width = std::max(resize_img.cols, batch_width);
    }

    // 初始化输入数据为指定大小的零向量
    std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
    // 调用 permute_op_ 对象的 Run 方法对归一化后的图像批次进行排列
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    // 计算预处理时间
    preprocess_diff += preprocess_end - preprocess_start;
    // 推理阶段
    // 获取输入名称
    auto input_names = this->predictor_->GetInputNames();
    // 获取输入数据句柄
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    // 重新调整输入数据的形状
    input_t->Reshape({batch_num, 3, imgH, batch_width});
    auto inference_start = std::chrono::steady_clock::now();
    // 将输入数据从 CPU 复制到输入句柄
    input_t->CopyFromCpu(input.data());
    // 运行推理
    this->predictor_->Run();

    // 存储推理结果
    std::vector<float> predict_batch;
    // 获取输出名称
    auto output_names = this->predictor_->GetOutputNames();
    // 获取输出数据句柄
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    // 获取预测结果的形状
    auto predict_shape = output_t->shape();

    // 计算预测结果的元素个数
    int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1,
                                  std::multiplies<int>());
    predict_batch.resize(out_num);
    // 将预测结果从输出句柄复制到 CPU
    output_t->CopyToCpu(predict_batch.data());
    auto inference_end = std::chrono::steady_clock::now();
    // 累加推理时间
    inference_diff += inference_end - inference_start;
    // CTC 解码
    auto postprocess_start = std::chrono::steady_clock::now();
    // 遍历预测结果的第一维
    for (int m = 0; m < predict_shape[0]; m++) {
      std::string str_res;
      int argmax_idx;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      // 遍历预测结果的第二维
      for (int n = 0; n < predict_shape[1]; n++) {
        // 获取最大值的索引
        argmax_idx = int(Utility::argmax(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
        // 获取最大值
        max_value = float(*std::max_element(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]));

        // 如果最大值索引大于0且不是连续相同的索引
        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          str_res += label_list_[argmax_idx];
        }
        last_index = argmax_idx;
      }
      score /= count;
      // 如果得分为 NaN，则跳过
      if (std::isnan(score)) {
        continue;
      }
      rec_texts[indices[beg_img_no + m]] = str_res;
      rec_text_scores[indices[beg_img_no + m]] = score;
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    // 累加后处理时间
    postprocess_diff += postprocess_end - postprocess_start;
  }
  // 将预处理时间、推理时间和后处理时间添加到时间列表中
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
// 加载 CRNN 模型，参数为模型目录
void CRNNRecognizer::LoadModel(const std::string &model_dir) {
  // 创建配置对象
  paddle_infer::Config config;
  // 设置模型文件路径
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");
  // 输出提示信息
  std::cout << "In PP-OCRv3, default rec_img_h is 48,"
            << "if you use other model, you should set the param rec_img_h=32"
            << std::endl;
  // 如果使用 GPU
  if (this->use_gpu_) {
    // 启用 GPU，设置 GPU 内存和 GPU ID
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    // 如果使用 TensorRT
    if (this->use_tensorrt_) {
      // 设置精度为 float32
      auto precision = paddle_infer::Config::Precision::kFloat32;
      // 根据精度设置不同的值
      if (this->precision_ == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      if (this->precision_ == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      // 如果文件不存在，则收集形状信息
      if (!Utility::PathExists("./trt_rec_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_rec_shape.txt");
      } else {
        // 启用调整 TensorRT 动态形状
        config.EnableTunedTensorRtDynamicShape("./trt_rec_shape.txt", true);
      }
    }
  } else {
    // 禁用 GPU
    config.DisableGpu();
    // 如果使用 MKLDNN
    if (this->use_mkldnn_) {
      // 启用 MKLDNN
      config.EnableMKLDNN();
      // 设置 MKLDNN 缓存容量
      config.SetMkldnnCacheCapacity(10);
    }
    // 设置 CPU 数学库线程数
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // 获取 pass_builder 对象
  auto pass_builder = config.pass_builder();
  // 删除 "matmul_transpose_reshape_fuse_pass" 优化 pass
  pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");
  // 切换使用 FeedFetchOps
  config.SwitchUseFeedFetchOps(false);
  // 设置是否指定输入名称为 true
  config.SwitchSpecifyInputNames(true);

  // 启用 IR 优化
  config.SwitchIrOptim(true);

  // 启用内存优化
  config.EnableMemoryOptim();
  // 禁用 GlogInfo

  // 创建预测器对象
  this->predictor_ = paddle_infer::CreatePredictor(config);
}

} // namespace PaddleOCR
```