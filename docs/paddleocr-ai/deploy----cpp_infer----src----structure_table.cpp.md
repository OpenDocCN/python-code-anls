# `.\PaddleOCR\deploy\cpp_infer\src\structure_table.cpp`

```py
// 版权声明，告知代码版权归属及许可协议
// 包含结构表头文件

namespace PaddleOCR {

// 结构表识别器运行函数，接受图像列表、结构 HTML 标签、结构得分、结构框、时间列表作为参数
void StructureTableRecognizer::Run(
    std::vector<cv::Mat> img_list,
    std::vector<std::vector<std::string>> &structure_html_tags,
    std::vector<float> &structure_scores,
    std::vector<std::vector<std::vector<int>>> &structure_boxes,
    std::vector<double> &times) {
  // 计算预处理、推理和后处理的时间差
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  // 获取图像数量
  int img_num = img_list.size();
  // 循环处理图像列表
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->table_batch_num_) {
    // 预处理开始时间
    auto preprocess_start = std::chrono::steady_clock::now();
    // 计算结束图像编号和批次数量
    int end_img_no = std::min(img_num, beg_img_no + this->table_batch_num_);
    int batch_num = end_img_no - beg_img_no;
    // 初始化归一化图像批次、宽度列表和高度列表
    std::vector<cv::Mat> norm_img_batch;
    std::vector<int> width_list;
    std::vector<int> height_list;
    // 遍历图像列表，对每张图像进行预处理操作
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      // 从图像列表中复制一张图像到 srcimg
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      // 创建 resize_img 和 pad_img 用于存储处理后的图像
      cv::Mat resize_img;
      cv::Mat pad_img;
      // 运行 resize_op 对 srcimg 进行尺寸调整
      this->resize_op_.Run(srcimg, resize_img, this->table_max_len_);
      // 运行 normalize_op 对 resize_img 进行归一化处理
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      // 运行 pad_op 对 resize_img 进行填充操作
      this->pad_op_.Run(resize_img, pad_img, this->table_max_len_);
      // 将处理后的图像添加到 norm_img_batch 中
      norm_img_batch.push_back(pad_img);
      // 记录原始图像的宽度和高度
      width_list.push_back(srcimg.cols);
      height_list.push_back(srcimg.rows);
    }

    // 创建用于存储输入数据的 input 向量
    std::vector<float> input(
        batch_num * 3 * this->table_max_len_ * this->table_max_len_, 0.0f);
    // 运行 permute_op 对 norm_img_batch 进行排列操作，将结果存储到 input 中
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;
    // 推理过程
    // 获取输入名称
    auto input_names = this->predictor_->GetInputNames();
    // 获取输入句柄
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    // 重新调整输入数据的形状
    input_t->Reshape(
        {batch_num, 3, this->table_max_len_, this->table_max_len_});
    auto inference_start = std::chrono::steady_clock::now();
    // 将输入数据从 CPU 复制到输入句柄
    input_t->CopyFromCpu(input.data());
    // 运行预测
    this->predictor_->Run();
    // 获取输出名称
    auto output_names = this->predictor_->GetOutputNames();
    // 获取输出句柄
    auto output_tensor0 = this->predictor_->GetOutputHandle(output_names[0]);
    auto output_tensor1 = this->predictor_->GetOutputHandle(output_names[1]);
    // 获取预测结果的形状
    std::vector<int> predict_shape0 = output_tensor0->shape();
    std::vector<int> predict_shape1 = output_tensor1->shape();

    // 计算输出数据的元素个数
    int out_num0 = std::accumulate(predict_shape0.begin(), predict_shape0.end(),
                                   1, std::multiplies<int>());
    int out_num1 = std::accumulate(predict_shape1.begin(), predict_shape1.end(),
                                   1, std::multiplies<int>());
    // 创建用于存储预测结果的 loc_preds 和 structure_probs 向量
    std::vector<float> loc_preds;
    std::vector<float> structure_probs;
    loc_preds.resize(out_num0);
    structure_probs.resize(out_num1);
    // 将输出张量0的数据复制到CPU
    output_tensor0->CopyToCpu(loc_preds.data());
    // 将输出张量1的数据复制到CPU
    output_tensor1->CopyToCpu(structure_probs.data());
    // 记录推理结束时间点
    auto inference_end = std::chrono::steady_clock::now();
    // 计算推理时间差
    inference_diff += inference_end - inference_start;
    // 进行后处理
    auto postprocess_start = std::chrono::steady_clock::now();
    // 初始化存储结构化HTML标签的二维向量
    std::vector<std::vector<std::string>> structure_html_tag_batch;
    // 初始化存储结构化得分的一维向量
    std::vector<float> structure_score_batch;
    // 初始化存储结构化框的三维向量
    std::vector<std::vector<std::vector<int>>> structure_boxes_batch;
    // 运行后处理器，生成结构化HTML标签、结构化框和结构化得分
    this->post_processor_.Run(loc_preds, structure_probs, structure_score_batch,
                              predict_shape0, predict_shape1,
                              structure_html_tag_batch, structure_boxes_batch,
                              width_list, height_list);
    // 遍历每个样本
    for (int m = 0; m < predict_shape0[0]; m++) {
      // 在结构化HTML标签的开头插入标签
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<table>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<body>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                         "<html>");
      // 在结构化HTML标签的末尾添加标签
      structure_html_tag_batch[m].push_back("</table>");
      structure_html_tag_batch[m].push_back("</body>");
      structure_html_tag_batch[m].push_back("</html>");
      // 将当前样本的结构化HTML标签、结构化得分和结构化框添加到对应的向量中
      structure_html_tags.push_back(structure_html_tag_batch[m]);
      structure_scores.push_back(structure_score_batch[m]);
      structure_boxes.push_back(structure_boxes_batch[m]);
    }
    // 记录后处理结束时间点
    auto postprocess_end = std::chrono::steady_clock::now();
    // 计算后处理时间差
    postprocess_diff += postprocess_end - postprocess_start;
    // 将预处理时间、推理时间和后处理时间的毫秒数添加到时间向量中
    times.push_back(double(preprocess_diff.count() * 1000));
    times.push_back(double(inference_diff.count() * 1000));
    times.push_back(double(postprocess_diff.count() * 1000));
  }
// 结构表识别器类的加载模型函数，用于加载模型文件
void StructureTableRecognizer::LoadModel(const std::string &model_dir) {
  // 创建配置对象
  paddle_infer::Config config;
  // 设置模型文件路径
 config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  // 如果使用 GPU
  if (this->use_gpu_) {
    // 启用 GPU，并设置 GPU 内存和 GPU ID
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    // 如果使用 TensorRT
    if (this->use_tensorrt_) {
      // 设置精度
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (this->precision_ == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      if (this->precision_ == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      // 启用 TensorRT 引擎
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      // 如果不存在表格形状文件
      if (!Utility::PathExists("./trt_table_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_table_shape.txt");
      } else {
        config.EnableTunedTensorRtDynamicShape("./trt_table_shape.txt", true);
      }
    }
  } else {
    // 禁用 GPU
    config.DisableGpu();
    // 如果使用 MKLDNN
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
    }
    // 设置 CPU 数学库线程数
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // 关闭零拷贝张量
  config.SwitchUseFeedFetchOps(false);
  // 开启多输入
  config.SwitchSpecifyInputNames(true);

  // 开启 IR 优化
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