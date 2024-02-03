# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ppredictor.cpp`

```
namespace ppredictor {
// 定义 PPredictor 类的构造函数，初始化成员变量
PPredictor::PPredictor(int use_opencl, int thread_num, int net_flag,
                       paddle::lite_api::PowerMode mode)
    : _use_opencl(use_opencl), _thread_num(thread_num), _net_flag(net_flag), _mode(mode) {}

// 从内存中初始化模型
int PPredictor::init_nb(const std::string &model_content) {
  // 创建 MobileConfig 对象，并从内存中设置模型内容
  paddle::lite_api::MobileConfig config;
  config.set_model_from_buffer(model_content);
  // 调用 _init 函数进行初始化
  return _init(config);
}

// 从文件中初始化模型
int PPredictor::init_from_file(const std::string &model_content) {
  // 创建 MobileConfig 对象，并从文件中设置模型内容
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_content);
  // 调用 _init 函数进行初始化
  return _init(config);
}

// 初始化函数模板
template <typename ConfigT> int PPredictor::_init(ConfigT &config) {
  // 检查 OpenCL 后端是否有效
  bool is_opencl_backend_valid = paddle::lite_api::IsOpenCLBackendValid(/*check_fp16_valid = false*/);
  if (is_opencl_backend_valid) {
    if (_use_opencl != 0) {
      // 设置 OpenCL 二进制路径和名称
      const std::string bin_path = "/data/local/tmp/";
      const std::string bin_name = "lite_opencl_kernel.bin";
      config.set_opencl_binary_path_name(bin_path, bin_name);

      // 设置 OpenCL 调优选项
      const std::string tuned_path = "/data/local/tmp/";
      const std::string tuned_name = "lite_opencl_tuned.bin";
      config.set_opencl_tune(paddle::lite_api::CL_TUNE_NORMAL, tuned_path, tuned_name);

      // 设置 OpenCL 精度选项为 fp32
      config.set_opencl_precision(paddle::lite_api::CL_PRECISION_FP32);
      LOGI("ocr cpp device: running on gpu.");
    }
  } else {
    LOGI("ocr cpp device: running on cpu.");
    // 可以提供备用的 CPU 模型
    // 从文件中设置模型参数
    // config.set_model_from_file(cpu_nb_model_dir);
  }
  // 设置线程数
  config.set_threads(_thread_num);
  // 设置预测模式
  config.set_power_mode(_mode);
  // 创建 Paddle 预测器对象
  _predictor = paddle::lite_api::CreatePaddlePredictor(config);
  // 输出信息，表示 OCR C++ Paddle 实例已创建
  LOGI("ocr cpp paddle instance created");
  // 返回成功标识
  return RETURN_OK;
}

// 获取指定索引的输入数据，并返回 PredictorInput 对象
PredictorInput PPredictor::get_input(int index) {
  // 使用 _predictor 对象获取指定索引的输入数据
  PredictorInput input{_predictor->GetInput(index), index, _net_flag};
  // 标记输入已获取
  _is_input_get = true;
  // 返回 PredictorInput 对象
  return input;
}

// 获取多个输入数据，并返回包含这些数据的 PredictorInput 对象的向量
std::vector<PredictorInput> PPredictor::get_inputs(int num) {
  // 创建结果向量
  std::vector<PredictorInput> results;
  // 遍历获取指定数量的输入数据
 for (int i = 0; i < num; i++) {
    // 获取第 i 个输入数据并添加到结果向量中
    results.emplace_back(get_input(i));
  }
  // 返回结果向量
  return results;
}

// 获取第一个输入数据
PredictorInput PPredictor::get_first_input() { return get_input(0); }

// 进行推理操作，返回推理结果的向量
std::vector<PredictorOutput> PPredictor::infer() {
  // 输出日志信息
  LOGI("ocr cpp infer Run start %d", _net_flag);
  // 创建结果向量
  std::vector<PredictorOutput> results;
  // 如果输入数据未获取，则返回空结果向量
  if (!_is_input_get) {
    return results;
  }
  // 运行预测器
  _predictor->Run();
  // 输出日志信息
  LOGI("ocr cpp infer Run end");

  // 遍历获取输出数据并添加到结果向量中
  for (int i = 0; i < _predictor->GetOutputNames().size(); i++) {
    // 获取输出张量
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        _predictor->GetOutput(i);
    // 输出日志信息
    LOGI("ocr cpp output tensor[%d] size %ld", i, product(output_tensor->shape()));
    // 创建 PredictorOutput 对象并添加到结果向量中
    PredictorOutput result{std::move(output_tensor), i, _net_flag};
    results.emplace_back(std::move(result));
  }
  // 返回结果向量
  return results;
}

// 获取网络标识
NET_TYPE PPredictor::get_net_flag() const { return (NET_TYPE)_net_flag; }
}
```