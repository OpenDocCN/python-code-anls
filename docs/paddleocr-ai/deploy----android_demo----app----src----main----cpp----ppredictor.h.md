# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ppredictor.h`

```py
#pragma once

#include "paddle_api.h"
#include "predictor_input.h"
#include "predictor_output.h"

namespace ppredictor {

/**
 * PaddleLite Preditor Common Interface
 */
class PPredictor_Interface {
public:
  virtual ~PPredictor_Interface() {}

  // 获取网络类型标志
  virtual NET_TYPE get_net_flag() const = 0;
};

/**
 * Common Predictor
 */
class PPredictor : public PPredictor_Interface {
public:
  // 构造函数，初始化 Common Predictor
  PPredictor(
          int use_opencl, int thread_num, int net_flag = 0,
      paddle::lite_api::PowerMode mode = paddle::lite_api::LITE_POWER_HIGH);

  virtual ~PPredictor() {}

  /**
   * 初始化 PaddleLite 模型，nb 格式，或使用 ini_paddle
   * @param model_content
   * @return 0
   */
  virtual int init_nb(const std::string &model_content);

  // 从文件初始化模型
  virtual int init_from_file(const std::string &model_content);

  // 进行推理
  std::vector<PredictorOutput> infer();

  // 获取预测器
  std::shared_ptr<paddle::lite_api::PaddlePredictor> get_predictor() {
    return _predictor;
  }

  // 获取指定数量的输入
  virtual std::vector<PredictorInput> get_inputs(int num);

  // 获取指定索引的输入
  virtual PredictorInput get_input(int index);

  // 获取第一个输入
  virtual PredictorInput get_first_input();

  // 获取网络类型标志
  virtual NET_TYPE get_net_flag() const;

protected:
  // 初始化函数模板
  template <typename ConfigT> int _init(ConfigT &config);

private:
  int _use_opencl; // 使用 OpenCL
  int _thread_num; // 线程数量
  paddle::lite_api::PowerMode _mode; // 模式
  std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor; // 预测器
  bool _is_input_get = false; // 输入是否获取
  int _net_flag; // 网络类型标志
};
}
```