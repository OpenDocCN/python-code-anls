# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\predictor_input.h`

```py
#pragma once
// 防止头文件被重复包含

#include "common.h"
// 包含自定义的 common.h 头文件
#include <paddle_api.h>
// 包含 PaddlePaddle 的 API 头文件
#include <vector>
// 包含 C++ 标准库中的 vector 头文件

namespace ppredictor {
// 命名空间 ppredictor

class PredictorInput {
// 定义类 PredictorInput
public:
  // 公有成员函数

  PredictorInput(std::unique_ptr<paddle::lite_api::Tensor> &&tensor, int index,
                 int net_flag)
      : _tensor(std::move(tensor)), _index(index), _net_flag(net_flag) {}
  // 构造函数，接受一个独占指针 tensor、索引 index 和网络标志 net_flag

  void set_dims(std::vector<int64_t> dims);
  // 设置输入数据的维度

  float *get_mutable_float_data();
  // 获取可变的浮点数据指针

  void set_data(const float *input_data, int input_float_len);
  // 设置输入数据，接受输入数据指针和数据长度

private:
  // 私有成员变量

  std::unique_ptr<paddle::lite_api::Tensor> _tensor;
  // 独占指针 _tensor，用于存储 Tensor 对象
  bool _is_dims_set = false;
  // 布尔变量 _is_dims_set，表示维度是否已设置
  int _index;
  // 整型变量 _index，表示索引
  int _net_flag;
  // 整型变量 _net_flag，表示网络标志
};
}
// 命名空间结束
```