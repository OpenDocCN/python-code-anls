# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\predictor_output.h`

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

class PredictorOutput {
// 定义类 PredictorOutput
public:
  PredictorOutput() {}
  // 默认构造函数

  PredictorOutput(std::unique_ptr<const paddle::lite_api::Tensor> &&tensor,
                  int index, int net_flag)
      : _tensor(std::move(tensor)), _index(index), _net_flag(net_flag) {}
  // 构造函数，接受一个 Tensor 指针、索引和网络标志参数

  const float *get_float_data() const;
  // 返回浮点数数据的指针
  const int *get_int_data() const;
  // 返回整数数据的指针
  int64_t get_size() const;
  // 返回数据大小
  const std::vector<std::vector<uint64_t>> get_lod() const;
  // 返回数据的 Level of Details（LOD）
  const std::vector<int64_t> get_shape() const;
  // 返回数据的形状

  std::vector<float> data;    // 存储浮点数数据，或使用 data_int
  std::vector<int> data_int;  // 存储整数数据，或使用 data
  std::vector<int64_t> shape; // PaddleLite 输出的形状
  std::vector<std::vector<uint64_t>> lod; // PaddleLite 输出的 LOD

private:
  std::unique_ptr<const paddle::lite_api::Tensor> _tensor;
  // 私有成员变量，存储 Tensor 指针
  int _index;
  // 私有成员变量，存储索引
  int _net_flag;
  // 私有成员变量，存储网络标志
};
}
// 命名空间结束
```