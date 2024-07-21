# `.\pytorch\aten\src\ATen\native\mkldnn\Common.h`

```
#pragma once
// 只有在 AT_MKLDNN_ENABLED 宏定义被启用时才编译以下代码段

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

namespace at {
namespace native {
namespace mkldnn {

// 定义结构体 ContextConv，用于存储 MKLDNN 卷积操作的上下文信息
struct ContextConv final {
  // 存储打包后的权重 tensor
  ideep::tensor weight_packed_;
  // 可选的偏置 tensor
  std::optional<at::Tensor> at_bias_;
  // 填充信息数组
  std::vector<int64_t> padding_;
  // 步幅信息数组
  std::vector<int64_t> stride_;
  // 膨胀信息数组
  std::vector<int64_t> dilation_;
  // 分组数
  int64_t groups_;
  // MKLDNN 的属性对象
  ideep::attr_t attr_;

  // 禁止默认构造函数
  ContextConv() = delete;

  // 自定义构造函数，初始化各成员变量
  ContextConv(
      ideep::tensor&& weight_packed,
      std::optional<at::Tensor> at_bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups,
      ideep::attr_t attr)
      : weight_packed_(std::move(weight_packed)),
        at_bias_(std::move(at_bias)),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        attr_(attr) {}
};

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
// 结束条件，仅当 AT_MKLDNN_ENABLED 宏定义被启用时结束整个代码段
```