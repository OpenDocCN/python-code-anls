# `.\pytorch\c10\core\DefaultTensorOptions.h`

```
#pragma once


// 声明文件仅被编译一次，避免重复包含

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>

namespace c10 {

struct TensorOptions;

/// Like TensorOptions, but all fields are guaranteed to be filled.
struct DefaultTensorOptions {
  DefaultTensorOptions() = default;

  // 返回数据类型（dtype）
  caffe2::TypeMeta dtype() const noexcept {
    return dtype_;
  }

  // 返回设备类型
  Device device() const noexcept {
    return device_;
  }

  // 返回张量布局
  Layout layout() const noexcept {
    return layout_;
  }

  // 返回是否需要梯度
  bool requires_grad() const noexcept {
    return requires_grad_;
  }

  // 合并张量选项，来自TensorOptions.h中的定义
  inline DefaultTensorOptions& merge(const TensorOptions& options);

 private:
  // 数据类型默认为64位浮点数
  caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>();
  // 设备默认为CPU
  Device device_ = at::kCPU;
  // 布局默认为Strided（步幅布局）
  Layout layout_ = at::kStrided;
  // 默认不需要梯度
  bool requires_grad_ = false;
};

// 获取默认张量选项的静态实例
inline const DefaultTensorOptions& getDefaultTensorOptions() {
  static const auto options = DefaultTensorOptions();
  return options;
}

} // namespace c10
```