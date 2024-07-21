# `.\pytorch\aten\src\ATen\mkl\Descriptors.h`

```
#pragma once


// 使用 #pragma once 防止头文件重复包含，确保当前头文件只被编译一次
#include <ATen/mkl/Exceptions.h>
// 包含 ATen 库的 MKL 异常处理头文件
#include <mkl_dfti.h>
// 包含 MKL DFTI 头文件，提供了离散傅立叶变换接口
#include <ATen/Tensor.h>
// 包含 ATen 库的 Tensor 类头文件

namespace at::native {

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR* desc) {
    if (desc != nullptr) {
      MKL_DFTI_CHECK(DftiFreeDescriptor(&desc));
      // 如果描述符不为空，则释放 DFTI 描述符
    }
  }
};

class DftiDescriptor {
public:
  void init(DFTI_CONFIG_VALUE precision, DFTI_CONFIG_VALUE signal_type, MKL_LONG signal_ndim, MKL_LONG* sizes) {
    if (desc_ != nullptr) {
      throw std::runtime_error("DFTI DESCRIPTOR can only be initialized once");
      // 如果描述符已经初始化，则抛出运行时错误
    }
    DFTI_DESCRIPTOR *raw_desc;
    if (signal_ndim == 1) {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
      // 如果信号维度为 1，则创建 DFTI 描述符
    } else {
      MKL_DFTI_CHECK(DftiCreateDescriptor(&raw_desc, precision, signal_type, signal_ndim, sizes));
      // 如果信号维度不为 1，则创建 DFTI 描述符
    }
    desc_.reset(raw_desc);
    // 使用 unique_ptr 管理描述符，确保自动释放内存
  }

  DFTI_DESCRIPTOR *get() const {
    if (desc_ == nullptr) {
      throw std::runtime_error("DFTI DESCRIPTOR has not been initialized");
      // 如果描述符未初始化，则抛出运行时错误
    }
    return desc_.get();
    // 返回描述符指针
  }

private:
  std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter> desc_;
  // 使用 unique_ptr 管理 DFTI_DESCRIPTOR，通过自定义的释放器进行安全释放
};

} // namespace at::native
```