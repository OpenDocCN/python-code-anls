# `.\pytorch\aten\src\ATen\native\verbose_wrapper.cpp`

```py
// 包含 ATen 库的配置文件，用于获取配置信息
#include <ATen/Config.h>

// 如果 ATen 使用了 MKL，则包含 MKL 头文件
#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

// 如果 ATen 使用了 MKLDNN，则包含 MKLDNN 公共头文件
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#endif

// 包含 ATen 的 verbose 包装器头文件
#include <ATen/native/verbose_wrapper.h>

// 定义了 torch::verbose 命名空间
namespace torch::verbose {

// 设置 MKL 的详细信息输出级别
int _mkl_set_verbose(int enable) {
#if AT_MKL_ENABLED()
  // 调用 mkl_verbose 函数设置 MKL 的详细信息输出级别
  int ret = mkl_verbose(enable);

  // 如果 mkl_verbose 设置失败，则返回 0
  // 设置成功返回 1
  return ret != -1;
#else
  // 如果未启用 oneMKL，则始终返回 0
  return 0;
#endif
}

// 设置 MKLDNN 的详细信息输出级别
int _mkldnn_set_verbose(int level) {
#if AT_MKLDNN_ENABLED()
  // 调用 ATen 的 set_verbose 函数设置 MKLDNN 的详细信息输出级别
  return at::native::set_verbose(level);
#else
  // 如果未启用 MKLDNN，则始终返回 0
  return 0;
#endif
}

} // namespace torch::verbose
```