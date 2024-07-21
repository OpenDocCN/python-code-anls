# `.\pytorch\aten\src\ATen\Version.h`

```
#include <ATen/Context.h>

namespace at {

/// 返回一个详细描述 PyTorch 配置的字符串。
TORCH_API std::string show_config();

/// 返回 MKL 版本的字符串。
TORCH_API std::string get_mkl_version();

/// 返回 MKL-DNN 版本的字符串。
TORCH_API std::string get_mkldnn_version();

/// 返回 OpenMP 版本的字符串。
TORCH_API std::string get_openmp_version();

/// 返回 C++ 编译标志的字符串。
TORCH_API std::string get_cxx_flags();

/// 返回 CPU 能力的字符串。
TORCH_API std::string get_cpu_capability();

} // namespace at
```