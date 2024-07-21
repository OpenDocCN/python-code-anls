# `.\pytorch\aten\src\ATen\test\verify_api_visibility.cpp`

```
# 包含 ATen 库的头文件
#include <ATen/ATen.h>

# 如果 AT_CUDNN_ENABLED 宏已定义，则编译错误，不应在公共头文件中可见
#ifdef AT_CUDNN_ENABLED
#error "AT_CUDNN_ENABLED should not be visible in public headers"
#endif

# 如果 AT_MKL_ENABLED 宏已定义，则编译错误，不应在公共头文件中可见
#ifdef AT_MKL_ENABLED
#error "AT_MKL_ENABLED should not be visible in public headers"
#endif

# 如果 AT_MKLDNN_ENABLED 宏已定义，则编译错误，不应在公共头文件中可见
#ifdef AT_MKLDNN_ENABLED
#error "AT_MKLDNN_ENABLED should not be visible in public headers"
#endif

# 如果 AT_MKLDNN_ACL_ENABLED 宏已定义，则编译错误，不应在公共头文件中可见
#ifdef AT_MKLDNN_ACL_ENABLED
#error "AT_MKLDNN_ACL_ENABLED should not be visible in public headers"
#endif

# 如果 CAFFE2_STATIC_LINK_CUDA 宏已定义，则编译错误，不应在公共头文件中可见
#ifdef CAFFE2_STATIC_LINK_CUDA
#error "CAFFE2_STATIC_LINK_CUDA should not be visible in public headers"
#endif

# 定义主函数，返回整型
auto main() -> int {}
```