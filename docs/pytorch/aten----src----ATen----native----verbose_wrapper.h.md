# `.\pytorch\aten\src\ATen\native\verbose_wrapper.h`

```
#pragma once

# 防止头文件被多次包含，确保编译时只包含一次该头文件


#include <c10/macros/Export.h>

# 包含C10库中的Export.h头文件，该文件定义了用于导出符号的宏和声明


namespace torch::verbose {

# 定义了一个命名空间torch::verbose，用于封装相关的函数和变量


TORCH_API int _mkl_set_verbose(int enable);

# 在torch::verbose命名空间中声明了一个名为_mkl_set_verbose的函数，用于设置MKL库的详细输出，并返回一个整数结果


TORCH_API int _mkldnn_set_verbose(int level);

# 在torch::verbose命名空间中声明了一个名为_mkldnn_set_verbose的函数，用于设置MKLDNN库的详细输出级别，并返回一个整数结果


} // namespace torch::verbose

# 结束了torch::verbose命名空间的定义
```