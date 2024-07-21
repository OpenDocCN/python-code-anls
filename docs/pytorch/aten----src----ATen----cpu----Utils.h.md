# `.\pytorch\aten\src\ATen\cpu\Utils.h`

```
#pragma once

# 预处理指令，表示本头文件在编译时只包含一次


#include <c10/macros/Export.h>

# 包含外部头文件 `Export.h`，其中可能定义了导出宏


namespace at::cpu {

# 进入命名空间 `at::cpu`


TORCH_API bool is_cpu_support_avx2();

# 声明函数原型 `is_cpu_support_avx2()`，用于检测CPU是否支持 AVX2 指令集


TORCH_API bool is_cpu_support_avx512();

# 声明函数原型 `is_cpu_support_avx512()`，用于检测CPU是否支持 AVX-512 指令集


// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_avx512_vnni();

# 声明函数原型 `is_cpu_support_avx512_vnni()`，用于检测CPU是否支持 AVX-512 VNNI 指令集


// Detect if CPU support Advanced Matrix Extension.
TORCH_API bool is_cpu_support_amx_tile();

# 声明函数原型 `is_cpu_support_amx_tile()`，用于检测CPU是否支持 AMX 指令集


// Enable the system to use AMX instructions.
TORCH_API bool init_amx();

# 声明函数原型 `init_amx()`，用于启用系统使用 AMX 指令集


} // namespace at::cpu

# 结束命名空间 `at::cpu`
```