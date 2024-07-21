# `.\pytorch\test\custom_operator\op.h`

```
// 包含 PyTorch C++ 脚本接口头文件
#include <torch/script.h>

// 包含标准库头文件：大小、向量和字符串
#include <cstddef>
#include <vector>
#include <string>

// 关闭 clang 格式化器以保留预处理指令原始格式
// clang-format off
#  if defined(_WIN32)
// 如果是 Windows 平台
#    if defined(custom_ops_EXPORTS)
// 如果定义了 custom_ops_EXPORTS 宏，表示正在导出符号
#      define CUSTOM_OP_API __declspec(dllexport)
#    else
// 否则，表示正在导入符号
#      define CUSTOM_OP_API __declspec(dllimport)
#    endif
#  else
// 如果不是 Windows 平台，则不需要额外的导出/导入符号修饰
#    define CUSTOM_OP_API
#  endif
// clang-format on

// 声明一个自定义的 Torch 函数 custom_op，返回一个 Tensor 列表
CUSTOM_OP_API torch::List<torch::Tensor> custom_op(
    torch::Tensor tensor,  // 输入的 Tensor 参数
    double scalar,         // 输入的标量参数
    int64_t repeat);       // 输入的重复次数参数

// 声明一个自定义的 Torch 函数 custom_op2，返回一个 int64_t 类型的值
CUSTOM_OP_API int64_t custom_op2(
    std::string s1,  // 第一个输入的字符串参数
    std::string s2); // 第二个输入的字符串参数
```