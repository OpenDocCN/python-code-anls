# `.\pytorch\aten\src\ATen\core\ATenOpList.h`

```
#pragma once


// 使用 pragma once 防止头文件被多次包含，确保编译效率和避免重定义错误



#include <c10/macros/Export.h>


// 包含 c10 库中的 Export.h 头文件，其中可能定义了导出符号的宏或宏定义



namespace c10 {
struct OperatorName;
}


// 声明命名空间 c10，用于组织和封装 c10 库的内容
// 声明结构体 OperatorName，该结构体可能用于表示运算符的名称



namespace at {


// 声明命名空间 at，用于组织和封装 at 模块的内容



// check if an op is a custom op (i.e. did not come from native_functions.yaml)


// 检查一个运算符是否是自定义运算符（即不是来自 native_functions.yaml 文件）



TORCH_API bool is_custom_op(const c10::OperatorName& opName);


// 声明 TORCH_API 修饰的函数 is_custom_op，用于判断给定运算符名称 opName 是否是自定义运算符
// 函数返回布尔值，true 表示是自定义运算符，false 表示不是



}


// 结束命名空间 at 的定义
```