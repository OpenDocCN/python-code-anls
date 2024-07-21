# `.\pytorch\c10\util\UniqueVoidPtr.cpp`

```
# 包含 UniqueVoidPtr.h 头文件，该文件位于 c10/util/ 目录下
#include <c10/util/UniqueVoidPtr.h>

# 进入命名空间 c10::detail，用于定义 c10 库的内部细节
namespace c10::detail {

# deleteNothing 函数的实现，参数为 void* 类型指针，函数体为空，即不执行任何操作
void deleteNothing(void*) {}

} // namespace c10::detail
```