# `.\pytorch\test\cpp\api\support.cpp`

```
# 包含测试支持的头文件
#include <test/cpp/api/support.h>

# 定义命名空间 torch 下的 test 命名空间
namespace torch {
namespace test {

# 声明 AutoDefaultDtypeMode 类的静态成员变量 default_dtype_mutex
std::mutex AutoDefaultDtypeMode::default_dtype_mutex;

} // namespace test
} // namespace torch
```