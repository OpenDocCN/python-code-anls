# `.\pytorch\test\cpp\rpc\e2e_test_base.cpp`

```py
# 包含自定义头文件 "e2e_test_base.h"
#include "e2e_test_base.h"

# 进入命名空间 torch::distributed::rpc
namespace torch {
namespace distributed {
namespace rpc {

# 定义静态函数 getDistAutogradContainer，返回 DistAutogradContainer* 指针
DistAutogradContainer* getDistAutogradContainer() {
  # 静态变量 autogradContainer，用于保存 DistAutogradContainer 的实例化对象，
  # 并初始化为 DistAutogradContainer 类的静态方法 init(0) 的返回值
  static DistAutogradContainer* autogradContainer =
      &DistAutogradContainer::init(0);
  # 返回静态变量 autogradContainer 指针
  return autogradContainer;
}

# 定义 TestE2EBase 类中的静态常量 serverAddress，赋值为字符串 "127.0.0.1"
const char* TestE2EBase::serverAddress = "127.0.0.1";

# 定义 TestE2EBase 类中的静态常量 numIters，赋值为数值 100
const size_t TestE2EBase::numIters = 100;

# 定义 TestE2EBase 类中的静态常量 numWorkers，赋值为数值 1
const size_t TestE2EBase::numWorkers = 1;

} // namespace rpc
} // namespace distributed
} // namespace torch
```