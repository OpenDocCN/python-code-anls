# `.\pytorch\torch\csrc\api\src\imethod.cpp`

```
#include <torch/imethod.h>  // 引入torch库中的imethod头文件

namespace torch {

const std::vector<std::string>& IMethod::getArgumentNames() const {
  // 如果参数名已经初始化过，直接返回保存参数名的向量
  if (isArgumentNamesInitialized_) {
    return argumentNames_;
  }

  // 标记参数名已经初始化
  isArgumentNamesInitialized_ = true;
  // 调用setArgumentNames方法初始化参数名向量
  setArgumentNames(argumentNames_);
  // 返回保存参数名的向量
  return argumentNames_;
}

} // namespace torch
```