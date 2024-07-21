# `.\pytorch\torch\csrc\jit\runtime\static\init.cpp`

```
#include <torch/csrc/jit/runtime/static/init.h>
// 引入静态图模型初始化相关的头文件

#include <torch/csrc/jit/passes/freeze_module.h>
// 引入模型冻结相关的头文件

#include <torch/csrc/jit/runtime/static/fusion.h>
// 引入静态图模型融合相关的头文件

#include <torch/csrc/jit/runtime/static/impl.h>
// 引入静态图模型实现相关的头文件

// This number is a heuristic determined with pytorch/benchmark
#define DEFAULT_FUSION_SIZE 4
// 定义默认的融合大小为4，这个数值是通过pytorch/benchmark得出的启发式值

namespace torch::jit {
// 进入torch::jit命名空间

}
// 空行

} // namespace torch::jit
// 结束torch::jit命名空间的声明
```