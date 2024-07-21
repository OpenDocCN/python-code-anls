# `.\pytorch\torch\csrc\jit\codegen\onednn\register_interface.cpp`

```py
// 引入 Torch 的 JIT 运行时性能分析记录头文件
#include <torch/csrc/jit/runtime/profiling_record.h>

// 声明 torch 命名空间下的 jit 命名空间
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义静态函数，用于判断节点是否可以融合
static bool canFuseNode(const Node* node) {
  // 根据节点的操作种类进行判断
  switch (node->kind()) {
    // 下面的操作种类表示可以进行节点融合
    case aten::conv2d:
    case aten::_convolution:
    case aten::batch_norm:
    case aten::layer_norm:
    case aten::add:
    case aten::mul:
    case aten::tanh:
    case aten::relu:
    case aten::elu:
    case aten::sigmoid:
    case aten::gelu:
    case aten::sqrt:
    case aten::abs:
    case aten::square:
    case aten::hardtanh:
    case aten::relu6:
    case aten::softmax:
    case aten::max_pool2d:
    case aten::avg_pool2d:
    case aten::matmul:
    case aten::mm:
    case aten::linear:
    case aten::addmm:
      // 如果节点操作属于上述任意一种，返回 true 表示可以融合
      return true;

    // 默认情况下，节点操作不能融合，返回 false
    default:
      return false;
  }
}

// 匿名命名空间，用于封装注册接口类
namespace {
class RegisterInterface {
 public:
  RegisterInterface() {
    // 在构造函数中注册节点融合函数 canFuseNode
    RegisterProfilingNode(canFuseNode);
  }
};

// 静态实例化 RegisterInterface，用于自动注册
static RegisterInterface register_interface_;
} // namespace

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```