# `.\pytorch\torch\csrc\jit\backends\xnnpack\compiler\xnn_compiler.h`

```
// 包含 XNNPack 执行器的头文件
#include <caffe2/torch/csrc/jit/backends/xnnpack/executor/xnn_executor.h>
// 包含 XNNPack 库的头文件
#include <xnnpack.h>
// 包含内存管理的头文件
#include <memory>
// 包含向量操作的头文件
#include <vector>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {
// XNNPack 后端命名空间
namespace xnnpack {
// XNNPack 代理命名空间
namespace delegate {

// XNNCompiler 类定义
class XNNCompiler {
 public:
  // 编译模型的静态方法
  // 接受 Flatbuffer 序列化的 XNNPack 模型和字节大小作为参数
  // 重建 xnn 子图，并返回一个执行器对象，该对象包含 xnn 运行时对象
  // 可以用于设置输入和运行推断
  static void compileModel(
      const void* buffer_pointer,
      size_t num_bytes,
      XNNExecutor* executor);
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
```