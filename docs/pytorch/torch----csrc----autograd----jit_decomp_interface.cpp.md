# `.\pytorch\torch\csrc\autograd\jit_decomp_interface.cpp`

```py
# 包含 Torch 自动求导模块的 JIT 分解接口头文件
#include <torch/csrc/autograd/jit_decomp_interface.h>

# 定义 Torch 自动求导命名空间
namespace torch {
namespace autograd {
namespace impl {

# 匿名命名空间，用于保存 JIT 分解接口的实现指针
namespace {
    # JIT 分解接口的实现指针初始化为空指针
    JitDecompInterface* impl = nullptr;
}

# 设置 JIT 分解接口的实现指针
void setJitDecompImpl(JitDecompInterface* impl_) {
    impl = impl_;
}

# 获取 JIT 分解接口的实现指针
JitDecompInterface* getJitDecompImpl() {
    return impl;
}

} // namespace impl
} // namespace autograd
} // namespace torch
```