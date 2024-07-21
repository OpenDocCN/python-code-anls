# `.\pytorch\torch\csrc\inductor\resize_storage_bytes.cpp`

```py
// 引入 Torch 库的头文件
#include <torch/library.h>

// 引入功能性张量包装的头文件
#include <ATen/FunctionalTensorWrapper.h>
// 引入原生调整大小功能的头文件
#include <ATen/native/Resize.h>

// 如果编译时启用了 CUDA，引入 CUDA 调整大小的头文件
#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#endif

// 定义 torch 命名空间下的 inductor 命名空间
namespace torch {
namespace inductor {

// 使用 at 命名空间
using namespace at;

// 定义静态函数 resize_storage_bytes_，调整张量存储的字节数
static void resize_storage_bytes_(const Tensor& variable, SymInt new_size) {
  // 类似于 StorageMethods.cpp 中的 THPStorage_resize_，但可追踪
  // 检查张量存储的设备类型是否为 CUDA
  if (variable.storage().device_type() == at::kCUDA) {
    // 如果编译时启用了 CUDA 并且未启用 ROCm，调用 CUDA 版本的 resize_bytes_cuda 函数
#if defined(USE_CUDA) && !defined(USE_ROCM)
    at::native::resize_bytes_cuda(
        variable.storage().unsafeGetStorageImpl(), new_size.expect_int());
#else
    // 否则抛出错误，表明未编译 CUDA 支持
    TORCH_CHECK(false, "built without cuda");
#endif
  } else {
    // 如果不是 CUDA 设备，调用 CPU 版本的 resize_bytes_nocuda 函数
    at::native::resize_bytes_nocuda(variable.storage(), new_size);
  }
}

// 定义静态函数 resize_storage_bytes__functionalize，将调整大小功能化
static void resize_storage_bytes__functionalize(
    const Tensor& variable,
    SymInt new_size) {
  // 查找并调用 inductor 命名空间下的 resize_storage_bytes_ 操作的 schema
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("inductor::resize_storage_bytes_", "")
                       .typed<void(const Tensor&, SymInt)>();
  
  // 如果不是功能化张量，直接调用普通的 resize 操作
  if (!at::functionalization::impl::isFunctionalTensor(variable)) {
    // 功能化未激活时不做操作
    at::AutoDispatchSkipFunctionalize guard;
    op.call(variable, new_size);
    return;
  }
  
  // 如果是功能化张量，获取其内部张量的 mutable 操作
  auto functional_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(variable);
  {
    // 跳过功能化时执行操作
    at::AutoDispatchSkipFunctionalize guard;
    op.call(functional_impl->value(), new_size);
    return;
  }
}

// 定义 TORCH_LIBRARY_FRAGMENT，注册 resize_storage_bytes_ 操作到 torch 库的 inductor 分支中
TORCH_LIBRARY_FRAGMENT(inductor, m) {
  m.def(
      "resize_storage_bytes_(Tensor variable, SymInt new_size) -> ()",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, resize_storage_bytes_),
      {at::Tag::pt2_compliant_tag});
}

// 定义 TORCH_LIBRARY_IMPL，实现在 inductor 分支中的功能化版本的 resize_storage_bytes_ 操作
TORCH_LIBRARY_IMPL(inductor, Functionalize, m) {
  m.impl(
      "resize_storage_bytes_", TORCH_FN(resize_storage_bytes__functionalize));
}

} // namespace inductor
} // namespace torch
```