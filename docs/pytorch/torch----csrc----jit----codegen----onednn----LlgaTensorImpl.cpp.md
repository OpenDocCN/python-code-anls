# `.\pytorch\torch\csrc\jit\codegen\onednn\LlgaTensorImpl.cpp`

```
// 包含 ATen 配置文件
#include <ATen/Config.h>

// 如果 AT_MKLDNN_ENABLED 宏定义为真，则包含以下头文件
#if AT_MKLDNN_ENABLED()
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义静态函数 pytorch_default_allocator，用于提供默认的内存分配器
static void* pytorch_default_allocator(size_t size, size_t alignment) {
  // 使用 c10::GetCPUAllocator 获取当前 CPU 分配器的实例
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  // 调用分配器的 raw_allocate 方法进行内存分配
  return c10_allocator->raw_allocate(size);
}

// 定义静态函数 pytorch_default_deallocator，用于提供默认的内存释放器
static void pytorch_default_deallocator(void* buf) {
  // 使用 c10::GetCPUAllocator 获取当前 CPU 分配器的实例
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  // 调用分配器的 raw_deallocate 方法进行内存释放
  c10_allocator->raw_deallocate(buf);
}

// 实现 Engine 类的成员函数 getEngine，返回 DNNL 引擎对象的实例
dnnl::engine& Engine::getEngine() {
  // 定义静态变量 alloc，使用 pytorch_default_allocator 和 pytorch_default_deallocator 初始化 dnnl::graph::allocator
  static dnnl::graph::allocator alloc{
      pytorch_default_allocator, pytorch_default_deallocator};
  // 定义静态变量 cpu_engine，使用 make_engine_with_allocator 创建 DNNL 引擎对象
  static dnnl::engine cpu_engine = dnnl::graph::make_engine_with_allocator(
      dnnl::engine::kind::cpu, /* device_id = */ 0, alloc);
  // 返回静态变量 cpu_engine 的引用
  return cpu_engine;
}

// 实现 Stream 类的成员函数 getStream，返回 DNNL 流对象的实例
dnnl::stream& Stream::getStream() {
  // 定义静态变量 cpu_stream，使用 Engine::getEngine 创建 DNNL 流对象
  static dnnl::stream cpu_stream{Engine::getEngine()};
  // 返回静态变量 cpu_stream 的引用
  return cpu_stream;
}

// 实现 LlgaTensorImpl 类的构造函数，初始化 LlgaTensorImpl 对象
LlgaTensorImpl::LlgaTensorImpl(
    at::Storage&& storage,
    const caffe2::TypeMeta& data_type,
    const LlgaTensorDesc& desc)
    : at::TensorImpl(
          std::move(storage),
          c10::DispatchKeySet(c10::DispatchKey::MkldnnCPU),
          data_type),
      desc_(desc) {
  // 设置张量的大小和步长
  set_sizes_and_strides(desc.sizes(), desc.strides());
  // 刷新张量的元素数量
  refresh_numel();
}

// 定义函数 llga_to_aten_tensor，将 LlgaTensorImpl 转换为 ATen 张量
at::Tensor LlgaTensorImpl::llga_to_aten_tensor(LlgaTensorImpl* llgaImpl) {
  // 创建 ATen 张量，使用 make_tensor 方法
  auto aten_tensor = at::detail::make_tensor<TensorImpl>(
      std::move(llgaImpl->storage_),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      llgaImpl->data_type_);
  // 获取张量实现的指针
  auto impl = aten_tensor.unsafeGetTensorImpl();
  // 设置张量的存储偏移量
  impl->set_storage_offset(llgaImpl->storage_offset_);
  // 设置张量的大小和步长
  impl->set_sizes_and_strides(llgaImpl->sizes(), llgaImpl->strides());
  // 返回创建的 ATen 张量
  return aten_tensor;
}

// 定义函数 empty_llga，创建一个空的 LlgaTensorImpl 对象
at::Tensor empty_llga(
    const LlgaTensorDesc& desc,
    // 获取存储描述对象的字节大小
  auto nbytes = desc.storage_size();

  // 获取CPU分配器
  auto allocator = at::GetCPUAllocator();

  // 创建StorageImpl对象，使用字节大小作为参数
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,  // 分配的字节数
      allocator->allocate(nbytes),  // 分配内存并返回指针
      allocator,
      /*resizable=*/false);  // 不可调整大小

  // 创建LlgaTensorImpl类型的Tensor对象，使用移动语义传递storage_impl
  return at::detail::make_tensor<LlgaTensorImpl>(
      std::move(storage_impl),  // 移动语义传递storage_impl对象
      options.dtype(),  // 使用给定的数据类型
      desc);  // 使用给定的描述对象
}

// 结束命名空间 torch
} // namespace torch

// 结束命名空间 jit
} // namespace jit

// 结束命名空间 fuser
} // namespace fuser

// 结束命名空间 onednn
} // namespace onednn

// 如果 AT_MKLDNN_ENABLED 宏已定义，则结束文件引用
#endif // AT_MKLDNN_ENABLED()
```