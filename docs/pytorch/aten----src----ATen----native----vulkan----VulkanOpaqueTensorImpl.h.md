# `.\pytorch\aten\src\ATen\native\vulkan\VulkanOpaqueTensorImpl.h`

```
#pragma once
// 使用 #pragma once 防止头文件被多次包含

#include <ATen/OpaqueTensorImpl.h>
// 引入 ATen 库中的 OpaqueTensorImpl.h 头文件

namespace at {
// 定义 at 命名空间

// VulkanOpaqueTensorImpl 类继承自 OpaqueTensorImpl，用于在 Vulkan 后端上运行 TorchScript 模型
// 区别在于伪造了 strides()、stride() 和 is_contiguous() 方法，因为 Vulkan 不支持 strides，但计划未来支持
template <typename OpaqueHandle>
struct VulkanOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  VulkanOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes,
            false),
        strides_(strides.vec()) {}
  // 构造函数，初始化 VulkanOpaqueTensorImpl 对象，设置 key_set、data_type、device、opaque_handle、sizes 和 strides

  // 重载 strides_custom 方法，返回存储的自定义 strides
  IntArrayRef strides_custom() const override {
    return strides_;
  }

  // 重载 sym_strides_custom 方法，返回从 strides 转换得到的 SymIntArrayRef
  SymIntArrayRef sym_strides_custom() const override {
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  // 重载 is_contiguous_custom 方法，始终返回 true，因为 Vulkan 不涉及内存布局问题
  bool is_contiguous_custom(c10::MemoryFormat memory_format) const override {
    (void)memory_format;  // 防止编译器警告未使用的参数
    return true;
  }

 private:
  // 重载 tensorimpl_type_name 方法，返回类型名字串 "VulkanOpaqueTensorImpl"
  const char* tensorimpl_type_name() const override {
    return "VulkanOpaqueTensorImpl";
  }

  // TODO: storing strides separately is unnecessary, the base TensorImpl
  // has space for them
  // TODO 注释：存储 strides 是不必要的，因为基类 TensorImpl 已经为它们预留了空间
  SmallVector<int64_t, 5> strides_;  // 存储 strides 的容器，使用 SmallVector 类型
};

} // namespace at
// 结束 at 命名空间
```