# `.\pytorch\aten\src\ATen\native\metal\MetalTensorImpl.h`

```py
#ifndef MetalTensorImpl_h
#define MetalTensorImpl_h

#include <ATen/OpaqueTensorImpl.h>  // 引入 ATen 库中的 OpaqueTensorImpl 头文件
#include <ATen/WrapDimUtils.h>  // 引入 ATen 库中的 WrapDimUtils 头文件
#import <ATen/native/metal/MetalTensorImplStorage.h>  // 引入 MetalTensorImplStorage 头文件
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>  // 引入 MPSImageWrapper 头文件

namespace at {
template <typename OpaqueHandle>
struct TORCH_API MetalTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  MetalTensorImpl(
      at::DispatchKeySet key_set,  // 初始化 MetalTensorImpl 的 dispatch key 集合
      const caffe2::TypeMeta& data_type,  // 初始化数据类型的元数据
      c10::Device device,  // 初始化张量的设备类型
      OpaqueHandle opaque_handle,  // 初始化不透明句柄
      c10::IntArrayRef sizes,  // 初始化张量的尺寸
      c10::IntArrayRef strides)  // 初始化张量的步幅
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes),  // 调用父类 OpaqueTensorImpl 的构造函数进行初始化
        strides_(strides.vec()) {  // 将步幅数据转存到成员变量 strides_

  }

  // TODO: manually storing strides here is dumb
  // 注释: 这里手动存储步幅数据的做法不是最佳实践

  IntArrayRef strides_custom() const override {  // 自定义函数，返回张量的步幅数据引用
    return strides_;
  }

  c10::SymIntArrayRef sym_strides_custom() const override {  // 返回张量步幅数据的符号化引用
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  bool is_contiguous_custom(c10::MemoryFormat memory_format) const override {  // 自定义函数，检查张量是否是连续的
    return true;  // 总是返回 true，暗示张量是连续的
  }

 private:
  const char* tensorimpl_type_name() const override {  // 返回 MetalTensorImpl 类型名称的函数
    return "MetalTensorImpl";  // 返回 MetalTensorImpl
  }

  SmallVector<int64_t, 5> strides_;  // 成员变量，存储张量的步幅数据
};
} // namespace at

#endif /* MetalTensorImpl_h*/
```