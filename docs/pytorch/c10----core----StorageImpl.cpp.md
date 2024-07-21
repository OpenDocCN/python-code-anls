# `.\pytorch\c10\core\StorageImpl.cpp`

```py
// 包含 C10 库中的头文件 StorageImpl.h 和 flat_hash_map.h
#include <c10/core/StorageImpl.h>
#include <c10/util/flat_hash_map.h>

// c10 命名空间
namespace c10 {

// 保存自定义 StorageImpl 创建函数指针的数组
C10_API std::array<StorageImplCreateHelper, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    StorageImplCreate;

// 一个允许的设备类型白名单，目前只包括 PrivateUse1
inline ska::flat_hash_set<c10::DeviceType>& GetBackendMetaAllowlist() {
  // 静态变量，存储设备类型白名单的哈希集合
  static ska::flat_hash_set<c10::DeviceType> DeviceTypeAllowList{
      DeviceType::PrivateUse1};
  return DeviceTypeAllowList;
}

// 抛出空数据指针错误的函数
void throwNullDataPtrError() {
  TORCH_CHECK(
      false,
      "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor). "
      "If you're using torch.compile/export/fx, it is likely that we are erroneously "
      "tracing into a custom kernel. To fix this, please wrap the custom kernel into "
      "an opaque custom op. Please see the following for details: "
      "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html");
}

// [FakeTensor.data_ptr 废弃说明]
// 当前：
// - FakeTensor.data_ptr 在 torch.compile 中报错。
// - 否则 FakeTensor.data_ptr 引发以下废弃警告。
// - 该废弃警告目前仅适用于 FakeTensor。
//   未来可以考虑扩展到更多封装的 Tensor 子类。
void warnDeprecatedDataPtr() {
  TORCH_WARN_ONCE(
      "Accessing the data pointer of FakeTensor is deprecated and will error in "
      "PyTorch 2.5. This is almost definitely a bug in your code and will "
      "cause undefined behavior with subsystems like torch.compile. "
      "Please wrap calls to tensor.data_ptr() in an opaque custom op; "
      "If all else fails, you can guard accesses to tensor.data_ptr() on "
      "isinstance(tensor, FakeTensor).");
}

// 设置 StorageImpl 创建函数指针的函数
void SetStorageImplCreate(DeviceType t, StorageImplCreateHelper fptr) {
  // 获取设备类型白名单的引用
  const auto& DeviceTypeAllowlist = GetBackendMetaAllowlist();
  // 检查是否允许注册该设备类型的 StorageImpl 创建方法
  TORCH_CHECK(
      DeviceTypeAllowlist.find(t) != DeviceTypeAllowlist.end(),
      "It is only allowed to register the storageImpl create method ",
      "for PrivateUse1. ",
      "If you have related storageImpl requirements, ",
      "please expand the allowlist");
  // 注册函数指针
  int device_type = static_cast<int>(t);
  TORCH_CHECK(
      StorageImplCreate[device_type] == nullptr,
      "The StorageImplCreate function pointer for ",
      t,
      " has been registered.");
  StorageImplCreate[device_type] = fptr;
}

// 获取指定设备类型的 StorageImpl 创建函数指针
StorageImplCreateHelper GetStorageImplCreate(DeviceType t) {
  int device_type = static_cast<int>(t);
  return StorageImplCreate[device_type];
}

// 创建 StorageImpl 对象的函数
c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable,
    // 如果有自定义的 StorageImpl 构造函数与给定设备相关联，则此指针将非空
    c10::StorageImplCreateHelper fptr = nullptr;
    
    // 如果传入了设备参数，则检查是否有特定设备类型的 StorageImpl 构造函数
    if (device_opt.has_value()) {
        // 获取与设备类型相关联的 StorageImpl 构造函数指针
        fptr = c10::GetStorageImplCreate(device_opt.value().type());
    }
    
    // 如果找到了匹配的构造函数指针，则使用它来创建 StorageImpl 对象
    if (fptr != nullptr) {
        return fptr(
            use_byte_size,
            std::move(size_bytes),
            std::move(data_ptr),
            allocator,
            resizable);
    }
    
    // 如果没有特定设备的构造函数或者未传入有效的数据指针，则创建一个默认的 StorageImpl 对象
    // 使用给定的参数创建一个 c10::StorageImpl 对象
    if (data_ptr != nullptr) {
        return c10::make_intrusive<c10::StorageImpl>(
            use_byte_size,
            std::move(size_bytes),
            std::move(data_ptr),
            allocator,
            resizable);
    } else {
        return c10::make_intrusive<c10::StorageImpl>(
            use_byte_size, std::move(size_bytes), allocator, resizable);
    }
}

// 结束命名空间 c10
} // namespace c10
```