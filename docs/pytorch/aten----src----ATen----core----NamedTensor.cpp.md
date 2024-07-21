# `.\pytorch\aten\src\ATen\core\NamedTensor.cpp`

```
// 定义预处理指令，禁用 Torch 张量操作符的断言
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库中的命名张量相关头文件
#include <ATen/core/NamedTensor.h>

// 包含 ATen 库中的张量基础定义头文件
#include <ATen/core/TensorBase.h>

// ATen 命名空间
namespace at {

// 线程局部变量，控制命名模式是否启用，默认为启用
thread_local bool NamesMode_enabled = true;

// 查询当前命名模式是否启用
bool NamesMode::is_enabled() {
  return NamesMode_enabled;
}

// 设置命名模式是否启用
void NamesMode::set_enabled(bool enabled) {
  NamesMode_enabled = enabled;
  // 设置不包含 Named DispatchKey 如果禁用命名模式
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::Named, !enabled);
}

// 在给定张量上原地设置名称
const TensorBase& internal_set_names_inplace(const TensorBase& tensor, optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names, /*validate_names=*/true);
  return tensor;
}

// 在给定张量上原地设置名称（使用移动语义）
const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::vector<Dimname>&& names, bool validate_names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), std::move(names), validate_names);
  return tensor;
}

// 生成指定长度的默认名称列表
DimnameList default_names(size_t len) {
  static std::vector<Dimname> all_unnamed(kMaxNamedTensorDim, Dimname::wildcard());
  // 断言长度不超过最大命名张量维度
  TORCH_INTERNAL_ASSERT(
      len <= kMaxNamedTensorDim,
      "Only tensors with up to ", kMaxNamedTensorDim, " are supported.");
  return DimnameList(&all_unnamed.front(), len);
}

// 检查名称列表中是否有重复名称
static void check_unique_names(DimnameList names) {
  // 策略：比较每个元素与其后续元素
  // 尽管复杂度为 O(N^2)，实际上 N 很小（不超过 25）
  for (auto it = names.begin(); it != names.end(); ++it) {
    if (it->isWildcard()) continue;
    auto dup = std::find(it + 1, names.end(), *it);
    while (dup != names.end()) {
      TORCH_CHECK(false,
          "Cannot construct a tensor with duplicate names. Got names: ",
          names, ".");
    }
  }
}

// 检查给定张量是否适合使用指定名称列表的名称
void check_names_valid_for(const TensorBase& tensor, DimnameList names) {
  return impl::check_names_valid_for(tensor.unsafeGetTensorImpl(), names);
}

// 检查给定张量维度数和名称列表是否匹配，并检查名称的唯一性
void check_names_valid_for(size_t tensor_dim, DimnameList names) {
  TORCH_CHECK(
      tensor_dim <= kMaxNamedTensorDim,
      "Named tensors only support up to ", kMaxNamedTensorDim, " dims: "
      "Attempted to create a tensor with dim ", tensor_dim, " with names ", names);
  TORCH_CHECK(tensor_dim == names.size(),
      "Number of names (", names.size(), ") and "
      "number of dimensions in tensor (", tensor_dim, ") ",
      "do not match. Attempted to create a tensor with names ", names);
  check_unique_names(names);
}

// ATen 命名空间中的实现细节
namespace impl {

// 获取给定张量实现的命名张量元数据
static NamedTensorMeta* get_named_tensor_meta(TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<NamedTensorMeta*>(impl->named_tensor_meta());
}

// 获取给定张量实现的命名张量元数据（常量版本）
static const NamedTensorMeta* get_named_tensor_meta(const TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<const NamedTensorMeta*>(impl->named_tensor_meta());
}

// 检查给定张量实现是否适合使用指定名称列表的名称
void check_names_valid_for(TensorImpl* impl, DimnameList names) {
  check_names_valid_for(impl->dim(), names);
}
void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names, bool validate_names) {
    // 检查张量布局是否为Strided，命名张量仅支持Strided布局
    TORCH_CHECK(impl->layout() == Layout::Strided,
        "NYI: named tensors only support strided layout");
    // 检查设备类型是否为CPU、CUDA、XPU或私有后端类型，命名张量仅支持这些设备
    TORCH_CHECK(impl->device().is_cpu() || impl->device().is_cuda() || impl->device().is_xpu() || impl->device().is_privateuseone(),
        "NYI: named tensors only support CPU, CUDA, XPU or ", c10::get_privateuse1_backend(), " tensors.");
    // 如果没有传入names参数，设置张量的命名元数据为空，并返回
    if (!names) {
        impl->set_named_tensor_meta(nullptr);
        return;
    }
    // 如果需要验证命名是否有效，则调用函数检查命名是否合法
    if (validate_names) {
        check_names_valid_for(impl, *names);
    }
    // 在验证后执行以下操作！如果所有命名都是通配符，则设置命名元数据为空，并返回
    if (std::all_of(names->begin(), names->end(), [](const Dimname& n) { return n.isWildcard(); })) {
        impl->set_named_tensor_meta(nullptr);
        return;
    }
    // 获取当前张量的命名元数据指针
    auto* meta = get_named_tensor_meta(impl);
    // 如果命名元数据为空，使用非通配符的命名创建新的命名元数据对象
    if (meta == nullptr) {
        impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, *names));
    } else {
        // 否则，更新现有的命名元数据对象的命名信息
        meta->set_names(NamedTensorMeta::HasNonWildcard, *names);
    }
}

void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names) {
    // 如果需要验证命名是否有效，则调用函数检查命名是否合法
    if (validate_names) {
        check_names_valid_for(impl, names);
    }
    // 在验证后执行以下操作！如果所有命名都是通配符，则设置命名元数据为空，并返回
    if (std::all_of(names.begin(), names.end(), [](const Dimname& n) { return n.isWildcard(); })) {
        impl->set_named_tensor_meta(nullptr);
        return;
    }
    // 获取当前张量的命名元数据指针
    auto* meta = get_named_tensor_meta(impl);
    // 如果命名元数据为空，使用非通配符的命名列表创建新的命名元数据对象
    if (meta == nullptr) {
        impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, std::move(names)));
    } else {
        // 否则，更新现有的命名元数据对象的命名信息
        meta->set_names(NamedTensorMeta::HasNonWildcard, std::move(names));
    }
}

// 获取张量的可选命名列表，如果不存在命名元数据则返回空
optional<DimnameList> get_opt_names(const TensorImpl* impl) {
    const auto* meta = get_named_tensor_meta(impl);
    if (meta == nullptr) {
        return nullopt;
    } else {
        // 返回命名元数据对象中的命名列表
        return meta->names();
    }
}

// 获取张量的命名列表，如果不存在命名元数据则返回默认的命名列表
DimnameList get_names(const TensorImpl* impl) {
    // 获取张量的可选命名列表
    auto maybe_names = get_opt_names(impl);
    // 如果可选命名列表存在，则返回其中的命名列表，否则返回默认维度数量的命名列表
    if (maybe_names) {
        return *maybe_names;
    }
    return default_names(impl->dim());
}

// 检查张量是否具有命名元数据并且命名模式已启用
bool has_names(const TensorImpl* impl) {
    return impl->has_named_tensor_meta() && NamesMode::is_enabled();
}
```