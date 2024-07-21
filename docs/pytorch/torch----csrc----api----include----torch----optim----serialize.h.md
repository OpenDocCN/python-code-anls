# `.\pytorch\torch\csrc\api\include\torch\optim\serialize.h`

```
#pragma once
// 预处理指令：确保此头文件仅被编译一次

#include <c10/util/irange.h>
// 引入C10库中的irange工具，用于处理范围

#include <torch/optim/optimizer.h>
// 引入PyTorch优化器的头文件

#include <torch/serialize/archive.h>
// 引入PyTorch序列化存档的头文件

#include <torch/types.h>
// 引入PyTorch类型定义的头文件

#include <cstddef>
// 引入C标准库的stddef头文件，包含了NULL和size_t的定义

#include <cstdint>
// 引入C标准库的cstdint头文件，包含了整数类型的定义

#include <deque>
// 引入C++标准库的deque头文件，双端队列的定义

#include <string>
// 引入C++标准库的string头文件，字符串的定义

#include <vector>
// 引入C++标准库的vector头文件，向量的定义

namespace torch {
namespace optim {
namespace detail {

// 定义命名空间：torch::optim::detail，包含了下面的实用函数

// Utility function to save state
// 用于保存状态的实用函数
template <typename DerivedOptimizerParamState>
void serialize(
    serialize::OutputArchive& archive,
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
        state) {
  // 序列化函数，将优化器参数状态保存到输出存档中
  for (const auto& item : state) {
    // 遍历状态映射中的每个条目
    serialize::OutputArchive param_state_archive(archive.compilation_unit());
    // 创建输出存档以保存参数状态
    std::string tensorimpl_key =
        std::to_string(reinterpret_cast<size_t>(item.first));
    // 获取张量实现的键，将指针地址转换为字符串
    const DerivedOptimizerParamState& curr_state =
        static_cast<const DerivedOptimizerParamState&>(*(item.second.get()));
    // 获取当前状态，并进行类型转换
    curr_state.serialize(param_state_archive);
    // 序列化当前状态到参数状态存档中
    archive.write(tensorimpl_key, param_state_archive);
    // 在主存档中写入张量实现的键和参数状态存档
  }
}

// Utility function to load state
// 用于加载状态的实用函数
template <typename DerivedOptimizerParamState>
void serialize(
    serialize::InputArchive& archive,
    ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& state) {
  // 反序列化函数，从输入存档中加载优化器参数状态
  std::vector<std::string> tensorimpl_keys = archive.keys();
  // 获取存档中所有键的列表
  for (const std::string& tensorimpl_key : tensorimpl_keys) {
    // 遍历张量实现键列表
    serialize::InputArchive param_state_archive;
    // 创建输入存档以加载参数状态
    archive.read(tensorimpl_key, param_state_archive);
    // 从存档中读取参数状态存档
    DerivedOptimizerParamState param_state;
    // 创建派生优化器参数状态对象
    param_state.serialize(param_state_archive);
    // 反序列化参数状态存档到参数状态对象中
    state[reinterpret_cast<void*>(std::stoull(tensorimpl_key))] =
        std::make_unique<DerivedOptimizerParamState>(param_state);
    // 将参数状态添加到状态映射中，键为张量实现的地址
  }
}

// Utility function to save param_groups
// 用于保存参数组的实用函数
template <typename DerivedOptimizerParamOptions>
void serialize(
    serialize::OutputArchive& archive,
    const std::vector<OptimizerParamGroup>& param_groups) {
  // 序列化函数，将优化器参数组保存到输出存档中
  archive.write(
      "param_groups/size",
      torch::tensor(static_cast<int64_t>(param_groups.size())));
  // 在存档中写入参数组的数量
  for (const auto i : c10::irange(param_groups.size())) {
    // 遍历参数组列表
    serialize::OutputArchive param_group_archive(archive.compilation_unit());
    // 创建输出存档以保存参数组
    std::vector<Tensor> params = param_groups[i].params();
    // 获取当前参数组的张量列表
    param_group_archive.write(
        "params/size", torch::tensor(static_cast<int64_t>(params.size())));
    // 在参数组存档中写入张量列表的大小
    for (const auto index : c10::irange(params.size())) {
      // 遍历张量列表
      param_group_archive.write(
          "params/" + std::to_string(index),
          IValue(std::to_string(
              reinterpret_cast<size_t>(params[index].unsafeGetTensorImpl()))));
      // 在参数组存档中写入每个张量的地址作为字符串
    }
    const DerivedOptimizerParamOptions& param_group_options =
        static_cast<const DerivedOptimizerParamOptions&>(
            param_groups[i].options());
    // 获取当前参数组的派生优化器参数选项对象
    serialize::OutputArchive param_group_options_archive(
        param_group_archive.compilation_unit());
    // 创建输出存档以保存参数组选项
    param_group_options.serialize(param_group_options_archive);
    // 序列化参数组选项到参数组选项存档中
    param_group_archive.write("options", param_group_options_archive);
    // 在参数组存档中写入参数组选项存档
    archive.write("param_groups/" + std::to_string(i), param_group_archive);
    // 在主存档中写入参数组存档
  }
}
// 结束参数组序列化函数的定义

} // namespace detail
} // namespace optim
} // namespace torch
// 结束命名空间torch::optim::detail及其子命名空间的定义
// Utility function to load param_groups
// We take as input vector of pair of string and unique_ptr to optimizer options
// so that we can retain the state for each param by using the old tensor impl
// keys (saved during serialization) and map the new tensor impl keys to the
// correct state for each param
template <typename DerivedOptimizerParamOptions>
void serialize(
    serialize::InputArchive& archive,
    std::vector<
        std::pair<std::vector<std::string>, std::unique_ptr<OptimizerOptions>>>&
        param_groups) {
  // Read the size of param_groups tensor from the archive
  torch::Tensor param_groups_size_tensor;
  archive.read("param_groups/size", param_groups_size_tensor);
  const int64_t param_groups_size = param_groups_size_tensor.item<int64_t>();
  // Iterate over the number of param_groups
  for (const auto i : c10::irange(param_groups_size)) {
    // Create an archive for the current param_group
    serialize::InputArchive param_group_archive;
    archive.read("param_groups/" + std::to_string(i), param_group_archive);
    // Read the size of params tensor for the current param_group
    torch::Tensor size_tensor;
    param_group_archive.read("params/size", size_tensor);
    const int64_t size = size_tensor.item<int64_t>();
    std::vector<std::string> params;
    // Iterate over the number of params in the current param_group
    for (const auto index : c10::irange(size)) {
      // Read each param name as an IValue and convert to string
      IValue ivalue;
      param_group_archive.read("params/" + std::to_string(index), ivalue);
      std::string element = ivalue.toStringRef();
      params.emplace_back(element);
    }
    // Create an archive for options related to the current param_group
    serialize::InputArchive param_group_options_archive;
    param_group_archive.read("options", param_group_options_archive);
    // Deserialize the options for the current param_group
    DerivedOptimizerParamOptions param_group_options(0);
    param_group_options.serialize(param_group_options_archive);
    // Add the pair of params and options to param_groups vector
    param_groups.emplace_back(std::make_pair(
        params,
        std::make_unique<DerivedOptimizerParamOptions>(param_group_options)));
  }
}
} // namespace detail

// Note: These functions are all called `serialize()` so they can be called
// inside a template where the archive type is a template type and can thus be
// passed such that the appropriate overload is selected.

/// Utility function to save a value of `int64_t` type.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const int64_t& value);

/// Utility function to load a value of `int64_t` type.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    int64_t& value);

/// Utility function to save a vector of step buffers.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps);

/// Utility function to load a vector of step buffers.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<int64_t>& steps);

// Utility function to save state and param_groups
template <
    typename DerivedOptimizerParamState,
    typename DerivedOptimizerParamOptions>
// Serialize function for saving optimizer state using the given archive
void serialize(serialize::OutputArchive& archive, const Optimizer& optimizer) {
  // Write the PyTorch version into the archive
  archive.write("pytorch_version", IValue("1.5.0"));

  // Create a new OutputArchive for saving optimizer state
  serialize::OutputArchive state_archive(archive.compilation_unit());
  
  // Serialize the optimizer state using specialized serialization function
  detail::serialize<DerivedOptimizerParamState>(
      state_archive, optimizer.state());
  
  // Write the serialized state into the main archive
  archive.write("state", state_archive);

  // Create a new OutputArchive for saving optimizer parameter groups
  serialize::OutputArchive param_groups_archive(archive.compilation_unit());
  
  // Serialize the optimizer parameter groups using specialized serialization function
  detail::serialize<DerivedOptimizerParamOptions>(
      param_groups_archive, optimizer.param_groups());
  
  // Write the serialized parameter groups into the main archive
  archive.write("param_groups", param_groups_archive);
}

// Utility function to load optimizer state and parameter groups from the archive and update optimizer
template <
    typename DerivedOptimizerParamState,
    typename DerivedOptimizerParamOptions>
void serialize(serialize::InputArchive& archive, Optimizer& optimizer) {
  // Read and verify the PyTorch version from the archive
  IValue pytorch_version;
  archive.read("pytorch_version", pytorch_version);
  TORCH_INTERNAL_ASSERT(pytorch_version.toStringRef() == "1.5.0");

  // Initialize InputArchive for loading optimizer state
  serialize::InputArchive state_archive;
  archive.read("state", state_archive);
  
  // Data structure for storing loaded optimizer state
  ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>> saved_state;
  
  // Deserialize optimizer state using specialized function
  detail::serialize<DerivedOptimizerParamState>(state_archive, saved_state);

  // Initialize InputArchive for loading optimizer parameter groups
  serialize::InputArchive param_groups_archive;
  archive.read("param_groups", param_groups_archive);
  
  // Data structure for storing loaded optimizer parameter groups
  std::vector<
      std::pair<std::vector<std::string>, std::unique_ptr<OptimizerOptions>>>
      saved_param_groups;
  
  // Deserialize optimizer parameter groups using specialized function
  detail::serialize<DerivedOptimizerParamOptions>(
      param_groups_archive, saved_param_groups);

  // Update optimizer state and options based on the loaded data
  // Verify consistency of loaded parameter groups with the optimizer's current groups
  TORCH_CHECK(
      saved_param_groups.size() == optimizer.param_groups().size(),
      "loaded state dict has a different number of parameter groups");
  
  // Iterate over each parameter group and update its state and options
  for (const auto i : c10::irange(saved_param_groups.size())) {
    std::vector<std::string> param_group_old_keys = saved_param_groups[i].first;
    std::vector<Tensor> params = optimizer.param_groups()[i].params();
    
    // Verify consistency of parameter group sizes
    TORCH_CHECK(
        param_group_old_keys.size() == params.size(),
        "loaded state dict contains a parameter group that has a different size than the optimizer's parameter group");

    // Update state for each parameter in the group based on loaded data
    for (const auto idx : c10::irange(params.size())) {
      auto param_group_old_key =
          reinterpret_cast<void*>(std::stoull(param_group_old_keys[idx]));
      
      // Check if state exists for the parameter and update optimizer's state
      if (saved_state.find(param_group_old_key) != saved_state.end()) {
        optimizer.state()[params[idx].unsafeGetTensorImpl()] =
            std::move(saved_state[param_group_old_key]);
      }
    }

    // Update optimizer's options for the current parameter group
    auto& saved_options = reinterpret_cast<DerivedOptimizerParamOptions&>(
        *saved_param_groups[i].second);
    auto& current_options = reinterpret_cast<DerivedOptimizerParamOptions&>(
        optimizer.param_groups()[i].options());
    current_options = saved_options;
  }
}

/// Utility function to save a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const BufferContainer& buffers) {
  // Serialize a vector of buffers into the archive under a specified key
  archive.write(key, buffers);
}
    const BufferContainer& buffers) {
```  

// 定义一个函数，接受一个常量引用类型的 BufferContainer 参数
archive.write(
    key + "/size", torch::tensor(static_cast<int64_t>(buffers.size())));
```  

// 使用 archive 对象写入键为 key+"/size" 的条目，其值是 buffers 容器的大小转换成整数的张量
for (const auto index : c10::irange(buffers.size())) {
```  

// 遍历 buffers 容器中的每个索引，index 变量会依次取到容器中的每个索引值
archive.write(
    key + "/" + std::to_string(index), buffers[index], /*is_buffer=*/true);
```  

// 使用 archive 对象写入键为 key+"/"+当前索引值 的条目，其值为 buffers 容器中对应索引的数据，并标记其为缓冲区数据
}

/// Utility function to load a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    BufferContainer& buffers) {
  // 清空 buffers 容器
  buffers.clear();
  // 从存档中读取 key + "/size" 对应的 size_tensor
  torch::Tensor size_tensor;
  archive.read(key + "/size", size_tensor);
  // 将 size_tensor 转换为 int64_t 类型的 size
  const size_t size = size_tensor.item<int64_t>();
  // 根据 size 循环读取 buffers
  for (const auto index : c10::irange(size)) {
    // 向 buffers 中添加新元素
    buffers.emplace_back();
    // 从存档中读取 key + "/" + std::to_string(index) 对应的数据，写入 buffers.back()
    archive.read(
        key + "/" + std::to_string(index), buffers.back(), /*is_buffer=*/true);
  }
}

template <typename T>
c10::List<T> deque_to_list(const std::deque<T>& dq) {
  // 创建一个 c10::List<T> 类型的 list
  c10::List<T> list;
  // 保留足够的空间以容纳 dq 的所有元素
  list.reserve(dq.size());
  // 将 dq 中的元素逐个添加到 list 中
  for (const auto& e : dq) {
    list.emplace_back(e);
  }
  // 返回转换后的 list
  return list;
}

template <typename T>
std::deque<T> list_to_deque(const c10::List<T>& list) {
  // 创建一个 std::deque<T> 类型的 dq
  std::deque<T> dq;
  // 将 list 中的元素逐个添加到 dq 中
  for (const auto& e : list) {
    dq.emplace_back(e);
  }
  // 返回转换后的 dq
  return dq;
}

#define _TORCH_OPTIM_SERIALIZE(name) \
  torch::optim::serialize(archive, #name, self.name)

#define _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(OptimizerName)               \
  torch::optim::serialize<OptimizerName##ParamState, OptimizerName##Options>( \
      archive, self)

#define _TORCH_OPTIM_SERIALIZE_TORCH_ARG(name)           \
  {                                                      \
    auto ivalue = torch::IValue(name());                 \
    /* 若 name 是未定义的张量，则不进行序列化*/ \
    if (!(ivalue.isTensor() &&                           \
          ivalue.unsafeToTensorImpl() ==                 \
              at::UndefinedTensorImpl::singleton())) {   \
      archive.write(#name, ivalue);                      \
    }                                                    \
  }

#define _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(name)           \
  {                                                            \
    c10::IValue ivalue = torch::IValue(deque_to_list(name())); \
    archive.write(#name, ivalue);                              \
  }

#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(T, name)                   \
  {                                                                   \
    c10::IValue ivalue;                                               \
    // 尝试从存档中读取 name 对应的值到 ivalue 中，并判断是否成功读取
    bool exists = archive.try_read(#name, ivalue);                    \
    if (exists) {                                                     \
      // 若成功读取，则将 ivalue 转换为类型 T 并赋值给 name
      name(ivalue.to<T>());                                           \
    } else {                                                          \
      // 若未成功读取，则断言 T 是否为 torch::Tensor 类型的子类
      bool is_tensor_type = std::is_base_of<torch::Tensor, T>::value; \
      TORCH_INTERNAL_ASSERT(is_tensor_type);                          \
    }                                                                 \
  }

#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(T, name) \
  {                                                          \
    c10::IValue ivalue;                                      \
    // 尝试从存档中读取 name 对应的值到 ivalue 中，并判断是否成功读取
    bool exists = archive.try_read(#name, ivalue);           \
    // 对于成功读取的情况，将 ivalue 转换为类型 T 并赋值给 name
    if (exists) {                                            \
    // 如果 exists 为真，则执行以下操作
    if (exists) {
      // 调用 name 函数，并传递 ivalue.toOptional<T>() 的结果作为参数
      name(ivalue.toOptional<T>());
    }
  }
#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(T, name) \  // 定义宏 _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE，用于反序列化 Torch 参数中的双端队列
  {                                                       \  // 宏定义开始
    c10::IValue ivalue;                                   \  // 声明 c10::IValue 对象 ivalue，用于存储反序列化后的值
    archive.read(#name, ivalue);                          \  // 使用传入的参数名称 name，从存档 archive 中读取对应的值，并存储到 ivalue 中
    auto list = ivalue.to<c10::List<T::value_type>>();    \  // 将 ivalue 转换为 c10::List<T::value_type> 类型的列表 list
    name(list_to_deque(list));                            \  // 将 list 转换为双端队列，并将其传递给参数 name 所指示的函数或变量
  }                                                       \  // 宏定义结束

} // namespace optim
} // namespace torch
```