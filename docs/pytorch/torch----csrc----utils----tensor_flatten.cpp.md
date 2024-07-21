# `.\pytorch\torch\csrc\utils\tensor_flatten.cpp`

```py
#include <torch/csrc/utils/tensor_flatten.h>  // 包含 Torch 库中用于张量操作的头文件

#include <map>  // 包含标准库中的 map 容器头文件
#include <unordered_map>  // 包含标准库中的 unordered_map 容器头文件

namespace torch::utils {  // 定义 torch::utils 命名空间

using namespace at;  // 引入 at 命名空间的所有内容

std::vector<TensorGroup> take_tensors(  // 定义函数 take_tensors，接收张量列表和大小限制等参数，返回张量组的向量
    TensorList tensors,  // 张量列表参数
    size_t size_limit,  // 大小限制参数
    bool fine_grained) {  // 是否细粒度处理的布尔参数
  std::vector<TensorGroup> results;  // 存储结果的张量组向量
  // an overapproximation, but at least we won't have to copy stuff around
  results.reserve(tensors.size());  // 预留足够的空间以容纳所有张量

  std::map<int64_t, TensorGroup> groups;  // 使用 int64_t 类型作为键的映射，映射到张量组的 map 容器
  size_t cur_group_size = 0;  // 当前组的大小计数器

  for (const auto& tensor : tensors) {  // 遍历输入的每个张量
    size_t tensor_size = 0;  // 张量的大小计数器

    if (tensor.is_sparse()) {  // 如果张量是稀疏张量
      const auto& indices = tensor._indices();  // 获取稀疏张量的索引
      const auto& values = tensor._values();  // 获取稀疏张量的值
      tensor_size = indices.numel() * indices.element_size() +
          values.numel() * indices.element_size();  // 计算稀疏张量的总大小
    } else {  // 如果张量不是稀疏张量
      tensor_size = tensor.numel() * tensor.element_size();  // 计算张量的总大小
    }

    auto& type_group = groups[static_cast<int64_t>(type_id(tensor))];  // 根据张量的类型 ID 获取对应的张量组
    type_group.tensors.push_back(tensor);  // 将当前张量添加到相应类型组的张量列表中

    if (fine_grained) {  // 如果需要细粒度处理
      cur_group_size += tensor_size;  // 累加当前组的大小
      if (cur_group_size >= size_limit) {  // 如果当前组大小超过了限制
        // 将所有类型组都移动到结果向量中，形成单独的组
        for (auto& entry : groups) {
          auto& group = entry.second;
          results.emplace_back(std::move(group));
        }
        cur_group_size = 0;  // 重置当前组大小计数器
        groups.clear();  // 清空类型组映射
      }
    } else {  // 如果不需要细粒度处理
      type_group.size += tensor_size;  // 累加当前类型组的大小
      if (type_group.size >= size_limit) {  // 如果当前类型组大小超过了限制
        results.emplace_back();  // 将空组添加到结果中
        std::swap(results.back(), type_group);  // 交换结果向量中的最后一个组和当前类型组
      }
    }
  }

  // End case. Look for any remaining groups and return them.
  // 处理剩余的类型组，并将它们添加到结果中
  for (auto& entry : groups) {
    auto& group = entry.second;
    if (!group.tensors.empty()) {  // 如果类型组中还有张量
      results.emplace_back(std::move(group));  // 将类型组移动到结果向量中
    }
  }

  return results;  // 返回所有组成的结果向量
}

void reorder_tensors_like(std::vector<Tensor>& tensors, TensorList order) {
  AT_ASSERT(tensors.size() == order.size());  // 断言输入张量和排序张量列表的大小相等

  std::unordered_map<size_t, std::vector<size_t>> type_id_to_indices;  // 使用类型 ID 映射到索引向量的无序映射
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_id_to_indices[type_id(tensors[i])].push_back(i);  // 将张量索引添加到对应类型 ID 的索引向量中

  std::unordered_map<size_t, size_t> type_id_to_type_used;  // 使用类型 ID 映射到已使用类型的数量的无序映射
  std::vector<Tensor> ordered_tensors;  // 按顺序存储张量的向量
  ordered_tensors.reserve(tensors.size());  // 预留足够的空间以容纳所有张量

  for (auto& tmpl_tensor : order) {  // 遍历排序张量列表中的每个模板张量
    size_t tmpl_type_id = type_id(tmpl_tensor);  // 获取模板张量的类型 ID
    auto& indices = type_id_to_indices[tmpl_type_id];  // 获取对应类型 ID 的张量索引向量
    auto& used = type_id_to_type_used[tmpl_type_id];  // 获取已使用的类型数量
    ordered_tensors.push_back(tensors[indices[used++]]);  // 将对应索引的张量添加到按顺序存储张量的向量中
  }

  std::swap(tensors, ordered_tensors);  // 交换输入张量和按顺序存储张量的向量
}

namespace {

at::Tensor get_indices(const at::Tensor& t) {  // 定义获取稀疏张量索引的函数
  return t._indices();  // 返回稀疏张量的索引
}

at::Tensor get_values(const at::Tensor& t) {  // 定义获取稀疏张量值的函数
  return t._values();  // 返回稀疏张量的值
}

} // namespace

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(
    // 接受一个张量列表作为输入参数
    auto flat_indices = utils::flatten_dense_tensors(fmap(tensors, &get_indices));
    // 对输入张量列表执行 fmap 操作，使用 get_indices 函数将每个张量转换为平铺的索引张量，并将结果赋给 flat_indices
    
    auto flat_values = utils::flatten_dense_tensors(fmap(tensors, &get_values));
    // 对输入张量列表执行 fmap 操作，使用 get_values 函数将每个张量转换为平铺的数值张量，并将结果赋给 flat_values
    
    // 返回一个 std::pair 对象，包含 flat_indices 和 flat_values 作为其成员
    return std::make_pair(flat_indices, flat_values);
} // 结束命名空间 torch::utils
```