# `.\pytorch\aten\src\ATen\native\IndexingUtils.h`

```py
#pragma once
#include <ATen/ExpandUtils.h>  // 包含扩展工具的头文件
#include <ATen/native/CanUse32BitIndexMath.h>  // 包含32位索引数学工具的头文件
#include <ATen/native/TensorIterator.h>  // 包含张量迭代器的头文件
#include <ATen/core/IListRef.h>  // 包含列表引用的头文件
#include <c10/util/irange.h>  // 包含范围迭代器的头文件

namespace at::native {

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
  // 如果索引张量的形状与张量自身的形状不匹配，则抛出错误
}

static C10_UNUSED std::vector<Tensor> expandTensors(const Tensor & self, IOptTensorListRef indices) {
  // 如果索引是 ByteTensor 或者 BoolTensor（掩码），则将它们扩展为相当于 LongTensor 的索引
  std::vector<Tensor> result;
  for (const auto& index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();  // 将空张量放入结果向量中
    } else {
      const auto& index = *index_opt;
      if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
        if (index.scalar_type() == kByte) {
          TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
          " please use a dtype torch.bool instead.");
        }
        // ByteTensor 或者 bool 张量的尺寸必须与 self 对应维度的尺寸匹配
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = static_cast<int64_t>(result.size() + j);
          if (index.size(j) != self.size(srcIdx)) {
            invalid_mask(self, srcIdx, index, j);  // 检查索引掩码尺寸是否正确
          }
        }
        // 用非零值替换索引张量
        auto nonzero = index.nonzero();
        for (const auto j : c10::irange(index.dim())) {
          result.emplace_back(nonzero.select(1, j));  // 将非零值作为结果之一
        }
      } else {
        result.emplace_back(index);  // 直接将索引张量放入结果向量中
      }
    }
  }
  return result;  // 返回扩展后的结果向量
}

static C10_UNUSED void checkIndexTensorTypes(IOptTensorListRef indices, bool allow_int=false) {
  for (const auto& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      auto scalarType = tensor->scalar_type();
      if (allow_int) {
        if (scalarType != kLong && scalarType != kByte && scalarType != kBool && scalarType != kInt) {
            TORCH_CHECK_INDEX(false, "tensors used as indices must be long, int, byte or bool tensors");
        }
      } else {
        if (scalarType != kLong && scalarType != kByte && scalarType != kBool) {
            TORCH_CHECK_INDEX(false, "tensors used as indices must be long, byte or bool tensors");
        }
      }
    }
  }
}

inline torch::List<std::optional<Tensor>> toListOfOptionalTensors(ArrayRef<Tensor> list) {
  torch::List<std::optional<Tensor>> result;
  result.reserve(list.size());
  for (const Tensor& a : list) {
    result.push_back(a);  // 将张量列表转换为包含可选张量的 Torch 列表
  }
  return result;
}

inline torch::List<std::optional<Tensor>> toListOfOptionalTensors(ArrayRef<IValue> list) {
  torch::List<std::optional<Tensor>> result;
  result.reserve(list.size());
  for (const IValue& a : list) {
    result.push_back(a.isTensor() ? std::optional<Tensor>(a.toTensor()) : std::optional<Tensor>());
  }
  return result;



    // 将条件表达式的结果推送到 result 向量中，条件是 a 是否为 Tensor 对象
    // 如果 a 是 Tensor 对象，则将其转换为 std::optional<Tensor> 类型后推送到 result 中
    // 如果 a 不是 Tensor 对象，则推送一个空的 std::optional<Tensor> 到 result 中
    result.push_back(a.isTensor() ? std::optional<Tensor>(a.toTensor()) : std::optional<Tensor>());
  }
  // 返回填充后的 result 向量，其中包含了每个元素可能是 Tensor 对象的可选项
  return result;
}

// 检查是否存在连续的子空间
static C10_UNUSED bool hasContiguousSubspace(TensorList tl) {
  // 如果所有非空张量是相邻的则返回 true
  auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}


// 将张量和索引一起转置，使得所有非空索引索引张量的前 k 维度。返回转置后的张量
// 和重新排序的索引。例如：
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// 返回
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static C10_UNUSED std::tuple<Tensor, std::vector<Tensor>>
transposeToFront(const Tensor& self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

// 将张量和索引一起转置到前面，并返回逆排列的维度。例如：
// transposeToFrontAndInvPerm(tensor, {nullptr, a, nullptr, b})
// 返回
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}, [2, 0, 3, 1]
inline std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(const Tensor& self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> invPerm;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  invPerm.resize(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    invPerm[dims[i]] = i;
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

// 表示高级索引的结构体
struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};


} //namespace at::native
```