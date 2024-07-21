# `.\pytorch\aten\src\ATen\native\ForeachUtils.h`

```
#pragma once

#include <ATen/Device.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/result_type_native.h>
#endif

#include <unordered_map>
#include <vector>

namespace at::native {
namespace {

// 检查张量列表中是否存在整数类型张量或布尔类型张量
inline bool has_integral_tensor(TensorList tensors, const bool includeBool) {
  return std::any_of(
      tensors.begin(), tensors.end(), [&includeBool](const auto& t) {
        return at::isIntegralType(t.scalar_type(), includeBool);
      });
}

// 检查张量列表中是否存在布尔类型张量
inline bool has_bool_tensor(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const auto& t) -> bool {
    return t.scalar_type() == ScalarType::Bool;
  });
}

// 检查 foreach API 的限制条件
// - 张量列表不能为空。
inline void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "Tensor list must have at least one tensor.");
}

// 检查 foreach API 的限制条件
// - 张量列表和标量列表必须具有相同数量的元素。
inline void check_foreach_api_restrictions(
    TensorList tensors,
    ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(
      tensors.size() == scalars.size(),
      "Tensor list must have same number of elements as scalar list.");
}

// 检查 foreach API 的限制条件
// - 张量列表1 和 张量列表2 必须具有相同数量的张量。
inline void check_foreach_api_restrictions(
    TensorList tensors1,
    TensorList tensors2) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(
      tensors1.size() == tensors2.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors2.size());
}

// 检查 foreach API 的限制条件
// - 张量列表1、张量列表2 和 张量列表3 必须具有相同数量的张量。
inline void check_foreach_api_restrictions(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors3.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(
      tensors1.size() == tensors2.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors2.size());
  TORCH_CHECK(
      tensors1.size() == tensors3.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors3.size());
}
    # 检查并确保在执行 foreach 操作时的 API 限制条件
    check_foreach_api_restrictions(tensors1, tensors2, tensors3)
    # 使用 TORCH_CHECK 进行断言检查，确保 tensors1 的长度与 scalars 的长度相同
    TORCH_CHECK(
        tensors1.size() == scalars.size(),
        "Tensor list must have same number of elements as scalar list, got ",
        tensors1.size(),
        " and ",
        scalars.size())
// Helper function called in check_fast_path_restrictions to check whether all
// corresponding tensors (aligning in index across the tensorLists) share the
// same device and dtype.
inline bool _check_tensors_share_device_and_dtype(
    ArrayRef<TensorList> tensorLists,
    const bool skip_dtype_check = false) {
  // 获取第一个张量的预期数据类型和设备
  const auto expected_dtype = tensorLists[0][0].dtype();
  const auto expected_device = tensorLists[0][0].device();

  // 判断张量是否符合条件的 Lambda 函数
  auto is_tensor_okay = [&](const Tensor& tensor) {
    return (skip_dtype_check || tensor.dtype() == expected_dtype) &&
        tensor.device() == expected_device && tensor.layout() == at::kStrided &&
        tensor.is_non_overlapping_and_dense();
  };

  // 遍历所有张量列表，检查每个张量是否符合条件
  for (const auto& tensorList : tensorLists) {
    for (const auto& tensor : tensorList) {
      if (!is_tensor_okay(tensor)) {
        return false;
      }
    }
  }

  // 所有张量满足条件则返回 true
  return true;
}

// Helper function called in check_fast_path_restrictions to check if
// corresponding tensors in tensor lists have the same sizes and strides.
inline bool _check_tensors_share_sizes_and_strides(
    ArrayRef<TensorList> tensorLists) {
  // 遍历除第一个张量列表外的所有张量列表
  for (const auto i : c10::irange(1, tensorLists.size())) {
    // 遍历第一个张量列表和当前张量列表中的张量，比较它们的大小和步长
    for (const auto j : c10::irange(tensorLists[0].size())) {
      if (tensorLists[0][j].sizes() != tensorLists[i][j].sizes() ||
          tensorLists[0][j].strides() != tensorLists[i][j].strides()) {
        return false;
      }
    }
  }

  // 所有张量的大小和步长都相同则返回 true
  return true;
}

// Helper function called in check_fast_path_restrictions to check whether
// all tensors type promote properly with the scalars in scalarList. This
// function assumes that _check_tensors_share_device_and_dtype has already been
// called so that all corresponding tensors in tensorLists have the same dtype.
// Then, it is sufficient to check the type promotion with just one tensorList.
inline bool _check_tensors_do_type_promotion_with_scalars(
    TensorList tensorList,
    ArrayRef<Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  // 遍历张量列表中的每个张量
  for (const auto i : c10::irange(tensorList.size())) {
    // 如果操作会将整数输入提升为浮点数，则检查是否所有整数张量都被正确提升
    if (does_op_promote_integer_inputs_to_float) {
      if (at::isIntegralType(
              tensorList[i].scalar_type(), /*includeBool*/ true)) {
        return false;
      }
    }
    // 如果提供了标量列表，则检查每个张量是否与其对应标量在类型上兼容
    if (!scalarList.empty()) {
      const auto& scalar =
          scalarList.size() == 1 ? scalarList[0] : scalarList[i];
      const auto& tensor = tensorList[i];
      // 检查每个张量与其对应标量之间的类型兼容性
      if (tensor.scalar_type() != at::native::result_type(scalar, tensor)) {
        return false;
      }
    }
  }

  // 所有张量与标量的类型兼容性检查通过则返回 true
  return true;
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors in all lists must have the same dtype.
// - All tensors must be on the same device
// - All tensors must have strided layout
// - 所有张量必须是非重叠且密集的
// - 结果张量的数据类型必须与输入张量相同

// [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
//     ``does_op_promote_integer_inputs_to_float=true`` 表示即使输入是整数或布尔型，操作的结果将是浮点数，但当前的快速路径不支持这种情况。
//     简而言之，这个标志在打开时，会阻止操作使用快速路径。

// 在调用此方法之前，请确保调用 check_foreach_api_restrictions。有一组必须满足的前提条件。

inline bool check_fast_path_restrictions(
    ArrayRef<TensorList> tensorLists,  // 张量列表的引用
    ArrayRef<Scalar> scalarList = {},  // 标量列表的引用，默认为空
    bool does_op_promote_integer_inputs_to_float = false) {  // 是否将整数输入提升为浮点数的标志，默认为假
  return _check_tensors_share_device_and_dtype(tensorLists) &&  // 检查张量是否共享设备和数据类型
      _check_tensors_share_sizes_and_strides(tensorLists) &&    // 检查张量是否共享大小和步幅
      _check_tensors_do_type_promotion_with_scalars(             // 检查张量和标量是否进行类型提升
             tensorLists[0],
             scalarList,
             does_op_promote_integer_inputs_to_float);
}

// 将张量转换为标量列表
inline std::vector<c10::Scalar> convert_tensor_to_scalar_list(
    const Tensor& scalarList_,  // 输入的标量张量
    int64_t expect_length) {    // 预期的长度
  std::vector<c10::Scalar> scalarList;  // 存储转换后的标量列表
  TORCH_CHECK(
      scalarList_.device() == c10::kCPU,  // 检查标量张量是否在CPU上
      "Expected scalars to be on CPU, got ",
      scalarList_.device(),
      " instead.");
  TORCH_CHECK(
      scalarList_.is_contiguous(),  // 检查标量张量是否连续
      "Expected scalars to be contiguous.");
  TORCH_CHECK(
      scalarList_.dim() == 1,  // 检查打包的标量张量是否是一维的
      "Expected packed scalar Tensor to be of dimension 1. Got ",
      scalarList_.dim(),
      " instead.");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      scalarList_.scalar_type(),
      "convert_tensor_to_scalar_list",
      [&]() {
        const scalar_t* scalar_data = scalarList_.const_data_ptr<scalar_t>();  // 获取标量数据指针
        TORCH_CHECK(
            (expect_length == scalarList_.size(0)),  // 检查标量张量的长度是否与预期长度相同
            "Expected length of scalars to match input of length ",
            expect_length,
            " but got ",
            scalarList_.size(0),
            " instead.");
        for (int64_t i = 0; i < scalarList_.size(0); i++) {
          scalarList.emplace_back(scalar_data[i]);  // 将标量数据逐个添加到标量列表中
        }
      });
  return scalarList;  // 返回转换后的标量列表
}

// see: [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
// 查看注释：``does_op_promote_integer_inputs_to_float=true``

// 检查是否可以使用快速路径执行操作
inline bool can_use_fast_route(
    ArrayRef<TensorList> tensorLists,  // 张量列表的引用
    ArrayRef<Scalar> scalarList = {},  // 标量列表的引用，默认为空
    bool does_op_promote_integer_inputs_to_float = false) {  // 是否将整数输入提升为浮点数的标志，默认为假
  return check_fast_path_restrictions(  // 调用检查快速路径限制的函数
      tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

// see: [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
// 查看注释：``does_op_promote_integer_inputs_to_float=true``

// 检查是否可以使用快速路径执行操作，针对两个张量列表的情况
inline bool can_use_fast_route(
    TensorList tensors1,  // 第一个张量列表
    TensorList tensors2,  // 第二个张量列表
    bool does_op_promote_integer_inputs_to_float = false) {  // 是否将整数输入提升为浮点数的标志，默认为假
  return can_use_fast_route(
      {tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);  // 调用带有两个张量列表的快速路径检查函数
}
    // 使用别名定义键值对类型，表示设备和张量数据类型的组合
    using DeviceDtypeKey = std::pair<at::Device, at::ScalarType>;
    // 使用别名定义索引类型，表示一组索引
    using IndicesT = std::vector<size_t>;
    // 使用别名定义嵌套的可选张量向量类型
    using nested_optional_tensorvec_t =
        std::vector<std::vector<std::optional<at::Tensor>>>;
    // 使用别名定义包含嵌套张量向量和索引的类型
    using TensorsAndIndicesT = std::pair<nested_optional_tensorvec_t, IndicesT>;
    // 使用别名定义扁平映射类型，将设备和数据类型键映射到张量向量和索引的组合
    using FlatMap = std::unordered_map<
        DeviceDtypeKey,
        TensorsAndIndicesT,
        ParamsHash<DeviceDtypeKey>>;

    // 内联函数：根据嵌套的可选张量向量按照第一个张量的设备和数据类型分组，并可能包含索引
    inline FlatMap _group_tensors_by_first_tensors_device_and_dtype(
        const nested_optional_tensorvec_t& nested_tensorlist,
        const bool with_indices) {
      // 初始化空的扁平映射，用于存储按照设备和数据类型分组后的张量向量和索引
      FlatMap grouped_tensors_with_indices;

      // 断言：嵌套的张量列表不应为空
      TORCH_CHECK(!nested_tensorlist.empty());
      // 断言：第一个张量列表不应为空
      TORCH_CHECK(!nested_tensorlist[0].empty());
      // 获取嵌套张量列表的总列表数和第一个张量列表中张量的数量
      const auto num_lists = nested_tensorlist.size();
      const auto num_tensors = nested_tensorlist[0].size();

      // 断言：所有嵌套的张量列表应具有相同的张量数量，或者为空
      TORCH_CHECK(std::all_of(
          nested_tensorlist.cbegin(),
          nested_tensorlist.cend(),
          [&](const auto& tensorlist) -> bool {
            // 注释：允许空的张量列表，参考：
            // https://github.com/pytorch/pytorch/blob/85885301fd3c6adb8b9dc3cf7afadf6945566684/torch/utils/_foreach_utils.py#L21-L24
            return tensorlist.size() == num_tensors || tensorlist.size() == 0;
          }));

      // 遍历每个张量的索引范围
      for (const auto& tensor_index : c10::irange(num_tensors)) {
        // 匿名函数：根据第一个张量列表中的每个张量创建设备和数据类型键
        const auto key = [&]() -> DeviceDtypeKey {
          const auto t = nested_tensorlist[0][tensor_index];
          // 断言：第一个张量列表中的张量应该被定义，但发现第 tensor_index 个张量未定义
          TORCH_CHECK(
              t.has_value(),
              "Tensors of the first list of nested Tensor lists are supposed to be defined but ",
              "the ",
              tensor_index,
              "-th Tensor is not.");
          return {t->device(), t->scalar_type()};
        }();
    // 使用 TORCH_CHECK 确保所有嵌套的 tensorlist 满足以下条件
    TORCH_CHECK(
        // 检查所有 tensorlist 是否满足以下条件
        std::all_of(
            // 遍历嵌套 tensorlist 的起始到终止迭代器
            nested_tensorlist.cbegin(),
            nested_tensorlist.cend(),
            [&](const auto& tensorlist) -> bool {
              // 如果 tensorlist 为空，则返回 true
              if (tensorlist.size() == 0) {
                return true;
              }
              // 获取指定索引位置的 tensor
              const auto& tensor = tensorlist[tensor_index];
              // 注意(crcrpar): 当前函数的作用域是优化器，因此可能存在 `state_steps` 和其他标量，
              // 其元素是浮点张量，无论参数的 dtype 是什么。
              if (!tensor.has_value()) {
                return true;
              } else {
                // 获取 tensor 的标量类型和设备
                const auto s = tensor->scalar_type();
                const auto d = tensor->device();
                // 注意: `step` 或 `state_step` 默认为 float32 类型。
                if (key.first == d) {
                  // 返回设备相同且标量类型相同或者为 Float 或 Double 的情况
                  return key.second == s || s == at::ScalarType::Float ||
                      s == at::ScalarType::Double;
                } else if (d.is_cpu()) {
                  // 注意(crcrpar): 有一些测试用例（例如 TestOptim::test_adam）中，state_steps 在 CPU 上，
                  // 其他张量在 CUDA 上。当前的 state_step 张量的 dtype 是 float。
                  // 返回标量类型为 Float 或 Double 的情况
                  return s == at::ScalarType::Float ||
                      s == at::ScalarType::Double;
                } else {
                  // 其他情况返回 false
                  return false;
                }
              }
            }),
        // 若不满足条件，抛出异常，提示相同索引的张量必须在相同设备和相同 dtype 下，除了 `step` 张量可以在 CPU 上为 float32/64。
        "Tensors of the same index must be on the same device and the same dtype except `step` tensors that can be CPU and float32/64 notwithstanding");

    // 如果 grouped_tensors_with_indices 不包含 key，则插入 key 和 TensorsAndIndicesT 对象
    if (!grouped_tensors_with_indices.count(key)) {
      grouped_tensors_with_indices.insert(
          {key,
           TensorsAndIndicesT{
               // 创建 nested_optional_tensorvec_t 对象
               [&]() -> nested_optional_tensorvec_t {
                 nested_optional_tensorvec_t nested_tensorvec;
                 nested_tensorvec.reserve(num_lists);
                 // 遍历 num_lists 范围内的索引 i
                 for (const auto& i : c10::irange(num_lists)) {
                   std::vector<std::optional<at::Tensor>> tensors;
                   // 如果 nested_tensorlist[i] 不为空，则预留空间以提升性能
                   if (!nested_tensorlist[i].empty()) {
                     // 注意: num_tensors 是任何内部 tensor 引用列表的最大可能长度。预留此最大长度的空间，以提升性能。
                     tensors.reserve(num_tensors);
                   }
                   // 将 tensors 添加到 nested_tensorvec 中
                   nested_tensorvec.emplace_back(tensors);
                 }
                 // 返回创建的 nested_tensorvec
                 return nested_tensorvec;
               }(),
               // 创建 IndicesT 对象
               [&]() -> IndicesT {
                 // 如果不需要索引，返回空对象
                 if (!with_indices) {
                   return {};
                 } else {
                   // 否则，创建并返回空间预留的 indices 对象
                   IndicesT indices;
                   indices.reserve(num_tensors);
                   return indices;
                 }
               }()}});
    }
    for (const auto& list_index : c10::irange(num_lists)) {
        // 遍历 num_lists 范围内的所有索引，使用 const auto& list_index 迭代每个索引
        if (!nested_tensorlist[list_index].empty()) {
            // 检查 nested_tensorlist[list_index] 是否非空
            // 如果非空，将 nested_tensorlist[list_index][tensor_index] 添加到 grouped_tensors_with_indices[key] 的第一个元素的 list_index 位置
            grouped_tensors_with_indices[key].first[list_index].emplace_back(
                nested_tensorlist[list_index][tensor_index]);
        }
    }
    if (with_indices) {
        // 如果 with_indices 为真，则将 tensor_index 添加到 grouped_tensors_with_indices[key] 的第二个元素中
        grouped_tensors_with_indices[key].second.emplace_back(tensor_index);
    }
  }

  // 返回最终的 grouped_tensors_with_indices 结果
  return grouped_tensors_with_indices;
}

} // namespace at::native
} // namespace
```