# `.\pytorch\torch\csrc\jit\mobile\register_ops_common_utils.cpp`

```py
// 引入 ATen 库中的头文件和命名空间
#include <ATen/core/dynamic_type.h>
#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>

namespace torch {
namespace jit {

// 定义函数 normalizeIndex，用于将负数索引转换为正数索引
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // 处理负数索引的情况，将其转换为正数索引
    idx = list_size + idx;
  }
  return idx;
}

// 定义函数 tensorToListRecursive，将 Tensor 数据递归转换为嵌套列表类型 IValue
IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    at::TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size) {
  // 如果 ty 是 ListType 类型，则获取其元素类型
  if (auto list_type = ty->cast<at::ListType>()) {
    ty = list_type->getElementType();
  } else {
    // 如果输出类型是标量，则读取并推入正确类型的标量值到堆栈
    if (ty == at::IntType::get()) {
      int64_t scalar = *(int64_t*)data;
      return IValue(scalar);
    } else if (ty == at::FloatType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::Float ||
              scalar_ty == at::ScalarType::Double,
          "Unexpected scalar type for Tensor");
      double scalar =
          scalar_ty == at::ScalarType::Float ? *(float*)data : *(double*)data;
      return IValue(scalar);
    } else if (ty == at::ComplexType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::ComplexFloat ||
              scalar_ty == at::ScalarType::ComplexDouble,
          "Unexpected scalar type for Tensor");
      // 复数类型需要特别处理
      c10::complex<double> scalar = scalar_ty == at::ScalarType::ComplexFloat
          ? *(c10::complex<float>*)data
          : *(c10::complex<double>*)data;
      return IValue(scalar);
    } else if (ty == at::BoolType::get()) {
      bool scalar = *(bool*)data;
      return IValue(scalar);
    } else {
      // 如果出现不支持的类型，则抛出异常
      TORCH_CHECK(
          false,
          ty->repr_str(),
          " is not one of the supported types for tolist: int, float, bool");
    }
  }

  // 创建存放 ty 类型元素的列表 result
  auto result = c10::impl::GenericList(ty);
  result.reserve(sizes[cur_dim]);

  // 因为 ty 是列表类型，需要在当前维度处理每个切片的递归调用
  for (int64_t i = 0, e = sizes[cur_dim]; i < e; ++i) {
    auto inner_result = tensorToListRecursive(
        data,
        cur_dim + 1,
        num_tensor_dims,
        ty,
        scalar_ty,
        sizes,
        strides,
        element_size);

    // 根据 inner_result 的类型，将其添加到 result 中
    if (inner_result.isList()) {
      result.emplace_back(inner_result.toList());
    } else if (inner_result.isComplexDouble()) {
      result.emplace_back(inner_result.toComplexDouble());
    } else if (inner_result.isDouble()) {
      result.emplace_back(inner_result.toDouble());
    } else if (inner_result.isInt()) {
      result.emplace_back(inner_result.toInt());
    }
    // 还可以添加其他类型的处理，具体根据需要补充
  }

  // 返回构建的列表 result
  return result;
}

// 命名空间 jit 的结束
} // namespace jit
} // namespace torch


这段代码是用于将 Tensor 数据递归转换为嵌套列表类型 IValue 的实现。
    } else if (inner_result.isBool()) {
      // 如果 inner_result 是布尔型，则将其转换为布尔值并添加到 result 向量中
      result.emplace_back(inner_result.toBool());
    } else {
      // 如果 inner_result 类型未知，则触发断言错误，并输出错误信息
      TORCH_INTERNAL_ASSERT(
          false && "Unknown return type for tensorToListRecursive");
    }

    // 更新数据指针，移动到下一个元素的位置
    data += strides[cur_dim] * element_size;
}

} // namespace jit
} // namespace torch
```