# `.\pytorch\aten\src\ATen\templates\FunctionalInverses.h`

```py
#pragma once
// 指定此头文件仅被编译一次

// ${generated_comment}
// 自动生成的注释，实际使用时会被生成器替换

#include <ATen/Tensor.h>
// 引入 ATen 库中的 Tensor 类

namespace at {
namespace functionalization {
// 命名空间 at 下的 functionalization 命名空间

enum class InverseReturnMode {
  /// Specifies that functional inverses should always return a view.
  AlwaysView,
  /// Specifies that functional inverses should always return a non-view / copy.
  NeverView,
  /// Specifies that functional inverses should return a view unless a (copying) scatter
  /// inverse exists, in which case that will be used instead.
  /// This avoids as_strided() calls that can be difficult for subclasses to handle.
  ViewOrScatterInverse,
};
// 枚举类型 InverseReturnMode，定义了函数反转返回的模式

struct FunctionalInverses {
// 定义结构体 FunctionalInverses

${view_inverse_declarations}
// 插入视图反转声明的代码块

// NB: These are not generated! They're manually implemented in the template.
// TODO: Change codegen to generate these. See the following link:
// https://github.com/pytorch/pytorch/blob/main/torchgen/model.py#L2583-L2585
// 这些不是自动生成的！它们在模板中是手动实现的。
// TODO: 更改代码生成器以生成这些内容。参见以下链接：

static at::Tensor chunk_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int chunks, int dim);
// 声明静态函数 chunk_inverse，用于执行块反转操作

static at::Tensor narrow_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int dim, c10::SymInt start, c10::SymInt length);
// 声明静态函数 narrow_inverse，用于执行窄化反转操作

};
// 结束结构体 FunctionalInverses 的定义

}
}
// 结束命名空间 functionalization 和 at
```