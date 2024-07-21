# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\iterators\make_residual_last.h`

```
/*
 * 包含公司及其关联公司的版权声明
 * 版权所有。
 *
 * 此源代码在根目录下的 LICENSE 文件中，采用 BSD 风格许可证授权使用。
 */
#pragma once

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_access_iterator_residual_last.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/predicated_tile_iterator_residual_last.h>

// 声明 cutlass 命名空间和 transform 子命名空间
namespace cutlass {
namespace transform {
namespace threadblock {

// 模板：根据基础迭代器类型创建 ResidualLast 迭代器
template <typename BaseIterator>
struct MakeIteratorResidualLast;

// 特化模板：针对 PredicatedTileIterator 创建 ResidualLast 迭代器
template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessSize,
    Gather>> {
  using Iterator = PredicatedTileIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessSize,
      Gather>;
};

// 特化模板：针对 PredicatedTileAccessIterator 创建 ResidualLast 迭代器
template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileAccessIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessType,
    Gather>> {
  using Iterator = PredicatedTileAccessIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType,
      Gather>;
};

} // namespace threadblock
} // namespace transform
} // namespace cutlass
```