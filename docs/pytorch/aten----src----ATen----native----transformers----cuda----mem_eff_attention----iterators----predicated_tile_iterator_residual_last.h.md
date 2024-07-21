# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\iterators\predicated_tile_iterator_residual_last.h`

```
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of tiles from pitch-linear rank=2
   tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile
   this iterator visits maybe partial, then the remaining tiles are complete.
   So, we only need to compute the predicates twice, once before the first tile
   and once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

#pragma once

#include <cutlass/arch/memory.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileIteratorResidualLast
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
/// Regular tile iterator concept optimized for efficient memory access patterns,
/// minimizing register usage and arithmetic operations.
///
/// The iterator assumes a fixed layout defined when the "Params" object is created,
/// including base pointer and tensor extents that remain immutable throughout its lifetime.
///
/// Initialization allows for an initial logical coordinate offset, which can be adjusted
/// later but with increased computational cost.
///
/// Iteration order starts by visiting a residual tile that may be partially filled in
/// both the advance and steady-state dimensions. This residual tile is assumed to be
/// the last one in the sequence. Advancing the iterator from its initial state moves
/// to the first fully filled tile in the advance dimension, recomputing predicates.
/// Subsequent accesses do not require predicate updates, optimizing register usage
/// and pointer arithmetic.
///
/// To maintain efficiency, the iterator should be dereferenced and advanced at least
/// once outside any loop structure to minimize arithmetic overhead.
///
/// Accesses out of bounds are considered safe as long as the `clear_mask()` method
/// is called before dereferencing the iterator.
///
/// Example:
///
/// An efficient pipeline structure can be constructed as shown below:
///
/// template <typename Iterator>
/// __global__ void kernel(
///   typename Iterator::Params params,
///   typename Iterator::Element *ptr,
///   TensorCoord extent) {
///
///   typename Iterator::Fragment fragment;
///
///   TensorCoord threadblock_offset(0, 0);
///
///   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offset);
///
///   fragment = *iter;        // loads the "residual" tile first
///   ++iter;                  // advances to the first "steady state" tile and updates internal masks
///
///   #pragma unroll
///   for (int i = Remaining - 1; i >= 0; --i) {
///
///     f(fragment);
///
///     if (!i) {
///       iter.clear_mask();   // clears masks efficiently - subsequent loads become NO-OPs
///     }
///
///     fragment = *iter;      // loads tiles during the "steady state" phase
///     ++iter;                // advances to the next tile efficiently due to steady-state masks
///   }
/// }
///
/// void host(TensorView<Element, 2, layout::PitchLinear> view) {
///
///   using Iterator = transform::threadblock::PredicatedTileIteratorResidualLast;
///
///   typename Iterator::Params params(view.layout());
///
///   kernel<Iterator>(params, view.data());
/// }
///
template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize = ThreadMap::kElementsPerAccess,
    bool Gather = false>
class PredicatedTileIteratorResidualLast;
////////////////////////////////////////////////////////////////////////////////
/// Specialization of PredicatedTileIteratorResidualLast for pitch-linear data.
/// 为基于线性分布的数据特化的 PredicatedTileIteratorResidualLast 类。
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
/// 满足的迭代器概念：前向迭代器、可读连续块迭代器、可写连续块迭代器、掩码块迭代器概念。
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    int AccessSize,
    bool Gather>
class PredicatedTileIteratorResidualLast<
    Shape_,
    Element_,
    layout::PitchLinear,
    AdvanceRank,
    ThreadMap_,
    AccessSize,
    Gather> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");
  // 静态断言：仅适用于按线性分布的迭代器，在连续(rank=0)或跨距(rank=1)维度上进行前进。

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<
      Element,
      AccessSize,
      (AccessSize * sizeof_bits<Element>::value / 8)>;
  // 用于内部内存访问的类型

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = PredicatedTileAccessIteratorResidualLast<
      Shape,
      Element,
      Layout,
      kAdvanceRank,
      ThreadMap,
      AccessType,
      Gather>;
  // 用于计算地址的基础迭代器

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;
  // 每个向量的访问次数，由基础迭代器决定

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  // 要加载或存储的片段对象类型

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;
  // 谓词向量存储用于保护访问的掩码

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    using Base = typename TileAccessIterator::Params::Base;

    friend PredicatedTileIteratorResidualLast;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;
    // 参数对象

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : params_(layout) {}
    // 给定一个按线性分布的张量布局，构造参数对象

    CUTLASS_HOST_DEVICE
    Params() {}
    // 默认构造函数

    CUTLASS_HOST_DEVICE
  Params(Base const& base) : params_(base) {}
  // 使用基类对象初始化 Params 对象

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;
  // 内部指针类型，允许快速地址算术运算

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;
  // 瓦片访问迭代器的数据成员

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      /// Gather indices
      int const* indices = nullptr)
      : address_iterator_(
            params.params_,
            pointer,
            extent,
            thread_id,
            threadblock_offset,
            indices) {}
  // 从预计算的状态、线程块偏移和线程 ID 构造 TileIterator

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}
  // 使用零线程块偏移构造 PredicatedTileIteratorResidualLast

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }
  // 在元素单位上添加指针偏移量

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }
  // 在内存中前进到下一个瓦片

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this);
    operator++();
    return self;
  }
  // 在内存中前进到下一个瓦片（后置版本）

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    address_iterator_.clear_mask(enable);
  }
  // 高效清除谓词集合

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
  // 设置剩余瓦片的使能状态
  address_iterator_.set_residual_tile(enable);
}

/// 高效清除谓词集合
CUTLASS_HOST_DEVICE
void enable_mask() {
  address_iterator_.enable_mask();
}

/// 设置谓词掩码，覆盖谓词迭代器中存储的值
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  address_iterator_.set_mask(mask);
}

/// 获取谓词掩码
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  address_iterator_.get_mask(mask);
}

/// 使用指针偏移量加载片段数据
CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  load_with_byte_offset(
      frag, pointer_offset * sizeof_bits<Element>::value / 8);
}

/// 使用字节偏移量加载片段数据
CUTLASS_DEVICE
void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
  AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        int idx = v +
            kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

        // 设置迭代索引
        address_iterator_.set_iteration_index(idx);

        // 计算指定偏移量处的字节指针
        char const* byte_ptr =
            reinterpret_cast<char const*>(address_iterator_.get()) +
            byte_offset;

        // 将字节指针转换为访问类型指针
        AccessType const* access_ptr =
            reinterpret_cast<AccessType const*>(byte_ptr);

        // 全局加载操作
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
            frag_ptr[idx], access_ptr, address_iterator_.valid());

        // 迭代器递增
        ++address_iterator_;
      }
    }
  }
}

/// 从内存加载一个片段
CUTLASS_DEVICE
void load(Fragment& frag) {
  load_with_byte_offset(frag, 0);
}

/// 使用指针偏移量将片段存储到内存
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  store_with_byte_offset(
      frag, pointer_offset * sizeof_bits<Element>::value / 8);
}

/// 使用字节偏移量将片段存储到内存
CUTLASS_DEVICE
void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
  address_iterator_.set_iteration_index(0);
  AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        int idx = v +
            kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

        // 计算指定偏移量处的字节指针
        char* byte_ptr =
            reinterpret_cast<char*>(address_iterator_.get()) + byte_offset;

        // 将字节指针转换为访问类型指针
        AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_ptr);

        // 如果迭代器有效，则执行存储操作
        if (address_iterator_.valid()) {
          *access_ptr = frag_ptr[idx];
        }

        // 迭代器递增
        ++address_iterator_;
      }
    }
  }
}
  }
}

/// 在内存中存储一个片段
CUTLASS_DEVICE
void store(Fragment const& frag) {
  // 调用带有字节偏移量的存储函数，将片段存储到内存中
  store_with_byte_offset(frag, 0);
}
  };

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                     // 模板参数：迭代器形状
    typename Element_,                   // 模板参数：元素类型
    int AdvanceRank,                     // 模板参数：前进的秩
    typename ThreadMap_,                 // 模板参数：线程映射策略
    int AccessSize,                      // 模板参数：访问尺寸
    bool Gather>                         // 模板参数：是否聚合
class PredicatedTileIteratorResidualLast<
    Shape_,                              // 迭代器形状
    Element_,                            // 元素类型
    layout::ColumnMajor,                 // 布局类型：列优先
    AdvanceRank,                         // 前进的秩
    ThreadMap_,                          // 线程映射策略
    AccessSize,                          // 访问尺寸
    Gather> {                            // 是否聚合
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                  // 别名：迭代器形状
  using Element = Element_;              // 别名：元素类型
  using Layout = layout::ColumnMajor;    // 别名：布局类型为列优先
  static int const kAdvanceRank = AdvanceRank;  // 静态常量：前进的秩
  using ThreadMap = ThreadMap_;          // 别名：线程映射策略

  using Index = typename Layout::Index;      // 别名：索引类型
  using LongIndex = typename Layout::LongIndex;  // 别名：长索引类型

  using TensorRef = TensorRef<Element, Layout>;    // 引用类型：张量引用
  using TensorView = TensorView<Element, Layout>;  // 视图类型：张量视图
  using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型

  using Pointer = Element*;              // 指针类型：元素指针
  using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量元素指针类型

  using UnderlyingIterator = PredicatedTileIteratorResidualLast<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,  // 内部迭代器类型：基于行列的线性间距形状
      Element,                            // 元素类型
      layout::PitchLinear,                // 布局类型：线性间距
      (kAdvanceRank == 0 ? 0 : 1),        // 前进的秩
      ThreadMap,                          // 线程映射策略
      AccessSize,                         // 访问尺寸
      Gather>;                            // 是否聚合

  using AccessType = typename UnderlyingIterator::AccessType;  // 访问类型：内部迭代器的访问类型

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 片段对象类型：用于加载或存储的数组

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;  // 谓词向量类型：存储掩码以保护访问

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;  // 内部迭代器的参数对象

   public:
    CUTLASS_HOST_DEVICE
    Params() {}                                  // 默认构造函数

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}  // 根据线性间距张量的布局构造参数对象

    CUTLASS_HOST_DEVICE
    ~Params() {}                                  // 析构函数
  };
};
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id, ///< ID of each participating thread
      TensorCoord const& threadblock_offset, ///< Initial offset of threadblock
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_, ///< Initialize the underlying iterator with parameters
            pointer, ///< Initialize the underlying iterator with pointer to tensor data
            layout::PitchLinearCoord(extent.row(), extent.column()), ///< Set tensor extent using pitch-linear coordinates
            thread_id, ///< Set thread ID for tile iterator
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column()), ///< Set threadblock offset using pitch-linear coordinates
            indices) {} ///< Initialize with optional gather/scatter indices

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(
            params, ///< Call the main constructor with precomputed parameters
            pointer, ///< Forward pointer to tensor data
            extent, ///< Forward tensor extent
            thread_id, ///< Forward thread ID
            make_Coord(0, 0)) {} ///< Use default threadblock offset (0, 0)

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset); ///< Delegate adding pointer offset to underlying iterator
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    ++iterator_; ///< Advance the underlying iterator to the next tile
    return *this; ///< Return a reference to this iterator
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this); ///< Create a copy of the current iterator
    operator++(); ///< Call prefix increment operator to advance iterator
    return self; ///< Return the copied iterator before advancing
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable); ///< Delegate clearing predicate mask to underlying iterator
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    // Implementation not provided in the given code snippet
  }
    /// 设置剩余瓦片的使能状态
    void set_residual_tile(bool enable) {
      iterator_.set_residual_tile(enable);
    }
    
    /// 高效地清除谓词集合
    CUTLASS_HOST_DEVICE
    void enable_mask() {
      iterator_.enable_mask();
    }
    
    /// 设置谓词掩码，覆盖谓词迭代器中存储的值
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
      iterator_.set_mask(mask);
    }
    
    /// 获取谓词掩码
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
      iterator_.get_mask(mask);
    }
    
    /// 根据指针偏移量从内存加载一个片段
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
      iterator_.load_with_pointer_offset(frag, pointer_offset);
    }
    
    /// 根据字节偏移量从内存加载一个片段
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
      iterator_.load_with_byte_offset(frag, byte_offset);
    }
    
    /// 从内存加载一个片段，默认从指针偏移量为0处开始加载
    CUTLASS_DEVICE
    void load(Fragment& frag) {
      load_with_pointer_offset(frag, 0);
    }
    
    /// 根据指针偏移量将一个片段存储到内存
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
      iterator_.store_with_pointer_offset(frag, pointer_offset);
    }
    
    /// 根据字节偏移量将一个片段存储到内存
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
      iterator_.store_with_byte_offset(frag, byte_offset);
    }
    
    /// 将一个片段存储到内存，默认从指针偏移量为0处开始存储
    CUTLASS_DEVICE
    void store(Fragment const& frag) {
      store_with_pointer_offset(frag, 0);
    }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                   // 模板参数：迭代器的形状类型
    typename Element_,                 // 模板参数：元素类型
    int AdvanceRank,                   // 模板参数：前进等级，可以是0或1
    typename ThreadMap_,               // 模板参数：线程映射策略类型
    int AccessSize,                    // 模板参数：访问大小
    bool Gather                        // 模板参数：是否进行收集
>
class PredicatedTileIteratorResidualLast<
    Shape_,                            // 迭代器的形状类型
    Element_,                          // 元素类型
    layout::RowMajor,                  // 布局类型：行主序
    AdvanceRank,                       // 前进等级
    ThreadMap_,                        // 线程映射策略类型
    AccessSize,                        // 访问大小
    Gather> {                          // 是否进行收集

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                // 使用模板参数指定的形状类型
  using Element = Element_;            // 使用模板参数指定的元素类型
  using Layout = layout::RowMajor;     // 使用行主序作为布局类型
  static int const kAdvanceRank = AdvanceRank;  // 设置前进等级为模板参数中的值
  using ThreadMap = ThreadMap_;        // 使用模板参数指定的线程映射策略类型

  using Index = typename Layout::Index;        // 使用布局类型的索引类型
  using LongIndex = typename Layout::LongIndex; // 使用布局类型的长索引类型

  using TensorRef = TensorRef<Element, Layout>;  // 引用类型：基于元素和布局类型的张量引用
  using TensorView = TensorView<Element, Layout>;  // 视图类型：基于元素和布局类型的张量视图
  using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型：基于布局类型的张量坐标

  using Pointer = Element*;                  // 指针类型：元素类型指针
  using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量指针类型

  // UnderlyingIterator是用于底层实现的迭代器类型
  using UnderlyingIterator = PredicatedTileIteratorResidualLast<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,  // 使用列和行组成的PitchLinearShape作为形状类型
      Element,                          // 元素类型
      layout::PitchLinear,               // 使用PitchLinear布局类型
      (kAdvanceRank == 0 ? 1 : 0),       // 根据前进等级确定AdvanceRank
      ThreadMap,                        // 线程映射策略类型
      AccessSize,                       // 访问大小
      Gather>;                          // 是否进行收集

  using AccessType = typename UnderlyingIterator::AccessType;  // 访问类型：基于底层迭代器的访问类型

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 片段类型，用于加载或存储

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;  // 谓词向量类型，用于保护访问

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;  // 底层迭代器的参数对象

   public:
    CUTLASS_HOST_DEVICE
    Params() {}  // 默认构造函数

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}  // 构造函数，根据PitchLinear张量的布局构造参数对象
  };

  CUTLASS_HOST_DEVICE
  // 这里应该还有更多的代码，但未在提供的片段中显示出来


注释：这段代码定义了一个特化的迭代器类 `PredicatedTileIteratorResidualLast`，用于处理行主序布局的Pitch线性数据。它满足了多个迭代器概念，并提供了对底层迭代器类型及其参数的引用和定义。
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id, ///< ID of each participating thread
      TensorCoord const& threadblock_offset, ///< Initial offset of threadblock
      int const* indices = nullptr ///< Gather indices
      )
      : iterator_(  // 初始化成员变量 iterator_
            params.params_,  // 使用预先计算的参数对象初始化迭代器的参数
            pointer,  // 设置指向张量起始位置的指针
            layout::PitchLinearCoord(extent.column(), extent.row()),  // 设置张量的行列范围
            thread_id,  // 设置参与线程的 ID
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row()),  // 设置线程块的初始偏移量
            indices) {}  // 设置 gather 操作的索引数组

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(  // 调用上面定义的构造函数，传递零偏移量作为参数
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);  // 调用迭代器的方法，增加指针偏移量
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    ++iterator_;  // 前缀递增运算符重载，使迭代器进入下一个内存中的瓦片
    return *this;  // 返回自身对象的引用
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this);  // 复制构造函数，创建当前对象的副本
    operator++();  // 调用前缀递增运算符重载，使迭代器进入下一个内存中的瓦片
    return self;  // 返回复制的对象副本
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);  // 调用迭代器的方法，有效地清除谓词集合
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);  // 调用迭代器的方法，设置是否为残留瓦片
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    /// 启用迭代器的掩码功能
    iterator_.enable_mask();
    
    
    
    /// 设置谓词掩码，覆盖谓词迭代器中存储的值
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
      iterator_.set_mask(mask);
    }
    
    
    
    /// 获取当前的掩码值
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
      iterator_.get_mask(mask);
    }
    
    
    
    /// 根据指针偏移量从内存加载一个片段
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
      iterator_.load_with_pointer_offset(frag, pointer_offset);
    }
    
    
    
    /// 根据字节偏移量从内存加载一个片段
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
      iterator_.load_with_byte_offset(frag, byte_offset);
    }
    
    
    
    /// 从内存加载一个片段，使用默认的指针偏移量（0）
    CUTLASS_DEVICE
    void load(Fragment& frag) {
      load_with_pointer_offset(frag, 0);
    }
    
    
    
    /// 根据指针偏移量将一个片段存储到内存中
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
      iterator_.store_with_pointer_offset(frag, pointer_offset);
    }
    
    
    
    /// 根据字节偏移量将一个片段存储到内存中
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
      iterator_.store_with_byte_offset(frag, byte_offset);
    }
    
    
    
    /// 将一个片段存储到内存中，使用默认的指针偏移量（0）
    CUTLASS_DEVICE
    void store(Fragment const& frag) {
      store_with_pointer_offset(frag, 0);
    }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for affine rank-2 data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    int AccessSize>
class PredicatedTileIteratorResidualLast<
    Shape_,
    Element_,
    layout::AffineRankN<2>,
    AdvanceRank,
    ThreadMap_,
    AccessSize,
    false> {
 public:
  // 确保AdvanceRank为0或1，因为此处是用于线性迭代器的特化，可以沿着连续（rank=0）或跨距（rank=1）维度进行推进。
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                           // 迭代器形状
  using Element = Element_;                       // 元素类型
  using Layout = layout::AffineRankN<2>;          // 使用二维仿射布局
  static int const kAdvanceRank = AdvanceRank;     // 推进的维度（rank）
  using ThreadMap = ThreadMap_;                   // 线程映射策略

  using Index = typename Layout::Index;           // 索引类型
  using LongIndex = typename Layout::LongIndex;   // 长整型索引类型

  using TensorRef = TensorRef<Element, Layout>;   // 引用型张量
  using TensorView = TensorView<Element, Layout>; // 视图型张量
  using TensorCoord = typename Layout::TensorCoord; // 张量坐标类型

  using Pointer = Element*;                       // 指针类型
  using NonConstPointer = typename platform::remove_const<Element>::type*; // 非常量指针类型

  /// Type used for internal memory accesses
  /// 用于内存访问的类型
  using AccessType = AlignedArray<
      Element,
      AccessSize,
      (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  /// 计算地址的底层迭代器
  using TileAccessIterator = PredicatedTileAccessIteratorResidualLast<
      Shape,
      Element,
      Layout,
      kAdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector; // 每向量访问的数量

  /// Fragment object to be loaded or stored
  /// 要加载或存储的片段对象
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  /// 谓词向量存储用于保护访问的掩码
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  /// 参数对象是预计算的状态，可由主机构造
  class Params {
   public:
    friend PredicatedTileIteratorResidualLast;

   private:
    /// Parameters object
    /// 参数对象
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    /// 给定线性张量的布局，构造参数对象
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : params_(layout) {}

    CUTLASS_HOST_DEVICE
    Params() {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : address_iterator_(
            params.params_, // 使用参数对象初始化地址迭代器
            pointer, // 设置指向张量起始位置的指针
            extent, // 设置张量的尺寸
            thread_id, // 设置每个参与线程的ID
            threadblock_offset) {} // 设置线程块的初始偏移量

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {} // 使用默认值构造带有零线程块偏移的迭代器

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset); // 增加指针偏移量的方法
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset(make_Coord(0, 1)); // 根据 kAdvanceRank 增加瓦片偏移量
    else
      address_iterator_.add_tile_offset(make_Coord(1, 0)); // 根据 kAdvanceRank 增加瓦片偏移量

    return *this; // 返回迭代器自身
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this); // 复制当前迭代器
    operator++(); // 调用前置 ++ 操作符
    return self; // 返回原始迭代器
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
  address_iterator_.clear_mask(enable);
  // 调用地址迭代器的方法，清除当前的掩码位

CUTLASS_HOST_DEVICE
void set_residual_tile(bool enable) {
  address_iterator_.set_residual_tile(enable);
  // 设置地址迭代器的残余瓦片标志位
}

/// Clears the predicate set efficiently
CUTLASS_HOST_DEVICE
void enable_mask() {
  address_iterator_.enable_mask();
  // 高效地清除谓词集合
}

/// Sets the predicate mask, overriding value stored in predicate iterator
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  address_iterator_.set_mask(mask);
  // 设置谓词掩码，覆盖存储在谓词迭代器中的值
}

/// Gets the mask
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  address_iterator_.get_mask(mask);
  // 获取当前的掩码值
}

CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  load_with_byte_offset(
      frag, pointer_offset * sizeof_bits<Element>::value / 8);
  // 使用指针偏移加载片段，转换为字节偏移并调用相应方法
}

CUTLASS_DEVICE
void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
  AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        int idx = v +
            kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

        address_iterator_.set_iteration_index(idx);
        // 设置地址迭代器的迭代索引

        char const* byte_ptr =
            reinterpret_cast<char const*>(address_iterator_.get()) +
            byte_offset;
        // 获取字节指针，并根据字节偏移调整

        AccessType const* access_ptr =
            reinterpret_cast<AccessType const*>(byte_ptr);
        // 转换为访问类型指针

        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
            frag_ptr[idx], access_ptr, address_iterator_.valid());
        // 使用全局加载函数从内存加载数据到片段中

        ++address_iterator_;
        // 递增地址迭代器
      }
    }
  }
}

/// Loads a fragment from memory
CUTLASS_DEVICE
void load(Fragment& frag) {
  load_with_byte_offset(frag, 0);
  // 从内存加载片段，字节偏移为0
}

/// Store a fragment to memory
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  store_with_byte_offset(
      frag, pointer_offset * sizeof_bits<Element>::value / 8);
  // 使用指针偏移存储片段，转换为字节偏移并调用相应方法
}

/// Store a fragment to memory
CUTLASS_DEVICE
void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
  address_iterator_.set_iteration_index(0);
  // 设置地址迭代器的迭代索引为0

  AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
  // 转换片段指针类型为访问类型指针

  CUTLASS_PRAGMA_UNROLL
  // 循环展开指令
    // 对于每个 strided 维度中的迭代次数，进行循环
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      // 对于每个 contiguous 维度中的迭代次数，进行循环
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        // 对于每个向量中的访问次数，进行循环
        for (int v = 0; v < kAccessesPerVector; ++v) {
          // 计算索引，以便从一维地址中获取二维数据
          int idx = v +
              kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          // 计算指向要访问的地址的字节指针
          char* byte_ptr =
              reinterpret_cast<char*>(address_iterator_.get()) + byte_offset;
          // 计算可以进行访问的数据类型指针
          AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_ptr);

          // 如果地址迭代器仍然有效，则进行存储操作
          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          // 递增地址迭代器，移动到下一个地址
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  // 将一个片段存储到内存中
  CUTLASS_DEVICE
  void store(Fragment const& frag) {
    // 调用带有字节偏移的存储函数，默认偏移为0
    store_with_byte_offset(frag, 0);
  }
////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for affine rank 2
/// column-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                         // 模板参数：形状类型
    typename Element_,                       // 模板参数：元素类型
    int AdvanceRank,                         // 模板参数：前进秩
    typename ThreadMap_,                     // 模板参数：线程映射类型
    int AccessSize>                          // 模板参数：访问大小
class PredicatedTileIteratorResidualLast<
    Shape_,                                  // 形状类型
    Element_,                                // 元素类型
    layout::AffineRank2ColumnMajor,          // 布局类型：仿射秩2列主布局
    AdvanceRank,                             // 前进秩
    ThreadMap_,                              // 线程映射类型
    AccessSize,                              // 访问大小
    false> {                                 // 布尔模板参数：指示是否存在残余部分

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                      // 使用形状类型的别名
  using Element = Element_;                  // 使用元素类型的别名
  using Layout = layout::AffineRank2ColumnMajor;  // 使用列主仿射秩2布局的别名
  static int const kAdvanceRank = AdvanceRank;     // 静态常量：前进秩
  using ThreadMap = ThreadMap_;              // 使用线程映射类型的别名

  using Index = typename Layout::Index;      // 索引类型
  using LongIndex = typename Layout::LongIndex;  // 长索引类型

  using TensorRef = TensorRef<Element, Layout>;  // 张量引用类型
  using TensorView = TensorView<Element, Layout>;  // 张量视图类型
  using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型

  using Pointer = Element*;                  // 元素指针类型
  using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量元素指针类型

  // Map to the underlying AffineRankN<2> layout
  using UnderlyingIterator = PredicatedTileIteratorResidualLast<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,  // 基础迭代器类型，使用行和列形状的线性间隔形状
      Element,                               // 元素类型
      layout::AffineRankN<2>,                 // 基础迭代器的布局类型：仿射秩N=2
      (kAdvanceRank == 0 ? 0 : 1),            // 基础迭代器的前进秩
      ThreadMap,                             // 基础迭代器的线程映射类型
      AccessSize>;                           // 基础迭代器的访问大小

  using AccessType = typename UnderlyingIterator::AccessType;  // 访问类型

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 片段对象类型，用于加载或存储

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;  // 谓词向量类型，用于控制访问保护

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;  // 参数对象，基于基础迭代器的参数类型

   public:
    CUTLASS_HOST_DEVICE
    Params() {}                                 // 默认构造函数

    /// Construct the Params object given an AffineRankN<2> tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::AffineRankN<2>(layout.stride(0), layout.stride(1))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying AffineRankN<2> tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id, ///< ID of each participating thread
      TensorCoord const& threadblock_offset, ///< Initial offset of threadblock
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(
            params.params_,  // Initialize the underlying iterator with precomputed parameters
            pointer,  // Set the pointer to the start of the tensor
            layout::PitchLinearCoord(extent.row(), extent.column()),  // Set the pitch-linear coordinates for the tensor extent
            thread_id,  // Initialize the thread ID for the iterator
            layout::PitchLinearCoord(
                threadblock_offset.row(),  // Set the row offset of the threadblock
                threadblock_offset.column())  // Set the column offset of the threadblock
            ) {}

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)  // Call the constructor with zero threadblock offset
            ) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);  // Add the given pointer offset to the iterator
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    ++iterator_;  // Advance the iterator to the next tile
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this);
    operator++();  // Post-increment operator, advances the iterator
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);  // Clear the predicate mask of the iterator
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
  // 设置残余瓦片的使能状态
  iterator_.set_residual_tile(enable);
}

/// 高效地清除谓词集合
CUTLASS_HOST_DEVICE
void enable_mask() {
  // 调用迭代器的使能掩码函数
  iterator_.enable_mask();
}

/// 设置谓词掩码，覆盖谓词迭代器中存储的值
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  // 调用迭代器的设置掩码函数，传入给定的掩码
  iterator_.set_mask(mask);
}

/// 获取掩码值
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  // 调用迭代器的获取掩码函数，将结果写入给定的掩码引用
  iterator_.get_mask(mask);
}

/// 从内存加载一个片段
CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  // 调用迭代器的根据指针偏移加载函数，将加载的数据写入给定的片段
  iterator_.load_with_pointer_offset(frag, pointer_offset);
}

/// 从内存加载一个片段
CUTLASS_DEVICE
void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
  // 调用迭代器的根据字节偏移加载函数，将加载的数据写入给定的片段
  iterator_.load_with_byte_offset(frag, byte_offset);
}

/// 从内存加载一个片段
CUTLASS_DEVICE
void load(Fragment& frag) {
  // 调用带有指针偏移的加载函数，加载数据到给定的片段，偏移为0
  load_with_pointer_offset(frag, 0);
}

/// 将一个片段存储到内存
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  // 调用迭代器的根据指针偏移存储函数，将给定的片段存储到内存中
  iterator_.store_with_pointer_offset(frag, pointer_offset);
}

/// 将一个片段存储到内存
CUTLASS_DEVICE
void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
  // 调用迭代器的根据字节偏移存储函数，将给定的片段存储到内存中
  iterator_.store_with_byte_offset(frag, byte_offset);
}

/// 将一个片段存储到内存
CUTLASS_DEVICE
void store(Fragment const& frag) {
  // 调用带有指针偏移的存储函数，将给定的片段存储到内存中，偏移为0
  store_with_pointer_offset(frag, 0);
}
  // 结束类定义的分号

  ////////////////////////////////////////////////////////////////////////////////
  
  /// Specialization of PredicatedTileIteratorResidualLast for affine rank 2
  /// row-major data.
  ///
  /// Satisfies: ForwardTileIteratorConcept |
  ///            ReadableContiguousTileIteratorConcept |
  ///            WriteableContiguousTileIteratorConcept |
  ///            MaskedTileIteratorConcept
  ///
  template <
      typename Shape_,                 // 模板参数：迭代器的形状
      typename Element_,               // 模板参数：元素类型
      int AdvanceRank,                 // 模板参数：前进的秩
      typename ThreadMap_,             // 模板参数：线程映射类型
      int AccessSize>                  // 模板参数：访问大小
  class PredicatedTileIteratorResidualLast<
      Shape_,                          // 迭代器形状
      Element_,                        // 元素类型
      layout::AffineRank2RowMajor,     // 布局类型：二阶行主序
      AdvanceRank,                     // 前进的秩
      ThreadMap_,                      // 线程映射类型
      AccessSize,                      // 访问大小
      false> {                         // 是否支持遗留迭代器（默认为假）
   public:
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,  // 静态断言：AdvanceRank 必须为 0 或 1
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;                 // 使用迭代器形状别名
    using Element = Element_;             // 使用元素类型别名
    using Layout = layout::AffineRank2RowMajor;  // 使用行主序二阶布局别名
    static int const kAdvanceRank = AdvanceRank;  // 前进的秩常量
    using ThreadMap = ThreadMap_;         // 使用线程映射类型别名

    using Index = typename Layout::Index;          // 使用索引类型别名
    using LongIndex = typename Layout::LongIndex;  // 使用长索引类型别名

    using TensorRef = TensorRef<Element, Layout>;      // 张量引用类型别名
    using TensorView = TensorView<Element, Layout>;    // 张量视图类型别名
    using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型别名

    using Pointer = Element*;                         // 指针类型别名
    using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量指针类型别名

    // Map to the underlying AffineRankN<2> layout
    using UnderlyingIterator = PredicatedTileIteratorResidualLast<
        layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,  // 使用线性间距形状别名
        Element,                         // 元素类型
        layout::AffineRankN<2>,          // 二阶仿射秩布局类型
        (kAdvanceRank == 0 ? 1 : 0),     // 前进的秩为0时为1，否则为0
        ThreadMap,                       // 线程映射类型
        AccessSize>;                     // 访问大小

    using AccessType = typename UnderlyingIterator::AccessType;  // 访问类型别名

    /// Fragment object to be loaded or stored
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 片段对象类型别名

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;  // 断言向量类型别名

    /// Parameters object is precomputed state and is host-constructible
    class Params {
     private:
      friend PredicatedTileIteratorResidualLast;

      /// Parameters object
      typename UnderlyingIterator::Params params_;  // 参数对象

     public:
      CUTLASS_HOST_DEVICE
      Params() {}  // 构造函数：默认构造函数

      /// Construct the Params object given an AffineRankN<2> tensor's layout
      CUTLASS_HOST_DEVICE
  /// 构造函数，接受一个 Layout 对象作为参数，初始化 params_ 成员
  Params(Layout const& layout)
      : params_(layout::AffineRankN<2>(layout.stride(1), layout.stride(0))) {}

private:
  //
  // 数据成员
  //

  /// 下层 AffineRankN<2> 的瓦片迭代器
  UnderlyingIterator iterator_;

public:
  /// 从预先计算的状态、线程块偏移和线程 ID 构造一个 TileIterator
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< 预先计算的参数对象
      Pointer pointer,     ///< 指向张量起始位置的指针
      TensorCoord extent,  ///< 张量的尺寸
      int thread_id,       ///< 每个参与线程的 ID
      TensorCoord const& threadblock_offset, ///< 线程块的初始偏移
      int const* indices = nullptr ///< gather/scatter 索引，当前未支持在此特化中的 gather/scatter
      )
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.column(), extent.row()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row())) {}

  /// 构造一个 PredicatedTileIteratorResidualLast，使用零线程块偏移
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< 预先计算的参数对象
      Pointer pointer,     ///< 指向张量起始位置的指针
      TensorCoord extent,  ///< 张量的尺寸
      int thread_id        ///< 每个参与线程的 ID
      )
      : PredicatedTileIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// 在单位 Element 的偏移量上增加指针偏移量
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// 将迭代器前进到内存中的下一个瓦片
  ///
  /// 第一次调用此方法时，更新谓词，并将迭代器的内部指针恢复到第一个“稳态”瓦片。
  /// 后续调用轻量级，只需更新内部指针。
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// 将迭代器前进到内存中的下一个瓦片
  ///
  /// 第一次调用此方法时，更新谓词，并将迭代器的内部指针恢复到第一个“稳态”瓦片。
  /// 后续调用轻量级，只需更新内部指针。
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// 高效清除谓词集合
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
  /// 设置残留瓦片使能标志
  iterator_.set_residual_tile(enable);
}

/// 高效地清除谓词集合
CUTLASS_HOST_DEVICE
void enable_mask() {
  iterator_.enable_mask();
}

/// 设置谓词掩码，覆盖谓词迭代器中存储的值
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  iterator_.set_mask(mask);
}

/// 获取谓词掩码
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  iterator_.get_mask(mask);
}

/// 根据指针偏移量从内存中加载片段
CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  iterator_.load_with_pointer_offset(frag, pointer_offset);
}

/// 根据字节偏移量从内存中加载片段
CUTLASS_DEVICE
void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
  iterator_.load_with_byte_offset(frag, byte_offset);
}

/// 从内存中加载片段
CUTLASS_DEVICE
void load(Fragment& frag) {
  load_with_pointer_offset(frag, 0);
}

/// 根据指针偏移量将片段存储到内存中
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  iterator_.store_with_pointer_offset(frag, pointer_offset);
}

/// 根据字节偏移量将片段存储到内存中
CUTLASS_DEVICE
void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
  iterator_.store_with_byte_offset(frag, byte_offset);
}

/// 将片段存储到内存中
CUTLASS_DEVICE
void store(Fragment const& frag) {
  store_with_pointer_offset(frag, 0);
}
  };

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for interleaved data.
/// It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                                 // 模板参数：迭代器形状
    typename Element_,                               // 模板参数：元素类型
    int AdvanceRank,                                 // 模板参数：前进秩
    typename ThreadMap_,                             // 模板参数：线程映射
    int AccessSize,                                  // 模板参数：访问大小
    int InterleavedK>                                // 模板参数：交错因子
class PredicatedTileIteratorResidualLast<
    Shape_,                                          // 迭代器形状
    Element_,                                        // 元素类型
    layout::ColumnMajorInterleaved<InterleavedK>,    // 布局类型：列主交错布局
    AdvanceRank,                                     // 前进秩
    ThreadMap_,                                      // 线程映射
    AccessSize,
    false> {                                         // 布尔参数：不支持尾部残留

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,           // 静态断言：前进秩只能是0或1
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                               // 别名定义：形状
  using Element = Element_;                           // 别名定义：元素类型
  static int const kInterleavedK = InterleavedK;       // 常量定义：交错因子
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;  // 别名定义：布局类型
  static int const kAdvanceRank = AdvanceRank;         // 常量定义：前进秩
  using ThreadMap = ThreadMap_;                       // 别名定义：线程映射

  using Index = typename Layout::Index;                // 别名定义：索引类型
  using LongIndex = typename Layout::LongIndex;        // 别名定义：长索引类型

  using TensorRef = TensorRef<Element, Layout>;        // 别名定义：张量引用类型
  using TensorView = TensorView<Element, Layout>;      // 别名定义：张量视图类型
  using TensorCoord = typename Layout::TensorCoord;    // 别名定义：张量坐标类型

  using Pointer = Element*;                           // 别名定义：指针类型
  using NonConstPointer = typename platform::remove_const<Element>::type*;  // 别名定义：非常量指针类型

  using UnderlyingIterator = PredicatedTileIteratorResidualLast<
      layout::PitchLinearShape<
          Shape::kRow * kInterleavedK,
          Shape::kColumn / kInterleavedK>,            // 基础迭代器类型，适用于底层迭代器
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),                    // 根据前进秩选择使用0或1
      ThreadMap,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;  // 别名定义：访问类型

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 别名定义：片段类型

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;       // 别名定义：掩码类型

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;        // 成员变量：基础迭代器的参数对象

   public:
    CUTLASS_HOST_DEVICE
    Params() {}                                        // 默认构造函数

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}  // 构造函数：根据布局构造参数对象

    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(                        // 初始化成员变量 iterator_
            params.params_,              // 使用 params 对象中的 params_ 初始化 iterator_
            pointer,                     // 使用指向张量起始位置的指针初始化 iterator_
            layout::PitchLinearCoord(    // 使用 PitchLinearCoord 对象初始化 iterator_，用于表示张量的行和列的大小
                extent.row() * kInterleavedK,     // 计算扩展的行大小乘以 kInterleavedK
                extent.column() / kInterleavedK), // 计算扩展的列大小除以 kInterleavedK
            thread_id,                   // 使用线程 ID 初始化 iterator_
            layout::PitchLinearCoord(    // 使用 PitchLinearCoord 对象初始化 iterator_，用于表示线程块的行和列的大小
                threadblock_offset.row() * kInterleavedK,    // 计算线程块偏移的行大小乘以 kInterleavedK
                threadblock_offset.column() / kInterleavedK)) {}  // 计算线程块偏移的列大小除以 kInterleavedK

  /// Construct a PredicatedTileIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileIteratorResidualLast(    // 调用上面定义的构造函数，使用默认的线程块偏移 (0, 0)
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);  // 调用 iterator_ 的 add_pointer_offset 方法，添加指定的指针偏移量
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast& operator++() {
    ++iterator_;  // 调用 iterator_ 的前置递增操作符，使迭代器指向下一个内存中的瓦片
    return *this;  // 返回自身对象的引用
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorResidualLast operator++(int) {
    PredicatedTileIteratorResidualLast self(*this);  // 创建一个当前对象的副本 self
    operator++();  // 调用前置递增操作符以更新迭代器内部指针
    return self;   // 返回副本 self
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {

  /// Clears the predicate set efficiently by calling the clear_mask method of the iterator_ member variable
  iterator_.clear_mask(enable);
  }
  /// 清除迭代器的掩码
  iterator_.clear_mask(enable);
}

CUTLASS_HOST_DEVICE
void set_residual_tile(bool enable) {
  // 设置迭代器的剩余瓦片
  iterator_.set_residual_tile(enable);
}

/// 高效地清除谓词集合
CUTLASS_HOST_DEVICE
void enable_mask() {
  // 启用迭代器的掩码
  iterator_.enable_mask();
}

/// 设置谓词掩码，覆盖谓词迭代器中存储的值
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  // 设置迭代器的掩码
  iterator_.set_mask(mask);
}

/// 获取掩码
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  // 获取迭代器的掩码
  iterator_.get_mask(mask);
}

/// 从内存中加载一个片段
CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  // 使用指针偏移量从迭代器中加载一个片段
  iterator_.load_with_pointer_offset(frag, pointer_offset);
}

/// 从内存中加载一个片段
CUTLASS_DEVICE
void load(Fragment& frag) {
  // 使用默认的指针偏移量从迭代器中加载一个片段
  load_with_pointer_offset(frag, 0);
}

/// 将一个片段存储到内存中
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  // 使用指针偏移量将一个片段存储到迭代器中
  iterator_.store_with_pointer_offset(frag, pointer_offset);
}

/// 将一个片段存储到内存中
CUTLASS_DEVICE
void store(Fragment const& frag) {
  // 使用默认的指针偏移量将一个片段存储到迭代器中
  store_with_pointer_offset(frag, 0);
}
////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIteratorResidualLast for interleaved-32
/// data.  It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                             // 模板参数：迭代器形状
    typename Element_,                           // 模板参数：元素类型
    int AdvanceRank,                             // 模板参数：前进的秩
    typename ThreadMap_,                         // 模板参数：线程映射
    int AccessSize,                              // 模板参数：访问大小
    int InterleavedK>                            // 模板参数：交错因子
class PredicatedTileIteratorResidualLast<
    Shape_,                                      // 迭代器形状
    Element_,                                    // 元素类型
    layout::RowMajorInterleaved<InterleavedK>,    // 布局类型：行主序交错布局
    AdvanceRank,                                 // 前进的秩
    ThreadMap_,                                  // 线程映射
    AccessSize,
    false> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                          // 定义形状别名
  using Element = Element_;                      // 定义元素类型别名
  static int const kInterleavedK = InterleavedK;  // 设置交错因子常量
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;  // 定义布局类型
  static int const kAdvanceRank = AdvanceRank;    // 设置前进的秩常量
  using ThreadMap = ThreadMap_;                  // 定义线程映射类型

  using Index = typename Layout::Index;          // 定义索引类型
  using LongIndex = typename Layout::LongIndex;  // 定义长索引类型

  using TensorRef = TensorRef<Element, Layout>;  // 张量引用类型
  using TensorView = TensorView<Element, Layout>;  // 张量视图类型
  using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型

  using Pointer = Element*;                      // 指针类型
  using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量指针类型

  using UnderlyingIterator = PredicatedTileIteratorResidualLast<
      layout::PitchLinearShape<
          Shape::kColumn * kInterleavedK,
          Shape::kRow / kInterleavedK>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;  // 访问类型

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;  // 片段对象类型

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;  // 谓词向量类型

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;   // 内部参数对象

   public:
    CUTLASS_HOST_DEVICE
    Params() {}                                   // 默认构造函数

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}  // 根据布局构造参数对象

    CUTLASS_HOST_DEVICE
  Params(typename UnderlyingIterator::Params::Base const& base)
      : params_(base) {}
};

private:
//
// Data members
//

/// Underlying pitch-linear tile iterator
UnderlyingIterator iterator_;

public:
/// Constructs a TileIterator from its precomputed state, threadblock offset,
/// and thread ID
CUTLASS_HOST_DEVICE
PredicatedTileIteratorResidualLast(
    /// Precomputed parameters object
    Params const& params,
    /// Pointer to start of tensor
    Pointer pointer,
    /// Extent of tensor
    TensorCoord extent,
    /// ID of each participating thread
    int thread_id,
    /// Initial offset of threadblock
    TensorCoord const& threadblock_offset,
    int const* indices =
        nullptr ///< gather/scatter indices, note no support for
                ///< gather/scatter at this specialization
    )
    : iterator_(
          params.params_, // Initialize the underlying iterator with parameters
          pointer, // Start pointer of the tensor
          layout::PitchLinearCoord(
              extent.column() * kInterleavedK, // Compute the column extent multiplied by interleaved K
              extent.row() / kInterleavedK), // Compute the row extent divided by interleaved K
          thread_id, // ID of the participating thread
          layout::PitchLinearCoord(
              threadblock_offset.column() * kInterleavedK, // Compute the threadblock offset column multiplied by interleaved K
              threadblock_offset.row() / kInterleavedK)) {} // Compute the threadblock offset row divided by interleaved K

/// Construct a PredicatedTileIteratorResidualLast with zero threadblock
/// offset
CUTLASS_HOST_DEVICE
PredicatedTileIteratorResidualLast(
    Params const& params, ///< Precomputed parameters object
    Pointer pointer, ///< Pointer to start of tensor
    TensorCoord extent, ///< Extent of tensor
    int thread_id ///< ID of each participating thread
    )
    : PredicatedTileIteratorResidualLast(
          params, // Initialize with the given parameters
          pointer, // Start pointer of the tensor
          extent, // Extent of the tensor
          thread_id, // ID of the participating thread
          make_Coord(0, 0)) {} // Create a coordinate with zero offset for threadblock

/// Adds a pointer offset in units of Element
CUTLASS_HOST_DEVICE
void add_pointer_offset(LongIndex pointer_offset) {
  iterator_.add_pointer_offset(pointer_offset); // Add the given pointer offset to the iterator
}

/// Advances to the next tile in memory.
///
/// The first time this method is called, predicates are updated, and the
/// iterator's internal pointer is reverted to the first "steady state" tile.
/// Subsequent calls are lightweight and must only update the internal
/// pointer.
CUTLASS_HOST_DEVICE
PredicatedTileIteratorResidualLast& operator++() {
  ++iterator_; // Increment the iterator to advance to the next tile
  return *this;
}

/// Advances to the next tile in memory.
///
/// The first time this method is called, predicates are updated, and the
/// iterator's internal pointer is reverted to the first "steady state" tile.
/// Subsequent calls are lightweight and must only update the internal
/// pointer.
CUTLASS_HOST_DEVICE
PredicatedTileIteratorResidualLast operator++(int) {
  PredicatedTileIteratorResidualLast self(*this);
  operator++(); // Increment the iterator to advance to the next tile
  return self;
}

/// Clears the predicate set efficiently
CUTLASS_HOST_DEVICE
void clear_mask(bool enable = true) {
  // 清除迭代器的掩码（mask）
  iterator_.clear_mask(enable);
}

CUTLASS_HOST_DEVICE
void set_residual_tile(bool enable) {
  // 设置是否启用残余瓦片
  iterator_.set_residual_tile(enable);
}

/// 清除谓词集合的操作，高效实现
CUTLASS_HOST_DEVICE
void enable_mask() {
  // 启用迭代器的掩码
  iterator_.enable_mask();
}

/// 设置谓词掩码，覆盖谓词迭代器中存储的值
CUTLASS_HOST_DEVICE
void set_mask(Mask const& mask) {
  // 设置迭代器的掩码
  iterator_.set_mask(mask);
}

/// 获取掩码
CUTLASS_HOST_DEVICE
void get_mask(Mask& mask) {
  // 获取迭代器的掩码
  iterator_.get_mask(mask);
}

/// 从内存中加载一个片段
CUTLASS_DEVICE
void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
  // 使用指针偏移量从内存加载一个片段
  iterator_.load_with_pointer_offset(frag, pointer_offset);
}

/// 从内存中加载一个片段
CUTLASS_DEVICE
void load(Fragment& frag) {
  // 默认从内存加载一个片段（偏移量为0）
  load_with_pointer_offset(frag, 0);
}

/// 将一个片段存储到内存中
CUTLASS_DEVICE
void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
  // 使用指针偏移量将一个片段存储到内存中
  iterator_.store_with_pointer_offset(frag, pointer_offset);
}

/// 将一个片段存储到内存中
CUTLASS_DEVICE
void store(Fragment const& frag) {
  // 默认将一个片段存储到内存中（偏移量为0）
  store_with_pointer_offset(frag, 0);
}
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////


注释：


// 结束了当前的 namespace cutlass，表示 cutlass 命名空间的结束
};

////////////////////////////////////////////////////////////////////////////////

// 声明结束了 namespace cutlass 下的 transform 命名空间
} // namespace threadblock
// 声明结束了 namespace cutlass 下的 threadblock 命名空间
} // namespace transform
// 声明结束了 namespace cutlass，表示 cutlass 命名空间的完全结束
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////


这段代码是 C++ 中的命名空间声明和结尾标记。命名空间被用来组织代码，避免命名冲突，并提供模块化的代码结构。在上面的代码中：

- `};` 表示结束了当前的命名空间或类定义的结尾。
- `////////////////////////////////////////////////////////////////////////////////` 是一个注释行，可能用来标记不同部分的分割线或者注释段落。
- `} // namespace threadblock` 表示结束了 `threadblock` 命名空间的声明。
- `} // namespace transform` 表示结束了 `transform` 命名空间的声明，它是位于 `cutlass` 命名空间下的子命名空间。
- `} // namespace cutlass` 表示结束了 `cutlass` 命名空间的声明，这里是整个命名空间的结尾。

这些注释帮助阐明了每一行代码的作用，特别是在 C++ 中，命名空间声明和结束标记对于代码的结构和组织非常重要。
```