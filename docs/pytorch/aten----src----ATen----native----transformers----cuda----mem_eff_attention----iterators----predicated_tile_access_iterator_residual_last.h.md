# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\iterators\predicated_tile_access_iterator_residual_last.h`

```
/*!
    \file
    \brief Templates calculating the address and predicates to the load of tiles
    from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile
    this iterator visits maybe partial, then the remaining tiles are complete.
    So, we only need to compute the predicates twice, once before the first tile
    and once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be
    stored in registers, and integer addition is used to advance the pointer
    through memory.
*/

#pragma once

#include <cutlass/array.h>  //!< 引入 Cutlass 库中的数组实现
#include <cutlass/coord.h>  //!< 引入 Cutlass 库中的坐标操作
#include <cutlass/cutlass.h>  //!< 引入 Cutlass 库的核心功能
#include <cutlass/layout/matrix.h>  //!< 引入 Cutlass 库中的矩阵布局
#include <cutlass/layout/pitch_linear.h>  //!< 引入 Cutlass 库中的 pitch-linear 布局
#include <cutlass/matrix_shape.h>  //!< 引入 Cutlass 库中的矩阵形状定义
#include <cutlass/predicate_vector.h>  //!< 引入 Cutlass 库中的断言向量
#include <cutlass/tensor_ref.h>  //!< 引入 Cutlass 库中的张量引用
#include <cutlass/tensor_view.h>  //!< 引入 Cutlass 库中的张量视图
#include <cutlass/transform/threadblock/predicated_tile_access_iterator_params.h>  //!< 引入 Cutlass 库中的预测瓦片访问迭代器参数

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
  /// PredicatedTileAccessIteratorResidualLast
  ///
  template <
      typename Shape,
      typename Element,
      typename Layout,
      int AdvanceRank,
      typename ThreadMap,
      typename AccessType,
      bool Gather = false>
  class PredicatedTileAccessIteratorResidualLast;

  /// Specialization of PredicatedTileAccessIteratorResidualLast for pitch-linear
  /// data.
  ///
  template <
      typename Shape_,
      typename Element_,
      int AdvanceRank,
      typename ThreadMap_,
      typename AccessType_,
      bool Gather>
  class PredicatedTileAccessIteratorResidualLast<
      Shape_,
      Element_,
      layout::PitchLinear,
      AdvanceRank,
      ThreadMap_,
      AccessType_,
      Gather> {
   public:
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::PitchLinear;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
        Shape,
        Element,
        Layout,
        AdvanceRank,
        ThreadMap,
        AccessType>;

    static int const kAccessesPerVector =
        ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(
        !(ThreadMap::kElementsPerAccess % AccessType::kElements),
        "Vectors implied by the thread map must be divisible by the access type.");

    using Mask = typename UnderlyingPredicates::Mask;

    /// Uses a non-template class
    struct Params : PredicatedTileAccessIteratorParams {
      using Base = PredicatedTileAccessIteratorParams;

      // Default ctor
      CUTLASS_HOST_DEVICE
      Params() {}

      /// Construct the Params object given a pitch-linear tensor's layout
      CUTLASS_HOST_DEVICE
      Params(Layout const& layout)
          : Base(
                layout.stride(0),
                MakePredicatedTileAccessIteratorDesc<
                    Shape,
                    Element,
                    Layout,
                    kAdvanceRank,
                    ThreadMap>()()) {}

      CUTLASS_HOST_DEVICE
  Params(Base const& base) : Base(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  // 存储内部谓词
  UnderlyingPredicates the_predicates;
  // 剩余瓦片掩码
  Mask residual_tile_mask;

  /// Parameters object with precomputed internal state
  // 预计算内部状态的参数对象
  Params params_;

  /// Internal pointer to first access of tile
  // 内部指针，用于快速访问瓦片的起始位置
  BytePointer pointer_;

  /// Below is used when Gather is turned on.  We need to record strided_offset
  /// and contiguous_offset separated to compute the offset by using
  ///
  /// offset = contiguous_offset + indices[strided_offset]
  ///

  /// Gather indices
  // Gather 操作的索引数组
  int const* indices_;

  // Gather 偏移的跨度索引
  Index gather_offset_strided;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  // 根据每个线程内部跟踪的偏移量计算谓词
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent, is_steady_state);
  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  // 从预计算状态、线程块偏移和线程ID构造TileIterator
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        the_predicates(extent),
        indices_(indices) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    the_predicates.get_mask(residual_tile_mask);

    // Working around a weird compiler bug happening on P100 for the backward.
    // I've seen together: the_predicates.predicates_[0] = 14 (instead of 15)
    // residual_tile_mask[0] = 15 (correct)
    //
    // Adding prints when the value is calculated (in `compute_predicates_`)
    // sometimes removes the bug. The consequence is that we skip some
    // element of a tensor, leading to wrong results
    // Setting `compute_predicates_`'s second argument (`is_steady_state`) to
    // true also seems to get rid of the bug - at the cost of twice as many
    // comparisons.

    // 解决在P100上发生的反向编译器奇怪错误的方法。
    // 我观察到：the_predicates.predicates_[0] = 14（而不是15）
    // residual_tile_mask[0] = 15（正确）
    //
    // 在计算值时添加打印语句（在`compute_predicates_`中）有时可以消除此错误。
    // 结果是我们会跳过张量的某些元素，导致错误的结果。
    // 将`compute_predicates_`的第二个参数（`is_steady_state`）设置为true似乎也能消除此错误，但会增加两倍的比较次数。
  }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    // 如果不是 CUDA 架构或者是 CUDA 架构版本大于等于 700，则不需要编译器错误修复工作
    constexpr bool kWorkAroundCompilerBug = false;
#else
    // 否则需要编译器错误修复工作
    constexpr bool kWorkAroundCompilerBug = true;
#endif

// 使用计算出的谓词，执行谓词计算，考虑编译器错误修复工作
the_predicates.compute_predicates_(extent, true && !kWorkAroundCompilerBug);

// 更新内部指针
Layout layout(params_.stride_);

if (!Gather) {
  // 如果不是 Gather 模式，则添加指针偏移量
  add_pointer_offset(layout(the_predicates.thread_offset_));
} else {
  // 如果是 Gather 模式，则根据线程偏移生成连续偏移量，再添加指针偏移量
  gather_offset_strided = the_predicates.thread_offset_.strided();
  add_pointer_offset(
      layout(make_Coord(the_predicates.thread_offset_.contiguous(), 0)));
}
}

/// 使用零线程块偏移量构造 PredicatedTileAccessIteratorResidualLast
CUTLASS_HOST_DEVICE
PredicatedTileAccessIteratorResidualLast(
    /// 预先计算的参数对象
    Params const& params,
    /// 指向张量开始位置的指针
    Pointer pointer,
    /// 张量的尺寸
    TensorCoord extent,
    ///< 每个参与的线程的ID
    int thread_id)
    : PredicatedTileAccessIteratorResidualLast(
          params,
          pointer,
          extent,
          thread_id,
          make_Coord(0, 0)) {}

/// 重写内部迭代索引
CUTLASS_HOST_DEVICE
void set_iteration_index(int index) {
  the_predicates.set_iteration_index(index);
}

CUTLASS_HOST_DEVICE
void set_residual_tile(bool is_residual_tile) {
  if (is_residual_tile) {
    // 如果是残余瓦片，则设置掩码为残余瓦片掩码
    the_predicates.set_mask(residual_tile_mask);
  }
}

/// 以 Element 单位添加指针偏移量
CUTLASS_HOST_DEVICE
void add_pointer_offset(LongIndex pointer_offset) {
  pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
}

/// 以整个瓦片为单位在矩阵的逻辑维度上推进迭代器
CUTLASS_DEVICE
void add_tile_offset(TensorCoord const& tile_offset) {
  if (!Gather) {
    if (kAdvanceRank) {
      // 如果是前进等级，则按步进偏移量增加指针
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
      pointer_ += Shape::kContiguous * tile_offset.contiguous();
    } else {
      // 否则按连续偏移量增加指针
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
      pointer_ += Shape::kStrided * tile_offset.strided();
    }
  } else {
    // 如果是 Gather 模式，则按连续偏移量增加指针，并更新跨步偏移量
    add_pointer_offset(Shape::kContiguous * tile_offset.contiguous());
    gather_offset_strided += Shape::kStrided * tile_offset.strided();
  }
}

/// 返回指针
CUTLASS_HOST_DEVICE
AccessType* get() const {
    // 如果 Gather 标志为真，则执行以下操作
    if (Gather) {
      // 断言索引数组非空
      assert(indices_);

      // 如果当前迭代无效，则返回空指针
      if (!valid()) {
        return nullptr;
      }

      // 计算连续偏移量，基于迭代连续数和元素大小
      LongIndex contiguous_offset = the_predicates.iteration_contiguous_ *
              (ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value /
               8) +
          the_predicates.iteration_vector_;

      // 计算跨步索引，基于偏移和迭代跨步数
      int strided_index = gather_offset_strided +
          the_predicates.iteration_strided_ * ThreadMap::Delta::kStrided;

      // 计算跨步偏移量，基于索引、步长和元素大小
      LongIndex strided_offset = indices_[strided_index] *
          LongIndex(params_.stride_) * sizeof_bits<Element>::value / 8;

      // 返回指向存储区域的访问类型指针
      return reinterpret_cast<AccessType*>(
          pointer_ + contiguous_offset + strided_offset);
    }

    // 如果 Gather 标志为假，则执行以下操作
    return reinterpret_cast<AccessType*>(
               pointer_ +
               the_predicates.iteration_contiguous_ *
                   (ThreadMap::Delta::kContiguous *
                    sizeof_bits<Element>::value) /
                   8) +
        the_predicates.iteration_vector_;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    // 调用谓词迭代器的自增运算符
    the_predicates.operator++();

    // 增加向量迭代计数
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    // 重置向量迭代计数，增加连续迭代计数
    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    // 如果连续迭代计数未达到预定值，则返回自身
    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // 进入此分支表示连续迭代计数已达预定值
    // 重置连续迭代计数，增加跨步迭代计数
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    // 如果跨步迭代计数未达到预定值，则根据 Gather 标志决定是否更新指针
    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      if (!Gather) {
        pointer_ += params_.inc_strided_;
      }

      return *this;
    }

    // 进入此分支表示跨步迭代计数已达预定值
    // 重置跨步迭代计数
    the_predicates.iteration_strided_ = 0;

    // 如果 Gather 标志为假，则根据参数更新指针位置
    if (!Gather) {
      // 前进到下一个瓦片
      pointer_ += params_.inc_next_;

      // 返回到起始瓦片位置，这些减法和后续的整数加法都会被编译器优化
      pointer_ -= params_.inc_advance_;
    }

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    // 创建当前对象的副本
    PredicatedTileAccessIteratorResidualLast self(*this);
    // 调用前置递增运算符
    operator++();
    // 返回副本
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    // 清除谓词集合
    the_predicates.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    // 启用谓词集合
    the_predicates.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    // 设置谓词掩码，覆盖谓词迭代器中的值
    the_predicates.set_mask(mask);
  }
    /// 设置掩码给谓词对象
    the_predicates.set_mask(mask);
    
    
    
    /// 获取掩码
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
      // 从谓词对象获取掩码
      the_predicates.get_mask(mask);
    }
    
    
    
    /// 返回访问是否有效
    CUTLASS_HOST_DEVICE
    bool valid() const {
      // 返回谓词对象的有效性状态
      return the_predicates.valid();
    }
/// Specialization of PredicatedTileAccessIteratorResidualLast for column-major
/// data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,               // 模板参数：迭代器形状
    typename Element_,             // 模板参数：元素类型
    int AdvanceRank,               // 模板参数：迭代器前进等级
    typename ThreadMap_,           // 模板参数：线程映射类型
    typename AccessType_,          // 模板参数：访问类型
    bool Gather>                   // 模板参数：是否执行gather操作
class PredicatedTileAccessIteratorResidualLast<
    Shape_,                         // 迭代器形状类型
    Element_,                       // 元素类型
    layout::ColumnMajor,            // 布局类型：列主序
    AdvanceRank,                    // 迭代器前进等级
    ThreadMap_,                     // 线程映射类型
    AccessType_,                    // 访问类型
    Gather> {                       // 是否执行gather操作

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                    // 迭代器形状别名
  using Element = Element_;                // 元素类型别名
  using Layout = layout::ColumnMajor;      // 布局类型别名：列主序
  static int const kAdvanceRank = AdvanceRank;    // 常量：迭代器前进等级
  using ThreadMap = ThreadMap_;            // 线程映射类型别名
  using AccessType = AccessType_;          // 访问类型别名

  using Index = typename Layout::Index;            // 索引类型别名
  using LongIndex = typename Layout::LongIndex;    // 长索引类型别名

  using TensorRef = TensorRef<Element, Layout>;    // 引用型张量类型别名
  using TensorView = TensorView<Element, Layout>;  // 视图型张量类型别名
  using TensorCoord = typename Layout::TensorCoord;    // 张量坐标类型别名

  using Pointer = Element*;    // 指针类型别名
  using NonConstPointer = typename platform::remove_const<Element>::type*;    // 非常量指针类型别名

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,    // 基础迭代器类型别名
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessType,
      Gather>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;    // 谓词向量类型别名

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;    // 每个向量的访问次数常量

  /// Parameters object is precomputed state and is host-constructible
  class Params {    // 参数类定义

   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;    // 基础迭代器参数对象

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}    // 默认构造函数

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))){};    // 根据pitch-linear张量的布局构造参数对象

    /// Construct the Params object given a pitch-linear tensor's layout
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
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : iterator_(  // 初始化成员变量 iterator_，使用给定的参数
            params.params_,  // 使用 params 参数中的 params_ 字段
            pointer,  // 传入指向张量起始位置的指针
            layout::PitchLinearCoord(extent.row(), extent.column()),  // 使用行和列构建 PitchLinearCoord 对象作为张量的大小
            thread_id,  // 传入每个参与线程的线程 ID
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column()),  // 使用 threadblock_offset 构建 PitchLinearCoord 对象作为线程块的偏移量
            indices)  // 传入 gather/scatter 索引数组（此处不支持 gather/scatter 操作）
      {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}  // 调用上述构造函数，传入零偏移量作为 threadblock_offset

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);  // 调用 iterator_ 对象的 set_iteration_index 方法，设置迭代索引
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);  // 调用 iterator_ 对象的 set_residual_tile 方法，设置是否启用残余瓦片
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);  // 调用 iterator_ 对象的 add_pointer_offset 方法，添加指针偏移量
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});  // 调用 iterator_ 对象的 add_tile_offset 方法，添加瓦片偏移量
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());  // 调用 iterator_ 对象的 get 方法，返回对应类型的指针
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;  // 递增 iterator_ 对象，使其指向内存中的下一个瓦片
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    // Create a copy of the current iterator state
    PredicatedTileAccessIteratorResidualLast self(*this);
    // Increment the iterator to point to the next tile in memory
    operator++();
    // Return the copy of the iterator state before incrementing
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    // Clear the predicate mask of the iterator
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    // Enable the predicate mask of the iterator
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    // Set the predicate mask of the iterator to the provided mask
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    // Retrieve the current predicate mask of the iterator
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    // Check if the iterator's current access is valid
    return iterator_.valid();
  }
  };

  ////////////////////////////////////////////////////////////////////////////////

  /// Specialization of PredicatedTileAccessIteratorResidualLast for row-major
  /// data.
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
      typename AccessType_,
      bool Gather>
  class PredicatedTileAccessIteratorResidualLast<
      Shape_,
      Element_,
      layout::RowMajor,
      AdvanceRank,
      ThreadMap_,
      AccessType_,
      Gather> {
   public:
    // Asserts that AdvanceRank is either 0 (contiguous) or 1 (strided)
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,
        "Specialization for pitch-linear iterator may only advance along the "
        "contiguous (rank=0) or strided (rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    // Defines the underlying iterator type for pitch-linear access
    using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
        layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
        Element,
        layout::PitchLinear,
        (kAdvanceRank == 0 ? 1 : 0),
        ThreadMap,
        AccessType,
        Gather>;

    static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
     private:
      friend PredicatedTileAccessIteratorResidualLast;

      /// Parameters object for the underlying iterator
      typename UnderlyingIterator::Params params_;

     public:
      /// Default constructor
      CUTLASS_HOST_DEVICE
      Params() {}

      /// Construct the Params object given a pitch-linear tensor's layout
      CUTLASS_HOST_DEVICE
      Params(Layout const& layout)
          : params_(layout::PitchLinear(layout.stride(0))) {};
      
      /// Construct the Params object given a pitch-linear tensor's layout
      CUTLASS_HOST_DEVICE
  Params(typename UnderlyingIterator::Params::Base const& base)
      : params_(base) {}
  };


  // 构造函数，接受基础迭代器的参数对象，并将其作为成员变量保存
  Params(typename UnderlyingIterator::Params::Base const& base)
      : params_(base) {}



 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;


 private:
  //
  // 数据成员
  //

  /// 底层的基于pitch-linear的瓦片迭代器
  UnderlyingIterator iterator_;



 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      /// Gather indices
      int const* indices = nullptr)
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.column(), extent.row()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row()),
            indices) {}


  /// 根据预计算的状态、线程块偏移和线程ID构造TileIterator

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< 预计算的参数对象
      Params const& params,
      ///< 指向张量起始位置的指针
      Pointer pointer,
      ///< 张量的范围
      TensorCoord extent,
      ///< 每个参与线程的ID
      int thread_id,
      ///< 线程块的初始偏移量
      TensorCoord const& threadblock_offset,
      /// Gather索引
      int const* indices = nullptr)
      : iterator_(
            params.params_, // 使用params的参数对象初始化底层迭代器
            pointer, // 设置迭代器的指针位置
            layout::PitchLinearCoord(extent.column(), extent.row()), // 设置迭代器的布局
            thread_id, // 设置参与的线程ID
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row()), // 设置线程块的偏移量
            indices) {} // 设置gather索引



  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}


  /// 使用零线程块偏移量构造PredicatedTileAccessIteratorResidualLast

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< 预计算的参数对象
      Pointer pointer, ///< 指向张量起始位置的指针
      TensorCoord extent, ///< 张量的范围
      int thread_id ///< 每个参与线程的ID
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {} // 调用具有零偏移量的构造函数



  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }


  /// 重写内部迭代索引

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index); // 调用底层迭代器的设置迭代索引方法
  }



  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }


  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable); // 调用底层迭代器的设置残余瓦片方法
  }



  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }


  /// 添加一个以元素为单位的指针偏移量

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset); // 调用底层迭代器的添加指针偏移量方法
  }



  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }


  /// 以整个瓦片为单位，在矩阵的逻辑维度上推进迭代器

  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()}); // 调用底层迭代器的添加瓦片偏移量方法
  }



  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }


  /// 返回一个指针

  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get()); // 返回底层迭代器的指针，类型转换为AccessType*
  }



  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;


  /// 推进到内存中的下一个瓦片。

  ///
  /// 第一次调用此方法时，更新谓词，并将迭代器的内部指针恢复到第一个“稳定状态”瓦片。
  /// 后续调用则轻量级，只需更新内部指针。
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_; // 增加底层迭代器的迭代操作
    // 返回当前对象的引用
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  // 后置自增运算符重载，返回当前对象的副本并将自身递增
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    // 保存当前对象的副本
    PredicatedTileAccessIteratorResidualLast self(*this);
    // 调用前置自增运算符，更新对象的内部指针
    operator++();
    // 返回保存的副本
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  // 清除谓词集合，可选择是否启用
  void clear_mask(bool enable = true) {
    // 调用成员变量 iterator_ 的 clear_mask 方法
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  // 启用谓词集合
  void enable_mask() {
    // 调用成员变量 iterator_ 的 enable_mask 方法
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  // 设置谓词掩码，覆盖谓词迭代器中的值
  void set_mask(Mask const& mask) {
    // 调用成员变量 iterator_ 的 set_mask 方法
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  // 获取谓词掩码
  void get_mask(Mask& mask) {
    // 调用成员变量 iterator_ 的 get_mask 方法
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  // 返回访问是否有效
  bool valid() {
    // 调用成员变量 iterator_ 的 valid 方法，返回结果
    return iterator_.valid();
  }
////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank 2
/// data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                // 模板参数：迭代器形状类型
    typename Element_,              // 模板参数：元素类型
    int AdvanceRank,                // 模板参数：前进的维度等级
    typename ThreadMap_,            // 模板参数：线程映射类型
    typename AccessType_>           // 模板参数：访问类型
class PredicatedTileAccessIteratorResidualLast<
    Shape_,                         // 迭代器形状类型
    Element_,                       // 元素类型
    layout::AffineRankN<2>,         // 布局类型：二维仿射秩
    AdvanceRank,                    // 前进的维度等级
    ThreadMap_,                     // 线程映射类型
    AccessType_,                    // 访问类型
    false> {                        // 是否为常量迭代器的标志

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                    // 迭代器形状的别名
  using Element = Element_;                // 元素类型的别名
  using Layout = layout::AffineRankN<2>;   // 布局类型的别名：二维仿射秩
  static int const kAdvanceRank = AdvanceRank;   // 前进的维度等级的常量
  using ThreadMap = ThreadMap_;            // 线程映射类型的别名
  using AccessType = AccessType_;          // 访问类型的别名

  using Index = typename Layout::Index;            // 索引类型
  using LongIndex = typename Layout::LongIndex;    // 长索引类型

  using TensorRef = TensorRef<Element, Layout>;    // 引用张量类型
  using TensorView = TensorView<Element, Layout>;  // 视图张量类型
  using TensorCoord = typename Layout::TensorCoord;    // 张量坐标类型

  using Pointer = Element*;                        // 元素指针类型
  using NonConstPointer = typename platform::remove_const<Element>::type*;   // 非常量元素指针类型

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape, Element, layout::PitchLinear, AdvanceRank, ThreadMap, AccessType>;    // 底层谓词迭代器类型

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;   // 每向量访问的访问次数

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");   // 确保线程映射所暗示的向量数能被访问类型整除

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingPredicates::Mask;    // 谓词向量存储掩码来保护访问的类型

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PredicatedTileAccessIteratorResidualLast;   // 参数类是预先计算的状态，并且可以在主机上构造

   private:
    /// stride of pitch-linear layout (units of Element)
    Coord<Layout::kStrideRank, Layout::LongIndex> stride_;   // 俯仰线性布局的步幅（元素单位）

    /// amount (in byte) to increment pointer to move to next access along
    /// contiguous dimension
    LongIndex inc_contiguous_;    // 增加指针的字节数，以移动到沿着连续维度的下一个访问点

    /// amount (in byte) to increment pointer from first access of current
    /// contiguous dimension to first access of next one.
    LongIndex inc_strided_;   // 增加指针的字节数，从当前连续维度的第一个访问到下一个维度的第一个访问

    /// amount (in byte) to increment pointer from last access of current
    /// contiguous dimension to first access of next one.
    LongIndex inc_next_strided_;    // 增加指针的字节数，从当前连续维度的最后一个访问到下一个维度的第一个访问

    /// amount (in byte) to increment pointer from last access to first access
    /// of next tile
    LongIndex inc_next_;    // 增加指针的字节数，从最后一个访问到下一个瓦片的第一个访问

    /// amount (in byte) to increment pointer from first access of current tile
    /// to first access of next tile
    LongIndex inc_advance_;

   public:
    // Default ctor
    CUTLASS_HOST_DEVICE
    Params()
        : stride_(0),                            // 初始化成员变量 stride_ 为 0
          inc_contiguous_(0),                    // 初始化成员变量 inc_contiguous_ 为 0
          inc_strided_(0),                       // 初始化成员变量 inc_strided_ 为 0
          inc_next_(0),                          // 初始化成员变量 inc_next_ 为 0
          inc_advance_(0) {}                     // 初始化成员变量 inc_advance_ 为 0

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : stride_({layout.stride(0), layout.stride(1)}) {   // 根据给定的 pitch-linear 张量布局构造 Params 对象
      inc_contiguous_ =
          (LongIndex(stride_[0]) * ThreadMap::Delta::kContiguous) *
          sizeof_bits<Element>::value / 8;                 // 计算连续访问增量的字节数

      inc_strided_ = (LongIndex(stride_[1]) * ThreadMap::Delta::kStrided) *
          sizeof_bits<Element>::value / 8;                 // 计算跨步访问增量的字节数

      inc_next_strided_ = inc_strided_ -
          LongIndex(ThreadMap::Iterations::kContiguous - 1) * inc_contiguous_;  // 计算下一个跨步访问的增量字节数

      if (kAdvanceRank) {
        // advance along strided dimension
        inc_advance_ = Shape::kStrided * LongIndex(stride_[1]) *
            sizeof_bits<Element>::value / 8;               // 根据条件计算下一个访问步长的增量字节数（沿跨步维度）
      } else {
        // advance along contiguous dimension
        inc_advance_ =
            Shape::kContiguous * stride_[0] * sizeof_bits<Element>::value / 8;  // 根据条件计算下一个访问步长的增量字节数（沿连续维度）
      }

      inc_next_ = inc_advance_ -
          LongIndex(ThreadMap::Iterations::kContiguous - 1) * inc_contiguous_ -
          LongIndex(ThreadMap::Iterations::kStrided - 1) * inc_strided_;  // 计算下一个访问的总增量字节数
    };
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params params_;                            // 存储预计算内部状态的参数对象

  /// Internal pointer to first access of tile
  BytePointer pointer_;                      // 指向瓦片第一个访问的内部指针

  UnderlyingPredicates the_predicates;
  Mask residual_tile_mask;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent, is_steady_state);  // 根据线程内偏移量计算谓词
  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      : params_(params),                                  // 使用预计算状态、线程块偏移量和线程 ID 构造 TileIterator
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),       // 初始化指向张量开始的指针
        the_predicates(extent) {
    the_predicates.set_predicates(thread_id, threadblock_offset);  // 设置谓词以便于线程块偏移量

    // update internal pointers
  /// 构造一个 PredicatedTileAccessIteratorResidualLast 对象，带有零线程块偏移量
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< 预先计算的参数对象
      Pointer pointer, ///< 指向张量起始位置的指针
      TensorCoord extent, ///< 张量的尺寸
      int thread_id ///< 参与的每个线程的ID
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// 覆盖内部迭代索引
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool is_residual_tile) {
    if (is_residual_tile) {
      the_predicates.set_mask(residual_tile_mask);
    }
  }

  /// 在 Element 单位中添加指针偏移量
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
  }

  /// 在整个矩阵的逻辑维度中以整个瓦片为单位前进迭代器
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    if (kAdvanceRank) {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset[1]);
      pointer_ += Shape::kContiguous * tile_offset[0];
    } else {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset[0]);
      pointer_ += Shape::kStrided * tile_offset[1];
    }
  }

  /// 返回一个指针
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(pointer_) +
        the_predicates.iteration_vector_;
  }

  /// 前进到内存中的下一个瓦片
  ///
  /// 第一次调用此方法时，更新谓词，并将迭代器的内部指针恢复到第一个“稳态”瓦片。
  /// 后续调用轻量且仅需更新内部指针。
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    the_predicates.operator++();
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      pointer_ += params_.inc_contiguous_;
      return *this;
    }

    // 仅在 (iteration_contiguous_ == ThreadMap::Iteration::kContiguous) 时进入此处
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += params_.inc_next_strided_;
      return *this;
    }
    // 如果 iteration_stride_ == ThreadMap::Iteration::kStrided，则进入这里，表示进入下一个瓦片。
    the_predicates.iteration_strided_ = 0;

    // 移动到下一个瓦片
    pointer_ += params_.inc_next_;

    // 现在返回到起始瓦片 - 如果迭代器随后被推进，这个减法以及随后的整数加法都会被编译器省略。
    pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// 移动到内存中的下一个瓦片。
  ///
  /// 第一次调用此方法时，更新谓词，并将迭代器的内部指针恢复到第一个“稳定状态”瓦片。
  /// 后续调用轻量级，仅需更新内部指针。
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    // 创建自身的副本
    PredicatedTileAccessIteratorResidualLast self(*this);
    // 调用前缀自增运算符来执行移动到下一个瓦片的操作
    operator++();
    // 返回之前的自身副本
    return self;
  }

  /// 高效地清除谓词集合
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    // 清除谓词集合
    the_predicates.clear_mask(enable);
  }

  /// 高效地启用谓词集合
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    // 启用谓词集合
    the_predicates.enable_mask();
  }

  /// 设置谓词掩码，覆盖谓词迭代器中存储的值
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    // 设置谓词掩码
    the_predicates.set_mask(mask);
  }

  /// 获取谓词掩码
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    // 获取谓词掩码
    the_predicates.get_mask(mask);
  }

  /// 返回访问是否有效
  CUTLASS_HOST_DEVICE
  bool valid() {
    // 返回访问是否有效
    return the_predicates.valid();
  }
  };

  ////////////////////////////////////////////////////////////////////////////////

  /// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank 2
  /// column-major data.
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
      typename AccessType_>
  class PredicatedTileAccessIteratorResidualLast<
      Shape_,
      Element_,
      layout::AffineRank2ColumnMajor,
      AdvanceRank,
      ThreadMap_,
      AccessType_,
      false> {
   public:
    // Assertion to ensure AdvanceRank is either 0 or 1
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    // Type aliases for the iterator class
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::AffineRank2ColumnMajor;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using AccessType = AccessType_;

    // Type aliases for indices used in the iterator
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    // Type aliases for tensor references and views
    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    // Pointer types for element access
    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    // Alias to the underlying iterator specialized for AffineRankN<2>
    using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
        layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
        Element,
        layout::AffineRankN<2>,
        (kAdvanceRank == 0 ? 0 : 1),
        ThreadMap,
        AccessType>;

    // Constant for the number of accesses per vector
    static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
     private:
      friend PredicatedTileAccessIteratorResidualLast;

      /// Parameters object for the underlying iterator
      typename UnderlyingIterator::Params params_;

     public:
      /// Default constructor
      CUTLASS_HOST_DEVICE
      Params() {}

      /// Construct the Params object given an AffineRankN<2> tensor's layout
      CUTLASS_HOST_DEVICE
  Params(Layout const& layout)
      : params_(layout::AffineRankN<2>(layout.stride(0), layout.stride(1))){};
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
PredicatedTileAccessIteratorResidualLast(
    ///< Precomputed parameters object
    Params const& params,
    ///< Pointer to start of tensor
    Pointer pointer,
    ///< Extent of tensor
    TensorCoord extent,
    ///< ID of each participating thread
    int thread_id,
    ///< Initial offset of threadblock
    TensorCoord const& threadblock_offset,
    int const* indices =
        nullptr ///< gather/scatter indices, note no support for
                ///< gather/scatter at this specialization
    )
    : iterator_(
          // 初始化迭代器，使用预先计算的参数对象、指向张量起始位置的指针、张量的大小范围、线程 ID、以及线程块的初始偏移量来构造
          params.params_,
          pointer,
          layout::PitchLinearCoord(extent.row(), extent.column()),
          thread_id,
          layout::PitchLinearCoord(
              threadblock_offset.row(),
              threadblock_offset.column())) {}

/// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
/// offset
CUTLASS_HOST_DEVICE
PredicatedTileAccessIteratorResidualLast(
    Params const& params, ///< Precomputed parameters object
    Pointer pointer,      ///< Pointer to start of tensor
    TensorCoord extent,   ///< Extent of tensor
    int thread_id         ///< ID of each participating thread
    )
    : PredicatedTileAccessIteratorResidualLast(
          // 使用给定的参数对象、指向张量起始位置的指针、张量的大小范围和线程 ID 构造一个带有零线程块偏移的 PredicatedTileAccessIteratorResidualLast
          params,
          pointer,
          extent,
          thread_id,
          make_Coord(0, 0)) {}

/// Overrides the internal iteration index
CUTLASS_HOST_DEVICE
void set_iteration_index(int index) {
  // 调用迭代器的方法，设置内部迭代索引
  iterator_.set_iteration_index(index);
}

CUTLASS_HOST_DEVICE
void set_residual_tile(bool enable) {
  // 调用迭代器的方法，设置是否启用残余瓦片
  iterator_.set_residual_tile(enable);
}

/// Adds a pointer offset in units of Element
CUTLASS_HOST_DEVICE
void add_pointer_offset(LongIndex pointer_offset) {
  // 调用迭代器的方法，添加指针偏移量（单位为元素）
  iterator_.add_pointer_offset(pointer_offset);
}

/// Advances an iterator along logical dimensions of matrix in units of whole
/// tiles
CUTLASS_HOST_DEVICE
void add_tile_offset(TensorCoord const& tile_offset) {
  // 调用迭代器的方法，添加瓦片偏移量（单位为整个瓦片）
  iterator_.add_tile_offset(
      make_Coord(tile_offset.row(), tile_offset.column()));
}

/// Returns a pointer
CUTLASS_HOST_DEVICE
AccessType* get() const {
  // 返回迭代器当前位置的指针类型
  return reinterpret_cast<AccessType*>(iterator_.get());
}

/// Advances to the next tile in memory.
///
/// The first time this method is called, predicates are updated, and the
/// iterator's internal pointer is reverted to the first "steady state" tile.
/// Subsequent calls are lightweight and must only update the internal
/// pointer.
CUTLASS_HOST_DEVICE
PredicatedTileAccessIteratorResidualLast& operator++() {
  // 迭代器自增操作符重载，用于将迭代器指向下一个内存中的瓦片
  ++iterator_;
    /// 返回对当前对象的引用，用于支持链式操作。
    ///
    /// 此方法返回对象本身的引用，以支持链式操作。
    /// 例如，可以使用 `obj.operator++().some_method()` 这样的方式来连续调用对象的方法。
    return *this;
    }
    
    /// 将迭代器指向下一个内存中的瓦片。
    ///
    /// 第一次调用此方法时，更新谓词并将迭代器的内部指针恢复到第一个“稳定状态”的瓦片。
    /// 后续调用则是轻量级的，只需更新内部指针即可。
    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIteratorResidualLast operator++(int) {
      // 创建当前对象的副本
      PredicatedTileAccessIteratorResidualLast self(*this);
      // 前缀递增操作符
      operator++();
      // 返回先前创建的副本
      return self;
    }
    
    /// 高效地清除谓词集合。
    ///
    /// 通过调用底层迭代器的 `clear_mask` 方法来实现。
    CUTLASS_HOST_DEVICE
    void clear_mask(bool enable = true) {
      iterator_.clear_mask(enable);
    }
    
    /// 高效地启用谓词集合。
    ///
    /// 通过调用底层迭代器的 `enable_mask` 方法来实现。
    CUTLASS_HOST_DEVICE
    void enable_mask() {
      iterator_.enable_mask();
    }
    
    /// 设置谓词掩码，覆盖存储在谓词迭代器中的值。
    ///
    /// 通过调用底层迭代器的 `set_mask` 方法来实现。
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
      iterator_.set_mask(mask);
    }
    
    /// 获取当前谓词掩码。
    ///
    /// 通过调用底层迭代器的 `get_mask` 方法来实现。
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
      iterator_.get_mask(mask);
    }
    
    /// 检查访问是否有效。
    ///
    /// 通过调用底层迭代器的 `valid` 方法来实现。
    CUTLASS_HOST_DEVICE
    bool valid() {
      return iterator_.valid();
    }
  };

  ////////////////////////////////////////////////////////////////////////////////

  /// Specialization of PredicatedTileAccessIteratorResidualLast for affine rank-2
  /// row-major data.
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
      typename AccessType_>
  class PredicatedTileAccessIteratorResidualLast<
      Shape_,
      Element_,
      layout::AffineRank2RowMajor,
      AdvanceRank,
      ThreadMap_,
      AccessType_,
      false> {
   public:
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::AffineRank2RowMajor;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    // Map to the underlying AffineRankN<2> layout
    using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
        layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
        Element,
        layout::AffineRankN<2>,
        (kAdvanceRank == 0 ? 1 : 0),
        ThreadMap,
        AccessType>;

    static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;

    /// Parameters object is precomputed state and is host-constructible
    class Params {
     private:
      friend PredicatedTileAccessIteratorResidualLast;

      /// Parameters object
      typename UnderlyingIterator::Params params_;

     public:
      /// Default ctor
      CUTLASS_HOST_DEVICE
      Params() {}

      /// Construct the Params object given an AffineRankN<2> tensor's layout
      CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::AffineRankN<2>(layout.stride(1), layout.stride(0))){};
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
  PredicatedTileAccessIteratorResidualLast(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset,
      int const* indices =
          nullptr ///< gather/scatter indices, note no support for
                  ///< gather/scatter at this specialization
      )
      // Initialize the iterator using precomputed parameters, pointer to tensor data,
      // tensor extent, thread ID, and threadblock offset
      : iterator_(
            params.params_,
            pointer,
            layout::PitchLinearCoord(extent.column(), extent.row()),
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row())) {}

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      // Delegate constructor call to the primary constructor with zero threadblock offset
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset(
        make_Coord(tile_offset.column(), tile_offset.row()));
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
  }
    // 返回当前对象本身，用于支持连续的赋值操作
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  // 后缀自增运算符重载，返回自增前的迭代器对象
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    // 复制当前迭代器对象
    PredicatedTileAccessIteratorResidualLast self(*this);
    // 调用前缀自增运算符，更新迭代器内部指针并返回自身
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  // 清除谓词集合的方法，可以选择是否启用
  void clear_mask(bool enable = true) {
    // 调用迭代器对象的清除谓词集合方法
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  // 启用谓词集合的方法
  void enable_mask() {
    // 调用迭代器对象的启用谓词集合方法
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  // 设置谓词掩码的方法，用传入的掩码覆盖谓词迭代器中存储的值
  void set_mask(Mask const& mask) {
    // 调用迭代器对象的设置谓词掩码方法
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  // 获取谓词掩码的方法
  void get_mask(Mask& mask) {
    // 调用迭代器对象的获取谓词掩码方法
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  // 判断访问是否有效的方法，返回布尔值
  bool valid() {
    // 调用迭代器对象的判断访问是否有效的方法并返回结果
    return iterator_.valid();
  }
  };

  ////////////////////////////////////////////////////////////////////////////////

  /// Specialization of PredicatedTileAccessIteratorResidualLast for column-major
  /// interleaved data. It is mapped to the congruous layout.
  ///
  /// Satisfies: ForwardTileIteratorConcept |
  ///            ReadableContiguousTileIteratorConcept |
  ///            WriteableContiguousTileIteratorConcept |
  ///            MaskedTileIteratorConcept
  ///
  
  template <
      typename Shape_,                              // 模板参数：迭代器形状
      typename Element_,                            // 模板参数：元素类型
      int AdvanceRank,                              // 模板参数：提升等级
      typename ThreadMap_,                          // 模板参数：线程映射
      typename AccessType_,                         // 模板参数：访问类型
      int InterleavedK>                             // 模板参数：交织因子
  class PredicatedTileAccessIteratorResidualLast<
      Shape_,                                       // 迭代器形状
      Element_,                                     // 元素类型
      layout::ColumnMajorInterleaved<InterleavedK>,  // 列主交织布局
      AdvanceRank,                                  // 提升等级
      ThreadMap_,                                   // 线程映射
      AccessType_,                                  // 访问类型
      false> {                                      // 布尔值参数

   public:
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,        // 静态断言：提升等级只能是0或1
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;                           // 使用Shape别名代表形状类型
    using Element = Element_;                       // 使用Element别名代表元素类型
    static int const kInterleavedK = InterleavedK;  // 设置交织因子常量
    using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;  // 使用列主交织布局
    static int const kAdvanceRank = AdvanceRank;    // 设置提升等级常量
    using ThreadMap = ThreadMap_;                   // 使用ThreadMap别名代表线程映射类型
    using AccessType = AccessType_;                 // 使用AccessType别名代表访问类型

    using Index = typename Layout::Index;           // 使用Index别名代表索引类型
    using LongIndex = typename Layout::LongIndex;   // 使用LongIndex别名代表长索引类型

    using TensorRef = TensorRef<Element, Layout>;   // 张量引用类型
    using TensorView = TensorView<Element, Layout>; // 张量视图类型
    using TensorCoord = typename Layout::TensorCoord;  // 张量坐标类型

    using Pointer = Element*;                      // 指针类型
    using NonConstPointer = typename platform::remove_const<Element>::type*;  // 非常量指针类型

    using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
        layout::PitchLinearShape<
            Shape::kRow * kInterleavedK,
            Shape::kColumn / kInterleavedK>,       // 基础迭代器类型
        Element,
        layout::PitchLinear,
        (kAdvanceRank == 0 ? 0 : 1),                // 基础迭代器提升等级
        ThreadMap,
        AccessType>;

    static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;  // 每向量访问数常量

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;  // 谓词向量类型

    /// Parameters object is precomputed state and is host-constructible
    class Params {
     private:
      friend PredicatedTileAccessIteratorResidualLast;

      /// Parameters object
      typename UnderlyingIterator::Params params_;   // 基础迭代器的参数对象

     public:
      CUTLASS_HOST_DEVICE
      Params() {}

      /// Construct the Params object given a pitch-linear tensor's layout
      CUTLASS_HOST_DEVICE
      Params(Layout const& layout)
          : params_(layout::PitchLinear(layout.stride(0))) {}  // 使用布局构造参数对象

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
  PredicatedTileAccessIteratorResidualLast(
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
      : iterator_(  // 初始化成员变量 iterator_
            params.params_,  // 使用 params 对象的 params_ 参数初始化 iterator_
            pointer,  // 设置指向张量起始位置的指针
            layout::PitchLinearCoord(  // 计算张量的布局坐标
                extent.row() * kInterleavedK,  // 行维度乘以 kInterleavedK
                extent.column() / kInterleavedK),  // 列维度除以 kInterleavedK
            thread_id,  // 设置每个参与线程的线程 ID
            layout::PitchLinearCoord(  // 计算线程块的偏移坐标
                threadblock_offset.row() * kInterleavedK,  // 线程块行维度乘以 kInterleavedK
                threadblock_offset.column() / kInterleavedK)) {}  // 线程块列维度除以 kInterleavedK

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(  // 调用上面的构造函数，传入参数
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}  // 使用 make_Coord 创建 (0, 0) 坐标作为线程块偏移量

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {  // 设置迭代索引
    iterator_.set_iteration_index(index);  // 调用成员变量 iterator_ 的 set_iteration_index 方法
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {  // 设置是否启用残余瓦片
    iterator_.set_residual_tile(enable);  // 调用成员变量 iterator_ 的 set_residual_tile 方法
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {  // 添加指针偏移量（以元素为单位）
    iterator_.add_pointer_offset(pointer_offset);  // 调用成员变量 iterator_ 的 add_pointer_offset 方法
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {  // 按照整个瓦片单位在矩阵的逻辑维度上推进迭代器
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});  // 调用成员变量 iterator_ 的 add_tile_offset 方法，传入瓦片偏移坐标
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {  // 返回指针
  // 返回对迭代器指针的重新解释，作为特定访问类型的指针
  return reinterpret_cast<AccessType*>(iterator_.get());
}

/// 前进到内存中的下一个瓦片。
///
/// 首次调用此方法时，更新谓词并将迭代器的内部指针恢复到第一个"稳态"瓦片。
/// 后续调用轻量化，只需更新内部指针。
CUTLASS_HOST_DEVICE
PredicatedTileAccessIteratorResidualLast& operator++() {
  ++iterator_;
  return *this;
}

/// 前进到内存中的下一个瓦片。
///
/// 首次调用此方法时，更新谓词并将迭代器的内部指针恢复到第一个"稳态"瓦片。
/// 后续调用轻量化，只需更新内部指针。
CUTLASS_HOST_DEVICE
PredicatedTileAccessIteratorResidualLast operator++(int) {
  PredicatedTileAccessIteratorResidualLast self(*this);
  operator++();
  return self;
}

/// 高效清除谓词集合
CUTLASS_HOST_DEVICE
void clear_mask(bool enable = true) {
  iterator_.clear_mask(enable);
}

/// 高效启用谓词集合
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

/// 返回访问是否有效
CUTLASS_HOST_DEVICE
bool valid() {
  return iterator_.valid();
}
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIteratorResidualLast for row-major
/// interleaved data.
//  It is mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    typename Shape_,                     // 模板参数：迭代器的形状描述
    typename Element_,                   // 模板参数：元素类型
    int AdvanceRank,                     // 模板参数：进阶的维度(rank)
    typename ThreadMap_,                 // 模板参数：线程映射
    typename AccessType_,                // 模板参数：访问类型
    int InterleavedK>                    // 模板参数：交错因子
class PredicatedTileAccessIteratorResidualLast<
    Shape_,                              // 迭代器的形状描述
    Element_,                            // 元素类型
    layout::RowMajorInterleaved<InterleavedK>, // 行主序交错布局
    AdvanceRank,                         // 进阶的维度(rank)
    ThreadMap_,                          // 线程映射
    AccessType_,                         // 访问类型
    false> {                             // 布尔参数
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;                  // 迭代器的形状类型别名
  using Element = Element_;              // 元素类型别名
  static int const kInterleavedK = InterleavedK; // 交错因子
  using Layout = layout::RowMajorInterleaved<kInterleavedK>; // 布局类型别名
  static int const kAdvanceRank = AdvanceRank; // 进阶的维度(rank)
  using ThreadMap = ThreadMap_;          // 线程映射类型别名
  using AccessType = AccessType_;        // 访问类型别名

  using Index = typename Layout::Index;  // 索引类型别名
  using LongIndex = typename Layout::LongIndex; // 长索引类型别名

  using TensorRef = TensorRef<Element, Layout>; // 引用类型别名
  using TensorView = TensorView<Element, Layout>; // 视图类型别名
  using TensorCoord = typename Layout::TensorCoord; // 张量坐标类型别名

  using Pointer = Element*;              // 指针类型别名
  using NonConstPointer = typename platform::remove_const<Element>::type*; // 非常量指针类型别名

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
      layout::PitchLinearShape<
          Shape::kColumn * kInterleavedK,
          Shape::kRow / kInterleavedK>, // 嵌套的迭代器类型
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),       // 根据进阶维度选择迭代器类型
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector; // 每个向量的访问次数

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask; // 谓词向量类型别名

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_; // 嵌套的迭代器参数类型

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : params_(layout::PitchLinear(layout.stride(0))) {} // 使用布局构造参数对象

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
  PredicatedTileAccessIteratorResidualLast(
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
      : iterator_(  // Initialize the underlying iterator with parameters
            params.params_,  // Use parameters from `params` object
            pointer,  // Start pointer to the tensor data
            layout::PitchLinearCoord(  // Calculate pitch-linear layout coordinates
                extent.column() * kInterleavedK,  // Column dimension multiplied by interleaved K
                extent.row() / kInterleavedK),  // Row dimension divided by interleaved K
            thread_id,  // ID of the current thread
            layout::PitchLinearCoord(  // Threadblock offset in pitch-linear coordinates
                threadblock_offset.column() * kInterleavedK,  // Column offset multiplied by interleaved K
                threadblock_offset.row() / kInterleavedK)) {}  // Row offset divided by interleaved K

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      TensorCoord extent, ///< Extent of tensor
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
            params,
            pointer,
            extent,
            thread_id,
            make_Coord(0, 0)) {}  // Construct using the main constructor with zero threadblock offset

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);  // Set the iteration index of the underlying iterator
  }

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);  // Set whether residual tile handling is enabled in the underlying iterator
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);  // Add a pointer offset in terms of Element units in the underlying iterator
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});  // Add a tile offset in terms of matrix logical dimensions to the underlying iterator
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return iterator_.get();  // Return a pointer from the underlying iterator
  }
  /// Returns a pointer to the accessed element with the specified access type.
  ///
  /// This method returns a pointer cast to AccessType* of the current iterator position.
  /// It is used for direct access to elements in memory.
  /// The caller must ensure proper alignment and type safety of AccessType.
  CUTLASS_HOST_DEVICE
  AccessType* operator->() {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and only update the internal pointer.
  /// This method returns a copy of the iterator before incrementing.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate mask efficiently.
  ///
  /// This method clears the predicate mask associated with the iterator,
  /// allowing unmasked access to elements.
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Enables the predicate mask.
  ///
  /// This method enables the predicate mask associated with the iterator,
  /// allowing masked access to elements as per the mask's settings.
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask explicitly.
  ///
  /// This method sets the predicate mask associated with the iterator to the provided mask.
  /// It overrides any previously set predicate mask.
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Retrieves the current predicate mask.
  ///
  /// This method retrieves the current predicate mask stored in the iterator.
  /// The retrieved mask is stored in the provided 'mask' parameter.
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Checks if the current access is valid.
  ///
  /// This method returns true if the current access through the iterator is valid,
  /// indicating that the iterator is pointing to a valid memory location.
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
```