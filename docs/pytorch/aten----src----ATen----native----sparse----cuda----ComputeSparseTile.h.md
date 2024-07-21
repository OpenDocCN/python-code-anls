# `.\pytorch\aten\src\ATen\native\sparse\cuda\ComputeSparseTile.h`

```
// 针对一组4x4值，计算在2:4稀疏化后将保留的选定索引，以位掩码形式返回。
// 注意：某些情况下算法可能选择少于8个值。

namespace platform {
// 针对cutlass::bfloat16_t类型，重载numeric_limits结构以提供无穷大值
template <>
struct numeric_limits<cutlass::bfloat16_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::bfloat16_t infinity() {
    return cutlass::bfloat16_t::bitcast(0x7f80);
  }
};
} // namespace platform

namespace at::native {

// 定义一个结构模板TileValueOrderedT，包含值、列索引和行索引
template <typename Element, typename Pointwise>
struct TileValueOrderedT {
  union {
    struct {
      Element value;
      uint2b_t col; // 列索引
      uint2b_t row; // 行索引
    } parts;
    uint32_t raw;
  };
  // 比较运算符，用于按指定规则排序TileValueOrderedT对象
  CUTLASS_DEVICE bool operator<(
      TileValueOrderedT<Element, Pointwise> const& other) const {
    return Pointwise::apply(parts.value) < Pointwise::apply(other.parts.value);
  }
  CUTLASS_DEVICE TileValueOrderedT() {}
};

// 定义一个操作结构IdentityOp，应用于按原始值排序
struct IdentityOp {
  template <typename T>
  static T CUTLASS_HOST_DEVICE apply(T const& x) {
    return x;
  }
};

// 定义一个操作结构AbsOp，应用于按绝对值排序
struct AbsOp {
  template <typename T>
  static T CUTLASS_HOST_DEVICE apply(T const& x) {
    return cutlass::abs(x);
  }
};

// 给定4x4值，计算在2:4稀疏化后将保留的选定索引，以位掩码形式返回。
// 我们有两个约束：
// (1) 每行最多选择2个值
// (2) 每列最多选择2个值
// 这意味着我们最多可以选择8个值。
// 算法：我们使用贪婪算法，按降序选择4x4块中的值。如果值适合（因为行/列尚未填满），则选择它。
// 然后我们继续下一个值。
// 注意：在某些情况下，该算法可能选择少于8个值。
// 注意（2）：RF不可索引，因此我们不应依赖于任何点索引值，否则它们将存储在本地内存中。
template <typename Op = IdentityOp>
struct LargestValuesGreedy {
  // 返回超出边界填充值，对于Tile4x4Accessor中的T类型
  template <typename T>
  static CUTLASS_DEVICE T outOfBoundsFillValue() {
    return -platform::numeric_limits<T>::infinity();
  }

  // 操作符()重载，接受Tile4x4Accessor类型的参数values
  template <typename Tile4x4Accessor>
  CUTLASS_DEVICE Indices4x4 operator()(Tile4x4Accessor values) {
    using TileValueOrdered =
        TileValueOrderedT<typename Tile4x4Accessor::Element, Op>;
    using TileValuesFragment = cutlass::Array<TileValueOrdered, 4 * 4>;
    Indices4x4 indices; // 索引对象
    TileValuesFragment values_ordered; // 排序后的值片段对象
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) { // 遍历行
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) { // 遍历列
        TileValueOrdered& v = values_ordered[i * 4 + j];
        v.parts.value = values.at(i, j).get(); // 获取值
        v.parts.col = uint2b_t(j); // 设置列索引
        v.parts.row = uint2b_t(i); // 设置行索引
      }
    }
    // 使用排序网络（无分支）以避免warp分歧
    // 创建静态排序器对象，使用模板参数 TileValuesFragment::kElements
    StaticSort<TileValuesFragment::kElements> sorter;
    // 对 values_ordered 进行排序，使用创建的 sorter 对象
    sorter(values_ordered);

    // 用于存储每行和每列选择的位掩码
    // 当前选中 0 个：(numPerRow >> 2*row) = 00 (0)
    // 当前选中 1 个：(numPerRow >> 2*row) = 01 (1)
    // 当前选中 2 个：(numPerRow >> 2*row) = 11 (3)
    uint32_t numPerRow = 0;
    uint32_t numPerCol = 0;
    indices = 0;

    // 从最大值开始，尽可能地选择元素
    CUTLASS_PRAGMA_UNROLL
    for (int i = values_ordered.size() - 1; i >= 0; i--) {
      auto& e = values_ordered[i];

      // 获取当前行和列已选的数量的位掩码
      uint32_t rcount = uint2b_t(numPerRow >> 2 * e.parts.row);
      uint32_t ccount = uint2b_t(numPerCol >> 2 * e.parts.col);
      
      // 判断当前元素 e 是否可以被选择
      // 注意：这种写法更高效（但等价于）：`rcount != 3 && ccount != 3`
      bool selected = (rcount + ccount) <= 2;

      // 将选择结果按位写入 indices
      indices |= selected << (e.parts.col + 4 * e.parts.row);

      // 更新每行和每列的已选数量的位掩码
      numPerRow |= (rcount + selected) << 2 * e.parts.row;
      numPerCol |= (ccount + selected) << 2 * e.parts.col;
    }
    // 返回最终的 indices 结果
    return indices;
}
};

// We consider each rows independantly in order
// This is to ensure that a row's sparsity pattern is only determined
// by its values and the rows before (but never the rows after)
// This enforces causality strictly

template <typename Op = IdentityOp>
struct Causal1122 {
  // Returns the out-of-bounds fill value for type T, which is negative infinity
  template <typename T>
  static CUTLASS_DEVICE T outOfBoundsFillValue() {
    return -platform::numeric_limits<T>::infinity();
  }

  // Functor that computes indices based on values in a 4x4 tile
  template <typename Tile4x4Accessor>
  CUTLASS_DEVICE Indices4x4 operator()(Tile4x4Accessor values) {
    // Array specifying the maximum number of values per row in a 4x4 tile
    static constexpr int kMaxValuesPerRow[] = {1, 1, 2, 2};
    
    // Define types used for sorting and ordering tile values
    using TileValueOrdered = TileValueOrderedT<typename Tile4x4Accessor::Element, Op>;
    using TileValuesFragment = cutlass::Array<TileValueOrdered, 4>;
    
    // Initialize indices for the 4x4 tile
    Indices4x4 indices = 0;

    uint32_t numPerCol = 0; // <- see doc in `LargestValuesGreedy`

    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      int row_count = 0;
      TileValuesFragment values_ordered;
      
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < 4; ++col) {
        TileValueOrdered& v = values_ordered[col];
        v.parts.value = values.at(row, col).get();
        v.parts.col = uint2b_t(col);
      }
      
      // Use a sorting network (aka without branches) to avoid
      // warp divergence
      StaticSort<TileValuesFragment::kElements> sorter;
      sorter(values_ordered);

      // Take as many as we can, starting with the largest values
      CUTLASS_PRAGMA_UNROLL
      for (int i = values_ordered.size() - 1; i >= 0; i--) {
        auto& e = values_ordered[i];

        uint32_t ccount = uint2b_t(numPerCol >> 2 * e.parts.col);
        bool selected = ccount != 3 && (row_count < kMaxValuesPerRow[row]);
        indices |= selected << (e.parts.col + 4 * row);
        numPerCol |= (ccount + selected) << 2 * e.parts.col;
        row_count += selected;
      }
    }
    return indices;
  }
};

// Function template that accepts a callback and invokes it with specific algorithms
template <typename T>
void named_algorithms(T callback) {
  callback(LargestValuesGreedy<IdentityOp>(), "largest_values_greedy");
  callback(Causal1122<IdentityOp>(), "causal1122");
  callback(LargestValuesGreedy<AbsOp>(), "largest_abs_values_greedy");
  // default one
  callback(LargestValuesGreedy<IdentityOp>(), "");
}

} // namespace
```