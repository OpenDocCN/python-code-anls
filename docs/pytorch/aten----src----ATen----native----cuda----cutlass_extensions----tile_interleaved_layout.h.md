# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\tile_interleaved_layout.h`

```
/**
 * @file
 * @brief Defines new layouts needed for MoE (Mixtures of Experts)
 */

#pragma once

#include <cutlass/cutlass.h>          // 引入 Cutlass 库
#include <cutlass/fast_math.h>        // 引入快速数学运算库
#include <cutlass/matrix_coord.h>     // 引入矩阵坐标定义库
#include <cutlass/pitch_linear_coord.h>   // 引入线性偏移坐标定义库

namespace cutlass {
namespace layout {

/**
 * @brief Column-major tile interleaving layout template
 * @tparam RowsPerTile Number of rows per tile
 * @tparam ColumnsInterleaved Number of columns interleaved
 */
template<int RowsPerTile, int ColumnsInterleaved>
class ColumnMajorTileInterleave {
    static constexpr int kRowsPerTile        = RowsPerTile;        // 每个瓦片的行数
    static constexpr int kColumnsInterleaved = ColumnsInterleaved; // 交织的列数
};

/**
 * @brief Type trait to determine if a type is ColumnMajorTileInterleave
 * @tparam T Type to check
 */
template<class T>
struct IsColumnMajorTileInterleave {
    static constexpr bool value = false;    // 默认情况下，不是 ColumnMajorTileInterleave 类型
};

/**
 * @brief Specialization of IsColumnMajorTileInterleave for ColumnMajorTileInterleave template
 * @tparam U Template parameter of ColumnMajorTileInterleave
 * @tparam V Template parameter of ColumnMajorTileInterleave
 */
template<int U, int V>
struct IsColumnMajorTileInterleave<ColumnMajorTileInterleave<U, V>> {
    static constexpr bool value = true;     // 如果是 ColumnMajorTileInterleave 类型，则为 true
};

}  // namespace layout
}  // namespace cutlass
```