# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\transform\tile_smem_loader.h`

```
/*
 * 版权所有 (c) Meta Platforms, Inc. 和其附属公司。
 * 保留所有权利。
 *
 * 此源代码在根目录中的 LICENSE 文件中所述的 BSD 风格许可下许可。
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator.h>

template <
    typename scalar_t, // 标量类型
    typename ThreadblockTileShape, // 要加载的块状瓦片的大小
    int Threads, // 参与的线程数
    int ElementsPerAccess> // 每次访问的线程访问宽度（元素数）
class TileSmemLoader {
 public:
  using SmemTile =
      cutlass::AlignedBuffer<scalar_t, ThreadblockTileShape::kCount>; // 共享内存中的瓦片类型

  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
      cutlass::layout::PitchLinearShape<
          ThreadblockTileShape::kColumn, // 连续
          ThreadblockTileShape::kRow>, // 间隔
      Threads, // 线程数
      ElementsPerAccess>; // 每次访问的元素数

  using GmemTileIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          ThreadblockTileShape, // 形状
          scalar_t, // 元素类型
          cutlass::layout::RowMajor, // 布局
          0, // 前进等级
          ThreadMap>; // 线程映射类型

  using SmemTileIterator = cutlass::transform::threadblock::RegularTileIterator<
      ThreadblockTileShape, // 形状
      scalar_t, // 元素类型
      cutlass::layout::RowMajor, // 布局
      0, // 前进等级
      ThreadMap>; // 线程映射类型

  using Fragment = typename GmemTileIterator::Fragment; // 片段类型

  /// 从全局内存加载一个瓦片到共享内存中
  CUTLASS_DEVICE
  static void load(
      GmemTileIterator tile_load_iter, // 全局内存中的瓦片加载迭代器
      SmemTileIterator tile_store_iter) { // 共享内存中的瓦片存储迭代器
    Fragment tb_frag; // 创建片段对象
    tb_frag.clear(); // 清空片段内容
    tile_load_iter.load(tb_frag); // 使用加载迭代器加载瓦片到片段中
    tile_store_iter.store(tb_frag); // 使用存储迭代器将片段存储到共享内存中

    __syncthreads(); // 同步所有线程
  }
};
```