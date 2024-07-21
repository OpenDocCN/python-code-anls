# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseSemiStructuredPack.h`

```
#pragma once
// 包含必要的头文件，用于声明或导入所需的库和函数
#include <ATen/native/sparse/cuda/StaticSort.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>

// 将当前代码放入 at::native 命名空间中，以防止全局命名冲突
namespace at::native {

// 使用 cutlass 库中定义的数据类型
using cutlass::uint1b_t;
using cutlass::uint2b_t;
using cutlass::uint4b_t;
// 定义一个新的类型别名，具体类型由 cutlass::integer_subbyte 模板参数决定
using uint8b_t = cutlass::integer_subbyte<8, false>;
// 定义新的数据布局类型别名，用于输入数据的重新排列
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;
// 定义输入元素的类型别名
using ElementInputE = uint16_t;

// 定义常量，表示线程块和线程的维度
constexpr int kWarpX = 32;
constexpr int kWarpY = 64;
constexpr int kThreadX = 8;
constexpr int kThreadY = 8;

// 定义一个结构体，用于存储 8x8 区域的掩码数据
struct Tile8x8Masks {
  Indices4x4 a, b, c, d;
  // 构造函数，初始化所有成员为 0
  CUTLASS_DEVICE Tile8x8Masks() {
    a = b = c = d = 0;
  }
};

// 静态断言，确保 Tile8x8Masks 结构体的大小为 8 字节
static_assert(sizeof(Tile8x8Masks) == 8, "should be exactly uint64_t");

// 函数用于实现 warp 级别的数据重排，将数据发送到正确的线程中
// 参数 meta_ab 是要重排的元数据，transposed 表示是否需要进行转置操作
CUTLASS_DEVICE uint32_t
warp_shuffle_meta(uint32_t meta_ab, bool transposed = false) {
  // 以下是重排后数据在线程中的分布情况的描述和注释，详细信息见提供的链接
  // 使用 warp-shuffles 将数据发送到正确的线程中
  bool thread_left = (threadIdx.y % 2) == 0;
  bool thread_bottom = threadIdx.x % 2;

  if (transposed) {
    thread_left = (threadIdx.x % 2) == 0;
  }


这段代码中声明了一些常量、类型别名和函数原型，同时定义了一个结构体和一个静态断言。接下来的注释将继续解释这些代码的余下部分。
    // 计算线程在块中的底部位置
    thread_bottom = threadIdx.y % 2;
    
    // 创建一个包含两个元素的数组，每个元素表示从 meta_ab 中提取的字节
    uint8b_t stage0_data[2] = {
        uint8b_t(meta_ab >> (8 * thread_left)),                 // 提取第一个字节
        uint8b_t(meta_ab >> (8 * (thread_left + 2)))            // 提取第二个字节
    };
    
    // 执行 shfl 操作，根据 transposed 的值选择不同的偏移量
    // 如果 transposed 为真，则偏移量为1，否则为4
    stage0_data[0] = uint8b_t(__shfl_xor_sync(0xffffffff, stage0_data[0], transposed ? 1 : 4));
    stage0_data[1] = uint8b_t(__shfl_xor_sync(0xffffffff, stage0_data[1], transposed ? 1 : 4));
    
    // 构建 line0 和 line1 的值，用于后续操作
    uint16_t line0 = int(uint8b_t(meta_ab >> (8 * (1 - thread_left)))) << ((1 - thread_left) * 8);
    line0 |= int(stage0_data[0]) << (thread_left * 8);
    
    uint16_t line1 = int(uint8b_t(meta_ab >> (8 * (1 - thread_left + 2)))) << ((1 - thread_left) * 8);
    line1 |= int(stage0_data[1]) << (thread_left * 8);
    
    // 根据 thread_bottom 的值选择 stage1_data 的值
    uint16_t stage1_data = thread_bottom ? line0 : line1;
    
    // 执行 shfl 操作，根据 transposed 的值选择不同的偏移量
    // 如果 transposed 为真，则偏移量为4，否则为1
    stage1_data = __shfl_xor_sync(0xffffffff, stage1_data, transposed ? 4 : 1);
    
    // 根据 thread_bottom 的值选择 final_metadata 的值
    uint32_t final_metadata;
    if (thread_bottom) {
        final_metadata = uint32_t(stage1_data) | uint32_t(line1) << 16;
    } else {
        final_metadata = uint32_t(stage1_data) << 16 | uint32_t(line0);
    }
    
    // 返回计算得到的 final_metadata
    return final_metadata;
}

CUTLASS_DEVICE void warp_shuffle_and_write_meta(
    ElementInputE* metadata_quad,
    uint32_t meta_ab,
    bool transposed = false) {
  // 计算当前线程是否在左侧
  bool thread_left = (threadIdx.y % 2) == 0;
  // 计算当前线程是否在底部
  bool thread_bottom = threadIdx.x % 2;

  // 如果进行了转置操作
  if (transposed) {
    // 重新计算当前线程是否在左侧
    thread_left = (threadIdx.x % 2) == 0;
    // 重新计算当前线程是否在底部
    thread_bottom = threadIdx.y % 2;
  }

  // 使用 warp_shuffle_meta 函数进行元数据的重排和处理，获取最终的元数据值
  uint32_t final_metadata = warp_shuffle_meta(meta_ab, transposed);

  // 根据当前线程的位置计算元数据在 metadata_quad 中的索引
  int index = (!thread_left + 2 * thread_bottom) * 4;
  // 将最终的元数据写入到 metadata_quad 中的指定位置
  ((uint32_t*)metadata_quad)[index] = final_metadata;
}

template <typename Element_>
struct KernelTypes {
  using Element = Element_;
  using Fragment =
      cutlass::Array<Element, 8>; // 总是以128位的块从全局内存读取
  using Fragment4 = cutlass::Array<Element, 4>;
  using ValuesPacked = cutlass::Array<Element, 8>; // 前4列和后4列的数据

  struct Params {
    /// inputs
    Element const* input;  // 输入数据的指针
    int64_t input_s0;  // 输入数据的步长
    int64_t input_dim0;  // 输入数据的第一维大小
    int64_t input_dim1;  // 输入数据的第二维大小

    /// outputs
    Element* packed;  // 打包后的数据指针
    int64_t packed_stride;  // 打包后数据的步长

    Element* packed_trans;  // 转置后的打包数据指针
    int64_t packed_trans_stride;  // 转置后打包数据的步长

    uint64_t* threads_masks;  // 线程掩码数组指针

    __host__ dim3 getBlocksGrid() const {
      // 计算块的网格大小
      return dim3(
          cutlass::ceil_div(input_dim0, kWarpX),
          cutlass::ceil_div(input_dim1, kWarpY),
          1);
    }

    static CUTLASS_HOST_DEVICE dim3 getThreadsGrid() {
      // 获取线程的网格大小
      return dim3(kWarpX / kThreadX, kWarpY / kThreadY, 1);
    }

    CUTLASS_DEVICE Tile8x8Masks* getCurrentThreadIndices() const {
      // 将线程掩码转换为 Tile8x8Masks 类型指针
      Tile8x8Masks* gmem_threads_masks = (Tile8x8Masks*)threads_masks;
      // 根据当前块和线程的索引计算偏移量
      gmem_threads_masks += blockIdx.y * getThreadsGrid().y + threadIdx.y;
      int64_t strideX = gridDim.y * getThreadsGrid().y;
      gmem_threads_masks +=
          (blockIdx.x * getThreadsGrid().x + threadIdx.x) * strideX;
      return gmem_threads_masks;
    }
  };

  struct Tile4x4Accessor {
    using Element = Element_;

    Fragment (&_lines)[8];  // 引用到存储行的数组
    int _start_row;  // 起始行索引
    int _start_col;  // 起始列索引

    CUTLASS_DEVICE Tile4x4Accessor(
        Fragment (&lines)[8],
        int start_row,
        int start_col)
        : _lines(lines), _start_row(start_row), _start_col(start_col) {}

    CUTLASS_DEVICE typename Fragment::reference at(int r, int c) {
      // 获取指定位置的元素引用
      return _lines[r + _start_row][c + _start_col];
    }
  };

  struct Tile4x4Packed {
    Fragment4 values[2];  // 存储两个 Fragment4 类型的数组

    CUTLASS_DEVICE Tile4x4Packed() {
      // 初始化 values 数组中的元素
      values[0].clear();
      values[1].clear();
  };

  // 返回一个打包的 4x4 块（例如 2x4 值），对应于 `indices` 中的值
  // 同时填充 `meta` 数组，以适合在 TensorCores 中消耗的格式
  // 示例：
  //  indices:  0011
  //            1001
  //            1001
  //            0100 (<- 注意，最后一行只有一个值)
  //  packed: values[0][2] values[1][0] values[2][0] values[3][1]
  //          values[0][3] values[1][3] values[2][3] Element(0)
  CUTLASS_DEVICE static Tile4x4Packed pack_4x4(
      Indices4x4 indices,
      Tile4x4Accessor tile,
      uint32_t& meta,
      int meta_pos,
      bool transpose = false) {
    Tile4x4Packed packed;

    // 对每一行进行循环打包
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      uint2b_t col0_from, col1_from;

      // lambda 函数，用于打包每一个值
      auto packValue = [&](uint2b_t col_to, uint2b_t col_from) {
        auto value = transpose ? tile.at(col_from, row).get()
                               : tile.at(row, col_from).get();
        packed.values[col_to][row] = value;
        
        // 根据 col_to 的值设置 col0_from 或 col1_from
        if (col_to == uint2b_t(0)) {
          col0_from = col_from;
        } else {
          col1_from = col_from;
        }
      };

      // lambda 函数，检查是否选中某一列
      auto isSelected = [&](int col) {
        if (transpose) {
          return indices & (1 << (row + 4 * col));
        }
        return indices & (1 << (col + 4 * row));
      };

      // 处理列 0 和 1
      // 我们知道如果有的话，col0 总是打包到位置 0，col1 打包到位置 0 或 1（取决于是否选择了 col0）
      if (isSelected(1)) {
        packValue(uint2b_t(0), uint2b_t(1));
      }
      if (isSelected(0)) {
        packValue(uint2b_t(0), uint2b_t(0));
      }
      if (isSelected(0) && isSelected(1)) {
        packValue(uint2b_t(1), uint2b_t(1));
      }

      // 处理列 2 和 3
      // 类似的启发式方法
      if (isSelected(2)) {
        packValue(uint2b_t(1), uint2b_t(2));
      }
      if (isSelected(3)) {
        packValue(uint2b_t(1), uint2b_t(3));
      }
      if (isSelected(2) && isSelected(3)) {
        packValue(uint2b_t(0), uint2b_t(2));
      }

      // 根据 col0_from 和 col1_from 设置添加掩码，并更新 meta
      int add_mask = (col0_from | (col1_from << 2)) << (8 * row + meta_pos);
      meta |= add_mask;
    }

    // 返回打包好的块
    return packed;
  }

  struct Tile8x8Meta {
    // meta_ab[row] |= (real_col << (8*row + 2*pos))
    uint32_t meta_ab;
    uint32_t meta_cd;

    // meta_ac_trans[col] |= (real_row << (8*col + 2*pos))
    uint32_t meta_ac_trans;
    uint32_t meta_bd_trans;

    // 构造函数，初始化所有 meta 变量为 0
    CUTLASS_DEVICE Tile8x8Meta() {
      meta_ab = meta_cd = meta_ac_trans = meta_bd_trans = 0;
    }
  };

  // 将打包后的数据写入指针 `ptr` 所指向的内存位置
  CUTLASS_DEVICE static void writePacked(
      Element* ptr,
      Fragment4 packed0,
      Fragment4 packed1) {
    Fragment write;

    // 对于每一个元素，按照顺序将 packed0 和 packed1 中的数据写入 write 中
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      write[i] = packed0[i].get();
      write[i + 4] = packed1[i].get();
    }
    cutlass::arch::global_store<Fragment, sizeof(Fragment)>(write, ptr, true);
  }



  CUTLASS_DEVICE static void writePackedT(
      Element* ptr,
      int64_t stride,
      Tile4x4Packed a,
      Tile4x4Packed b) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      Fragment4 write;
      write[0] = a.values[0][i].get();
      write[1] = a.values[1][i].get();
      write[2] = b.values[0][i].get();
      write[3] = b.values[1][i].get();
      cutlass::arch::global_store<Fragment4, sizeof(Fragment4)>(
          write, ptr + i * stride, true);
    }
  }



  template <typename Algorithm, typename MetadataStore>
  CUTLASS_DEVICE static void sparse_semi_structured_tile_kernel(
      Params p,
      MetadataStore metadata_gmem,
      Algorithm compute_tile_indices) {
    // Each thread is responsible for an 8x8 tile, which contains 4 4x4 tiles:
    // A, B, C and D, as displayed in the following schema:
    // +---+---+
    // | A | B |
    // +---+---+
    // | C | D |
    // +---+---+
    // Each warp (32 threads) will then be responsible for a 32x64 tile of the
    // input.
    // This configuration allows to read/write data in 128bits chunks. These
    // memory accesses are coalesced at the warp-level into 128bytes. See also:
    // https://docs.google.com/presentation/d/1DtmKThv8S5QAyBktuLRYzZhRzCvS1qSkBbrqNCjMPeA/edit#slide=id.g2494f30c7cf_0_0

    // Top-left of the 8x8 tile we own
    int warp_x = blockIdx.x * kWarpX;
    int warp_y = blockIdx.y * kWarpY;
    int x = warp_x + threadIdx.x * kThreadX;
    int y = warp_y + threadIdx.y * kThreadY;

    Element const* input = p.input + x * p.input_s0 + y;
    Element* packed = p.packed + x * p.packed_stride + (y / 2);
    Element* packed_trans =
        p.packed_trans + (x / 2) + y * p.packed_trans_stride;

    Fragment lines[8]; // Contains all values from the 8x8 tile

    Tile8x8Meta metadata;
    Tile8x8Masks indices;

    // Load/process tiles `A` and `B`
    Element fillValue = Algorithm::template outOfBoundsFillValue<Element>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      lines[i].fill(fillValue);
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }
    indices.a = compute_tile_indices(Tile4x4Accessor(lines, 0, 0));
    indices.b = compute_tile_indices(Tile4x4Accessor(lines, 0, 4));

    // Compute packed tiles A & B
    {
      Tile4x4Packed packed_a = pack_4x4(
          indices.a, Tile4x4Accessor(lines, 0, 0), metadata.meta_ab, 0);
      Tile4x4Packed packed_b = pack_4x4(
          indices.b, Tile4x4Accessor(lines, 0, 4), metadata.meta_ab, 4);
      writePackedT(packed, p.packed_stride, packed_a, packed_b);
    }

    // Compute/store packed tiles A & B in transpose output
    Tile4x4Packed packed_trans_a = pack_4x4(
        indices.a,
        Tile4x4Accessor(lines, 0, 0),
        metadata.meta_ac_trans,
        0,
        true);



        p.packed_trans + (x / 2) + y * p.packed_trans_stride;



        Tile4x4Packed packed_b = pack_4x4(



        indices.a, Tile4x4Accessor(lines, 0, 0), metadata.meta_ab, 0);



        Tile4x4Packed packed_b = pack_4x4(



        packed, p.packed_stride, packed_a, packed_b);
    // Pack B into a 4x4 tile for transpose operation
    Tile4x4Packed packed_trans_b = pack_4x4(
        indices.b,
        Tile4x4Accessor(lines, 0, 4),
        metadata.meta_bd_trans,
        0,
        true);
    // (NOTE) Now we no longer need A & B (`lines[0:4]`)

    // Load and process tiles C and D
    CUTLASS_PRAGMA_UNROLL
    for (int i = 4; i < 8; ++i) {
      // Initialize lines[i] with fillValue
      lines[i].fill(fillValue);
      // Perform global load into lines[i] from input
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }
    // Compute indices for tiles C and D
    indices.c = compute_tile_indices(Tile4x4Accessor(lines, 4, 0));
    indices.d = compute_tile_indices(Tile4x4Accessor(lines, 4, 4));

    // Compute packed tiles C & D for non-transpose operation
    {
      // Pack tile C into a 4x4 format
      Tile4x4Packed packed_c = pack_4x4(
          indices.c, Tile4x4Accessor(lines, 4, 0), metadata.meta_cd, 0);
      // Pack tile D into a 4x4 format
      Tile4x4Packed packed_d = pack_4x4(
          indices.d, Tile4x4Accessor(lines, 4, 4), metadata.meta_cd, 4);
      // Write packed tiles C and D into memory
      writePackedT(
          packed + 4 * p.packed_stride, p.packed_stride, packed_c, packed_d);
    }

    // Compute and store packed tiles C & D in transpose output
    Tile4x4Packed packed_trans_c = pack_4x4(
        indices.c,
        Tile4x4Accessor(lines, 4, 0),
        metadata.meta_ac_trans,
        4,
        true);
    Tile4x4Packed packed_trans_d = pack_4x4(
        indices.d,
        Tile4x4Accessor(lines, 4, 4),
        metadata.meta_bd_trans,
        4,
        true);

    // Store thread-specific metadata indices
    *p.getCurrentThreadIndices() = indices;

    // Store packed A, B, C & D for transposed matrix
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_a, packed_trans_c);
    packed_trans += 4 * p.packed_trans_stride;
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_b, packed_trans_d);

    // Write metadata in a non-transposed format
    {
      // Retrieve metadata for non-transposed operation
      ElementInputE* packed_meta_reordered = metadata_gmem.get_metaN(
          warp_x, threadIdx.x * kThreadX, warp_y, threadIdx.y * kThreadY);
      // Shuffle and write metadata for tiles A & B
      warp_shuffle_and_write_meta(packed_meta_reordered, metadata.meta_ab);
      // Shuffle and write metadata for tiles C & D
      warp_shuffle_and_write_meta(packed_meta_reordered + 32, metadata.meta_cd);
    }

    // Write metadata in a transposed format
    {
      // Retrieve metadata for transposed operation
      ElementInputE* packed_trans_meta_reordered = metadata_gmem.get_metaT(
          warp_x, threadIdx.x * kThreadX, warp_y, threadIdx.y * kThreadY);
      // Shuffle and write transposed metadata for tiles A & C
      warp_shuffle_and_write_meta(
          packed_trans_meta_reordered, metadata.meta_ac_trans, true);
      // Shuffle and write transposed metadata for tiles B & D
      warp_shuffle_and_write_meta(
          packed_trans_meta_reordered + 32, metadata.meta_bd_trans, true);
    }
  }

  CUTLASS_DEVICE static void sparse_semi_structured_apply_kernel(Params p) {
    // See `sparse24_sparsify_both_ways_kernel`
    // It's basically the same, just that we skip
    // the part where we compute the indices we keep

    // Determine top-left coordinates of the 8x8 tile owned by the warp
    int warp_x = blockIdx.x * kWarpX;
    int warp_y = blockIdx.y * kWarpY;
    int x = warp_x + threadIdx.x * kThreadX;
    int y = warp_y + threadIdx.y * kThreadY;
    // 计算输入元素在二维数组中的偏移位置，并获取指向该元素的指针
    Element const* input = p.input + x * p.input_s0 + y;
    // 计算输出元素在二维数组中的偏移位置，并获取指向该元素的指针
    Element* packed = p.packed + x * p.packed_stride + (y / 2);
    // 计算转置后输出元素在二维数组中的偏移位置，并获取指向该元素的指针
    Element* packed_trans = p.packed_trans + (x / 2) + y * p.packed_trans_stride;

    // 声明一个包含8x8矩阵所有值的片段数组
    Fragment lines[8]; // Contains all values from the 8x8 tile

    // 声明8x8矩阵的元数据和线程相关的索引信息
    Tile8x8Meta metadata;
    Tile8x8Masks indices = *p.getCurrentThreadIndices();

    // 加载和处理矩阵块 `A` 和 `B`
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      // 注意：超出边界的值是未定义的，但不应在任何地方使用
      // 使用全局加载函数加载矩阵片段到 `lines[i]`
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }

    // 计算打包后的矩阵块 A 和 B
    {
      // 调用打包函数 `pack_4x4` 打包矩阵块 A
      Tile4x4Packed packed_a = pack_4x4(
          indices.a, Tile4x4Accessor(lines, 0, 0), metadata.meta_ab, 0);
      // 调用打包函数 `pack_4x4` 打包矩阵块 B
      Tile4x4Packed packed_b = pack_4x4(
          indices.b, Tile4x4Accessor(lines, 0, 4), metadata.meta_ab, 4);
      // 将打包后的矩阵块 A 和 B 写入输出数组 `packed`
      writePackedT(packed, p.packed_stride, packed_a, packed_b);
    }

    // 计算和存储转置后的矩阵块 A 和 B
    Tile4x4Packed packed_trans_a = pack_4x4(
        indices.a,
        Tile4x4Accessor(lines, 0, 0),
        metadata.meta_ac_trans,
        0,
        true);
    Tile4x4Packed packed_trans_b = pack_4x4(
        indices.b,
        Tile4x4Accessor(lines, 0, 4),
        metadata.meta_bd_trans,
        0,
        true);
    // 现在不再需要 `lines[0:4]`

    // 计算打包后的矩阵块 C 和 D
    {
      // 调用打包函数 `pack_4x4` 打包矩阵块 C
      Tile4x4Packed packed_c = pack_4x4(
          indices.c, Tile4x4Accessor(lines, 4, 0), metadata.meta_cd, 0);
      // 调用打包函数 `pack_4x4` 打包矩阵块 D
      Tile4x4Packed packed_d = pack_4x4(
          indices.d, Tile4x4Accessor(lines, 4, 4), metadata.meta_cd, 4);
      // 将打包后的矩阵块 C 和 D 写入输出数组 `packed` 的下一部分
      writePackedT(
          packed + 4 * p.packed_stride, p.packed_stride, packed_c, packed_d);
    }

    // 计算和存储转置后的矩阵块 C 和 D
    Tile4x4Packed packed_trans_c = pack_4x4(
        indices.c,
        Tile4x4Accessor(lines, 4, 0),
        metadata.meta_ac_trans,
        4,
        true);
    Tile4x4Packed packed_trans_d = pack_4x4(
        indices.d,
        Tile4x4Accessor(lines, 4, 4),
        metadata.meta_bd_trans,
        4,
        true);

    // 将打包后的矩阵块 A, B, C 和 D 写入转置后的输出数组 `packed_trans`
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_a, packed_trans_c);
    packed_trans += 4 * p.packed_trans_stride;
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_b, packed_trans_d);
};

// 结束 at::native 命名空间的定义
} // namespace at::native
```