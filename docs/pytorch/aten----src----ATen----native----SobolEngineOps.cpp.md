# `.\pytorch\aten\src\ATen\native\SobolEngineOps.cpp`

```
/// 定义编译选项，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

/// 包含头文件：张量操作的核心定义
#include <ATen/core/Tensor.h>
/// 包含头文件：调度相关定义
#include <ATen/Dispatch.h>

/// 包含头文件：SobolEngine 操作的实用工具
#include <ATen/native/SobolEngineOpsUtils.h>
/// 包含头文件：C++ range utilities
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
/// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含以下头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
/// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含以下头文件
#include <ATen/ops/_sobol_engine_draw_native.h>
#include <ATen/ops/_sobol_engine_ff_native.h>
#include <ATen/ops/_sobol_engine_initialize_state_native.h>
#include <ATen/ops/_sobol_engine_scramble_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

using namespace sobol_utils;

/// 这是从 `SobolEngine` 中抽取样本的核心函数，给定其状态变量 `sobolstate` 和 `quasi`。
/// `dimension` 可以从 `sobolstate` 推断出，但选择显式传递以避免获取 `sobolstate` 的第一个维度大小的额外操作。
std::tuple<Tensor, Tensor> _sobol_engine_draw(const Tensor& quasi, int64_t n, const Tensor& sobolstate,
                                              int64_t dimension, int64_t num_generated, optional<ScalarType> dtype) {
    // 检查 `sobolstate` 张量的数据类型必须是 `at::kLong`
    TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);
    // 检查 `quasi` 张量的数据类型必须是 `at::kLong`
    TORCH_CHECK(quasi.dtype() == at::kLong,
           "quasi needs to be of type ", at::kLong);

    // 复制 `quasi` 张量以确保其存储格式是连续的
    Tensor wquasi = quasi.clone(at::MemoryFormat::Contiguous);
    // 确定结果张量的数据类型，如果指定了 `dtype` 则使用该值，否则使用 `at::kFloat`
    auto result_dtype = dtype.has_value() ? dtype.value() : at::kFloat;
    // 创建空的结果张量，形状为 (n, dimension)，数据类型与 `sobolstate` 相同
    Tensor result = at::empty({n, dimension}, sobolstate.options().dtype(result_dtype));

    // 根据结果张量的数据类型分发不同的操作
    AT_DISPATCH_FLOATING_TYPES(result_dtype, "_sobol_engine_draw", [&]() -> void {
        // 处理性能问题，直接访问数据和步长
        int64_t l;
        int64_t* wquasi_data = wquasi.data_ptr<int64_t>();
        int64_t* sobolstate_data = sobolstate.data_ptr<int64_t>();
        scalar_t* result_data = result.data_ptr<scalar_t>();

        int64_t wquasi_stride = wquasi.stride(0);
        int64_t sobolstate_row_stride = sobolstate.stride(0), sobolstate_col_stride = sobolstate.stride(1);
        int64_t result_row_stride = result.stride(0), result_col_stride = result.stride(1);

        // 遍历 `n` 次，每次生成一个样本
        for (int64_t i = 0; i < n; i++, num_generated++) {
            // 计算右边第一个为零的位的位置 `l`
            l = rightmost_zero(num_generated);
            // 遍历 `dimension` 次，每次处理一个维度的数据
            for (const auto j : c10::irange(dimension)) {
                // 对 `wquasi` 和 `sobolstate` 进行异或操作
                wquasi_data[j * wquasi_stride] ^= sobolstate_data[j * sobolstate_row_stride + l * sobolstate_col_stride];
                // 将处理后的 `wquasi` 数据存入结果张量
                result_data[i * result_row_stride + j * result_col_stride] = wquasi_data[j * wquasi_stride];
            }
        }
    });

    // 结果张量乘以常数 `RECIPD`
    result.mul_(RECIPD);
    // 返回包含结果张量和 `wquasi` 张量的元组
    return std::tuple<Tensor, Tensor>(result, wquasi);
}

/// 这是快速前进 `SobolEngine` 的核心函数，给定其状态变量 `sobolstate` 和 `quasi`。
/// `dimension` 可以从 `sobolstate` 推断出，但为了上述相同的原因，作为参数传递。
// 该函数用于对 Sobol 引擎的状态变量进行混淆操作。函数接受一个被混淆后的 `sobolstate` 状态变量和一个由 0 和 1 组成的随机下三角矩阵列表 `ltm`。`dimension` 参数再次显式传递。
Tensor& _sobol_engine_scramble_(Tensor& sobolstate, const Tensor& ltm, int64_t dimension) {
  // 检查 `sobolstate` 张量的数据类型是否为 `at::kLong`
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);

  /// 为 `sobolstate` 张量创建访问器
  auto ss_a = sobolstate.accessor<int64_t, 2>();

  /// 对于 `ltm` 列表中的每个张量，将其对角线元素设置为 1
  /// 创建 `ltm` 中所有矩阵的元素对应位置相乘，并沿着最后一个维度求和
  /// 可以通过 ltm_d_a[d][m] 访问 `ltm` 中第 d 个方阵的第 m 行。m 和 d 是从零开始计数的索引
  Tensor diag_true = ltm.clone(at::MemoryFormat::Contiguous);
  diag_true.diagonal(0, -2, -1).fill_(1);
  Tensor ltm_dots = cdot_pow2(diag_true);
  auto ltm_d_a = ltm_dots.accessor<int64_t, 2>();

  /// 主混淆循环
  for (const auto d : c10::irange(dimension)) {
    for (const auto j : c10::irange(MAXBIT)) {
      // 获取 ss_a[d][j] 的值，并初始化 l 和 t2
      int64_t vdj = ss_a[d][j], l = 1, t2 = 0;
      for (int64_t p = MAXBIT - 1; p >= 0; --p) {
        // 获取 ltm_d_a[d][p] 和 vdj 的位子集乘积的和
        int64_t lsmdp = ltm_d_a[d][p];
        int64_t t1 = 0;
        for (const auto k : c10::irange(MAXBIT)) {
          t1 += (bitsubseq(lsmdp, k, 1) * bitsubseq(vdj, k, 1));
        }
        t1 = t1 % 2;
        t2 = t2 + t1 * l;
        l = l << 1;
      }
      // 将 t2 的结果赋值给 ss_a[d][j]
      ss_a[d][j] = t2;
    }
  }
  return sobolstate;
}
# 初始化 Sobol 序列的状态，使用给定的 `sobolstate` 张量和维度参数
Tensor& _sobol_engine_initialize_state_(Tensor& sobolstate, int64_t dimension) {
  # 检查 `sobolstate` 的数据类型是否为 int64_t
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);

  /// 使用张量访问器访问 `sobolstate`
  auto ss_a = sobolstate.accessor<int64_t, 2>();

  /// 将 `sobolstate` 的第一行全部设为 1
  for (const auto m : c10::irange(MAXBIT)) {
    ss_a[0][m] = 1;
  }

  /// 填充 `sobolstate` 的其余行（从第二行到第 `dimension` 行）
  for (const auto d : c10::irange(1, dimension)) {
    int64_t p = poly[d];
    int64_t m = bit_length(p) - 1;

    // 行 d 的前 m 个元素来自 initsobolstate[d][i]
    for (const auto i : c10::irange(m)) {
      ss_a[d][i] = initsobolstate[d][i];
    }

    // 根据 Bratley 和 Fox 的算法填充 v 的其余元素，详见文献引用
    for (const auto j : c10::irange(m, MAXBIT)) {
      int64_t newv = ss_a[d][j - m];
      int64_t pow2 = 1;
      for (const auto k : c10::irange(m)) {
        pow2 <<= 1;
        if ((p >> (m - 1 - k)) & 1) {
          newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
        }
      }
      ss_a[d][j] = newv;
    }
  }

  /// 将 `sobolstate` 的每一列乘以 2 的幂次方：
  /// sobolstate * [2^(maxbit-1), 2^(maxbit-2),..., 2, 1]
  Tensor pow2s = at::pow(
      2,
      at::native::arange(
          (MAXBIT - 1),
          -1,
          -1,
          optTypeMetaToScalarType(sobolstate.options().dtype_opt()),
          sobolstate.options().layout_opt(),
          sobolstate.options().device_opt(),
          sobolstate.options().pinned_memory_opt()));
  sobolstate.mul_(pow2s);
  return sobolstate;
}
```