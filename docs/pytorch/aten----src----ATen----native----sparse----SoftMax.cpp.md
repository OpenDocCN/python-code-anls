# `.\pytorch\aten\src\ATen\native\sparse\SoftMax.cpp`

```
  // 定义宏，仅在使用方法运算符时启用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
  // 包含张量相关的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/native/SparseTensorUtils.h>
  // 包含并行处理相关的头文件
#include <ATen/Parallel.h>
  // 包含累加工具的头文件
#include <c10/util/accumulate.h>
  // 包含范围迭代工具的头文件
#include <c10/util/irange.h>

  // 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 CPU 相关的函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
  // 否则，包含特定运算的头文件
#else
#include <ATen/ops/_log_softmax_backward_data_cpu_dispatch.h>
#include <ATen/ops/_log_softmax_cpu_dispatch.h>
#include <ATen/ops/_softmax_backward_data_cpu_dispatch.h>
#include <ATen/ops/_softmax_cpu_dispatch.h>
#include <ATen/ops/_sparse_log_softmax.h>
#include <ATen/ops/_sparse_log_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_log_softmax_native.h>
#include <ATen/ops/_sparse_softmax.h>
#include <ATen/ops/_sparse_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_softmax_native.h>
#endif

  // 包含标准库中的 map 头文件
#include <map>

  // at::native 命名空间
namespace at::native {
  // 匿名命名空间，用于实现内部功能
namespace {

  // 获取稀疏张量中稠密部分的条目数
int64_t get_nvalues(const IntArrayRef& sizes, int64_t sparse_dim) {
  /* 返回稀疏张量稠密部分的条目数。

     `sizes` 是一个稀疏张量维度的向量。
     `sparse_dim` 是稀疏张量的稀疏部分的维度。
   */
  return c10::multiply_integers(sizes.begin() + sparse_dim, sizes.end());
}

  // 获取稀疏张量索引的偏移量向量
std::vector<int64_t> get_offsets(const Tensor& indices, const IntArrayRef& sizes, const int64_t dim) {
  /*
    给定稀疏张量的索引，返回等效稠密张量中条目的偏移向量：

      如果
        offsets = get_offsets(A._indices(), A.sizes(), -1)
        data = A.to_dense().resize((nnz,))
      那么
        data[offsets[n]] == A._values()[n]

    `indices` 必须是一个连续的二维张量，其元素类型为 int64_t。
    `sizes` 必须是一个至少包含 ndim 个条目的向量。

    `dim` 是一个整数。当 >= 0 且 < ndim 时，将映射给定维度中所有条目的索引到计算偏移之前的第一个条目的索引。
    否则，忽略该值。

    例如，考虑一个稀疏张量

      11 ** ** 14 15
      ** 22 ** 24 **

    其中

      indices = [[0, 0, 0, 1, 1],
                 [0, 3, 4, 1, 3]]

    则

      get_offsets(indices, (2, 5), -1) -> [0, 3, 4, 6, 8]
      get_offsets(indices, (2, 5), 0) -> [0, 3, 4, 1, 3]
      get_offsets(indices, (2, 5), 1) -> [0, 0, 0, 5, 5]

  */
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<int64_t> offsets(nnz);
  std::vector<int64_t> strides(ndim, 1);
  auto indices_accessor = indices.accessor<int64_t, 2>();

  if (ndim > 1) {
    for (int64_t i=ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  for (const auto i : c10::irange(nnz)) {
    int64_t acc = 0;
    // 对于每个维度的循环迭代
    for (const auto j : c10::irange(ndim)) {
      // 获取当前维度的索引数组
      auto indices_row = indices_accessor[j];
      // 获取当前维度的步长
      auto stride = strides[j];
      // 如果当前维度不是指定的dim维度
      if (j != dim) {
        // 累加计算偏移量
        acc += stride * indices_row[i];
      }
    }
    // 将计算出的偏移量存入偏移量数组中
    offsets[i] = acc;
  }

  // 返回计算出的偏移量数组
  return offsets;
}

std::vector<std::vector<int64_t>> get_pools(const Tensor& indices, const IntArrayRef& sizes, const int64_t dim) {
  /*
    返回与给定维度对齐的索引池。

    参数:
      `indices` - 稀疏张量的索引
      `sizes`   - 稀疏张量的维度
      `dim`     - 给定的维度

    返回:
      `pools`   - 索引的一个不规则数组

    索引池被定义为参与相同 softmax 计算的索引列表:

    - 若 i != j，则 pools[i] 和 pools[j] 的交集为空
    - 所有池的并集是 set(range(nnz))
    - X.values[k]，k 在 pools[i] 中，不影响 softmax(X)[n] 的结果，其中 n 在 pools[j] 中，若 i != j
  */
  std::vector<std::vector<int64_t>> pools;

  auto ndim = indices.size(0);  // 获取张量的维数
  auto nnz = indices.size(1);   // 获取张量中非零元素的数量
  std::vector<int64_t> strides(ndim, 1);  // 初始化步长数组
  auto indices_accessor = indices.accessor<int64_t, 2>();  // 获取索引的访问器

  if (ndim > 1) {
    for (int64_t i=ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * (i + 1 == dim? 1 : sizes[i + 1]);  // 计算步长
    }
  }

  for (const auto i : c10::irange(nnz)) {  // 遍历所有非零元素的索引
    int64_t pool_index = 0;
    for (const auto j : c10::irange(ndim)) {  // 遍历所有维度
      if (j != dim) {
        const auto indices_row = indices_accessor[j];  // 获取当前维度的索引行
        const auto stride = strides[j];  // 获取当前维度的步长
        pool_index += stride * indices_row[i];  // 计算索引池中的索引
      }
    }
    if(static_cast<int64_t>(pools.size()) <= pool_index){
      pools.resize(pool_index + 1);  // 调整索引池的大小
    }
    pools.at(pool_index).push_back(i);  // 将索引加入到相应的索引池中
  }

  return pools;  // 返回索引池
}

template <typename scalar_t, bool LogSoftMax>
void cpu_sparse_coo_softmax(Tensor output, const Tensor& input, const int64_t dim) {
  /*
    请参考 test/test_sparse.py:test_softmax:sparse_softmax 中的 Python
    原型，了解本实现基于的稀疏 softmax 算法。

    稀疏 softmax 算法推导与示例
    ----------------------------------------------------------

    考虑以下 2-D 稀疏张量与 0-D 密集部分作为示例，记为 X:

      11 ** ** 14 15
      ** 22 ** 24 **

    其 COO 稀疏张量表示为:

      indices = [[0, 1, 0, 1, 0],
                 [0, 1, 3, 3, 4]]
      values = [11, 22, 14, 24, 15]

    在合并后变为

      indices = [[0, 0, 0, 1, 1],
                 [0, 3, 4, 1, 3]]
      values = [11, 14, 15, 22, 24]

    沿着给定维度 d 的 X 的 softmax 定义为

      S_d[i, j] = exp(X[i, j]) / sum(exp(X[I_d[k]]), k=0..X.shape[d]-1)

    其中索引元组 I_d[k] 定义为

      I_0[k] = k, j
      I_1[k] = i, k

    对于稀疏张量，未指定的条目在 softmax 指数和中被跳过，因此结果将是具有与输入相同索引的稀疏张量。在数学上，这对应于将未指定的条目解释为负无穷，而不是零。

  */
}
    # 为了最小化对具有非常大或非常小参数的指数计算结果的缺陷，
    # softmax 实现采用以下数值稳定的定义：
    #
    #   S_d[i, j] = exp(X[i, j] - maxX_d) / sum(exp(X[I_d[k]] - maxX_d), k=0...X.shape[d]-1)
    #
    # 其中
    #
    #   maxX_d = max(X[I_d[k]], k=0...X.shape[d]-1)
    #
    # 是沿着方向 d 的最大张量（其维度为 `maxX_d.ndim = X.ndim - 1`）。
    #
    # 对于示例稀疏张量 X，我们有：
    #
    #   S_0._indices() == S_1._indices() == X._indices()
    #
    #   maxX_0 = [11, 22, -inf, 24, 15]
    #   maxX_1 = [15, 24]
    #
    #   S_0._values() = [1, exp(-10)/(exp(-10) + 1), 1, 1, 1/(exp(-10) + 1)]
    #
    #   S_1._values() = [exp(-4) / (exp(-4) + exp(-1) + 1),
    #                    exp(-1) / (exp(-4) + exp(-1) + 1),
    #                    1 / (exp(-4) + exp(-1) + 1),
    #                    exp(-2) / (exp(-2) + 1),
    #                    1 / (exp(-2) + 1)]
    #
    # 为了通过 `nnz(=len(X._values()))` 的 for 循环获得上述结果，
    # 我们引入了指数映射 `pool`，如下所示：
    #
    #   indices = X._indices()
    #   for i in range(nnz):
    #       for j in range(nnz):
    #           if indices[d, i] == indices[d, j]:
    #               assert pool_d[i] == pool_d[j]
    #           else:
    #               assert pool_d[i] != pool_d[j]
    #
    # 即，具有值 indices i 和 j 的条目在张量索引网格中沿 softmax 计算方向对齐时属于相同的 pool。
    # `pool` 映射将 X._values() 的索引映射到相应的 pool 索引。
    #
    # 为了节省内存和处理器资源，我们预先计算 maxX 张量的条目和指数和，如下所示：
    #
    #   mx_d = [max(values[i] for i in range(nnz) if pool_0[i] == k) for k in pool_d]
    #   exp_sum_d = [sum(exp(values[i] - mx_d[k]) for i in range(nnz) if pool_d[i] == k) for k in pool_d]
    #
    # 例如，如果
    #
    #   pool_0 = [0, 1, 2, 3, 1]
    #   pool_1 = [0, 0, 0, 1, 1]
  /*
    获取稀疏张量的维度信息
    auto sparse_dim = input.sparse_dim();
    获取稀疏张量的索引并确保其连续性
    auto indices = input._indices().contiguous();
    获取稀疏张量的值并确保其连续性
    auto values = input._values().contiguous();
    获取输出张量的值
    auto out_values = output._values();
    获取输出张量的索引
    auto out_indices = output._indices();
    调整输出张量的值的大小与输入值相同
    out_values.resize_as_(values);
    调整输出张量的索引的大小与输入索引相同
    out_indices.resize_as_(indices);
    将输入索引复制到输出索引
    out_indices.copy_(indices);

    如果指定的维度超过稀疏张量的维度数，执行以下操作：
    if (dim >= sparse_dim) {
        如果需要进行 LogSoftMax 操作：
        if (LogSoftMax) {
            执行在指定维度上的 log_softmax 操作，不计算梯度
            auto new_values =
                at::cpu::_log_softmax(values, dim - sparse_dim + 1, false);
            将新的值设置为输出张量的值
            out_values.set_(new_values);
        } else {
            执行在指定维度上的 softmax 操作，不计算梯度
            auto new_values = at::cpu::_softmax(values, dim - sparse_dim + 1, false);
            将新的值设置为输出张量的值
            out_values.set_(new_values);
        }
    }
  */
    return;
  }

  auto nnz = values.size(0);  // 获取稀疏张量的非零元素数量
  auto sizes = input.sizes();  // 获取输入张量的维度大小信息
  auto nvalues = get_nvalues(sizes, sparse_dim);  // 计算每个稀疏张量元素的值数量

  /* Prepare accessors */
  auto values_2 = values.view({nnz, nvalues});  // 将稀疏张量的值重塑为二维数组
  auto values_accessor = values_2.accessor<scalar_t, 2>();  // 获取稀疏张量值的访问器

  auto out_values_2 = out_values.view({nnz, nvalues});  // 将输出张量的值重塑为二维数组
  auto out_values_accessor = out_values_2.accessor<scalar_t, 2>();  // 获取输出张量值的访问器

  /* Compute independent pools of indices */
  auto pools = get_pools(indices, sizes, dim);  // 计算独立的索引池子

  int64_t grain_size = 1;  // 设置并行任务的粒度大小
  parallel_for(0, pools.size(), grain_size, [&](int64_t begin, int64_t end) {  // 并行化循环处理索引池
      for (const auto p : c10::irange(begin, end)) {  // 遍历每个索引池
        auto pool_indices = pools[p];  // 获取当前索引池的索引集合

        // Skip empty pools
        if (pool_indices.empty())  // 如果当前索引池为空，则跳过处理
          continue;

        /* Prepare scratch space */
        std::vector<scalar_t> mx_row(nvalues, -std::numeric_limits<scalar_t>::infinity());  // 初始化临时存储空间用于计算最大值
        std::vector<scalar_t> exp_sums_row(nvalues, 0);  // 初始化临时存储空间用于计算指数和

        /* Compute mx */
        for (int64_t i : pool_indices) {  // 遍历当前索引池中的每个索引
          auto values_row = values_accessor[i];  // 获取稀疏张量中对应索引的数值行
          for (const auto j : c10::irange(nvalues)) {  // 遍历数值行中的每个值
            mx_row[j] = std::max(mx_row[j], values_row[j]);  // 更新最大值数组中的每个值
          }
        }

        /* Apply exp to (v - mx) and sum the results */
        for (int64_t i : pool_indices) {  // 再次遍历当前索引池中的每个索引
          auto values_row = values_accessor[i];  // 获取稀疏张量中对应索引的数值行
          auto out_values_row = out_values_accessor[i];  // 获取输出张量中对应索引的数值行
          for (const auto j : c10::irange(nvalues)) {  // 遍历数值行中的每个值
            auto v = std::exp(values_row[j] - mx_row[j]);  // 计算指数函数应用于（v - mx）
            if (!LogSoftMax) {  // 如果非 LogSoftMax 操作
              out_values_row[j] = v;  // 将结果写入输出张量的对应位置
            }
            exp_sums_row[j] += v;  // 更新指数和数组中的每个值
          }
        }

        for (const auto j : c10::irange(nvalues)) {  // 遍历每个值的位置
          if (LogSoftMax) {  // 如果是 LogSoftMax 操作
            mx_row[j] += std::log(exp_sums_row[j]);  // 对最大值数组中的每个值应用对数函数
          } else {  // 否则
            exp_sums_row[j] = 1.0 / exp_sums_row[j];  // 计算每个值的倒数并更新指数和数组
          }
        }

        /* Normalize with the sum of exponents */
        for (int64_t i : pool_indices) {  // 再次遍历当前索引池中的每个索引
          auto values_row = values_accessor[i];  // 获取稀疏张量中对应索引的数值行
          auto out_values_row = out_values_accessor[i];  // 获取输出张量中对应索引的数值行
          for (const auto j : c10::irange(nvalues)) {  // 遍历数值行中的每个值
            if (LogSoftMax) {  // 如果是 LogSoftMax 操作
              out_values_row[j] = values_row[j] - mx_row[j];  // 对输出张量中的每个值进行减法操作
            } else {  // 否则
              out_values_row[j] *= exp_sums_row[j];  // 对输出张量中的每个值乘以指数和数组中的对应值
            }
          }
        }
      }
    });
  /*

    如果 LogSoftMax == false，那么

      gI_i = sum_j d<output_j>/d<input_i> * grad_j = sum_j output_i * (1[i==j] - output_j) * grad_j
           = output_i * (grad_i - sum_j output_j * grad_j)

    否则

      gI_i = (1-exp(output_i)) * grad_i - sum_{j} 1[i!=j] * exp(output_i) * grad_j
           = grad_i - exp(output_i) * sum_j grad_j.

    其中

      i, j 在 range(shape[dim]) 范围内
      x_i = x[..., i_dim, ...]
      output.sparse_dim() == grad.sparse_dim()
  */
  auto sparse_dim = output.sparse_dim();  // 获取输出张量的稀疏维度
  auto sizes = output.sizes().vec();  // 获取输出张量的尺寸信息
  auto grad_indices = grad._indices().contiguous();  // 获取梯度张量的稀疏索引，并保证是连续的
  auto grad_values = grad._values().contiguous();  // 获取梯度张量的值，并保证是连续的
  auto out_indices = output._indices().contiguous();  // 获取输出张量的稀疏索引，并保证是连续的
  auto out_values = output._values().contiguous();  // 获取输出张量的值，并保证是连续的
  auto values = grad_input._values();  // 获取梯度输入张量的值
  auto indices = grad_input._indices();  // 获取梯度输入张量的索引
  auto out_nnz = out_values.size(0);  // 获取输出张量的非零元素数量
  auto grad_nnz = grad_values.size(0);  // 获取梯度张量的非零元素数量

  values.resize_as_(out_values);  // 调整梯度输入值的大小，使其与输出值相同
  values.zero_();  // 将梯度输入值初始化为零
  indices.resize_as_(out_indices);  // 调整梯度输入索引的大小，使其与输出索引相同，并复制输出索引的值

  auto out_offsets = get_offsets(out_indices, sizes, -1);  // 计算输出张量的偏移量
  auto grad_offsets = get_offsets(grad_indices, sizes, -1);  // 计算梯度张量的偏移量

  if (dim >= sparse_dim) {  // 如果指定的维度大于等于稀疏维度
    if (out_offsets == grad_offsets) {  // 如果输出张量和梯度张量的偏移量相同
      if (LogSoftMax) {  // 如果是对数softmax
        auto r = at::cpu::_log_softmax_backward_data(
            grad_values, out_values, dim - sparse_dim + 1, input_dtype);  // 计算对数softmax的反向传播数据
        values.set_(r);  // 设置梯度输入值为计算得到的数据
      } else {  // 如果是softmax
        auto r = at::cpu::_softmax_backward_data(grad_values, out_values, dim - sparse_dim + 1, input_dtype);  // 计算softmax的反向传播数据
        values.set_(r);  // 设置梯度输入值为计算得到的数据
      }
    } else {  // 如果输出张量和梯度张量的偏移量不同
      for (const auto i : c10::irange(out_nnz)) {  // 遍历输出张量中的非零元素
        auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);  // 在梯度张量的偏移量中寻找大于等于当前输出张量元素偏移量的最小值
        auto j = low - grad_offsets.begin();  // 计算找到的位置索引
        if (j < grad_nnz && out_offsets[i] == grad_offsets[j]) {  // 如果找到了匹配的偏移量
          if (LogSoftMax) {  // 如果是对数softmax
            auto r = at::cpu::_log_softmax_backward_data(
                grad_values[j], out_values[i], dim - sparse_dim, input_dtype);  // 计算对数softmax的反向传播数据
            values[i].copy_(r);  // 复制计算得到的数据到梯度输入值的对应位置
          } else {  // 如果是softmax
            auto r = at::cpu::_softmax_backward_data(grad_values[j], out_values[i], dim - sparse_dim, input_dtype);  // 计算softmax的反向传播数据
            values[i].copy_(r);  // 复制计算得到的数据到梯度输入值的对应位置
          }
        }
      }
    }
    return;
  }

  auto nnz = values.size(0);  // 获取稀疏张量的非零值数量
  auto nvalues = get_nvalues(sizes, sparse_dim);  // 计算稀疏张量中每个非零值的元素数量

  auto values_2 = values.view({nnz, nvalues});  // 将稀疏张量的值重新视图为二维张量
  auto values_accessor = values_2.accessor<scalar_t, 2>();  // 创建二维张量的访问器

  auto out_values_2 = out_values.view({out_nnz, nvalues});  // 将输出值的张量视图为二维张量
  auto out_values_accessor = out_values_2.accessor<scalar_t, 2>();  // 创建输出值的二维张量访问器

  auto grad_values_2 = grad_values.view({grad_nnz, nvalues});  // 将梯度值的张量视图为二维张量
  auto grad_values_accessor = grad_values_2.accessor<scalar_t, 2>();  // 创建梯度值的二维张量访问器

  /* Compute independent pools of indices */
  auto pools = get_pools(out_indices, sizes, dim);  // 根据输出索引、大小和维度获取独立的索引池

  int64_t grain_size = 1;  // 设置并行任务的粒度大小为1
  parallel_for(0, pools.size(), grain_size, [&](int64_t begin, int64_t end) {  // 并行循环处理索引池
      for (const auto p : c10::irange(begin, end)) {  // 遍历每个索引池
        auto pool_indices = pools[p];  // 获取当前索引池的索引列表

        // Skip empty pools
        if (pool_indices.empty())  // 如果索引池为空，则跳过
          continue;

        std::vector<scalar_t> tmp_row(nvalues, 0);  // 创建临时向量 tmp_row，初始值为0

        /* Compute tmp = - sum_j output_j * grad_j */
        for (int64_t i : pool_indices) {  // 遍历当前索引池中的每个索引
          auto out_values_row = out_values_accessor[i];  // 获取输出值的当前行
          auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);  // 在梯度偏移量中查找输出偏移量的下界
          auto j = low - grad_offsets.begin();  // 计算对应的梯度索引位置

          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {  // 如果找到匹配的梯度偏移量
            auto grad_values_row = grad_values_accessor[j];  // 获取梯度值的当前行
            for (const auto k : c10::irange(nvalues)) {  // 遍历每个元素值
              if (LogSoftMax) {  // 如果使用 LogSoftMax 操作
                tmp_row[k] -= grad_values_row[k];  // 更新临时向量中的值
              } else {
                tmp_row[k] -= out_values_row[k] * grad_values_row[k];  // 更新临时向量中的值
              }
            }
          }
        }

        /* Compute grad_input = output * (grad + tmp)*/
        for (int64_t i : pool_indices) {  // 再次遍历当前索引池中的每个索引
          auto out_values_row = out_values_accessor[i];  // 获取输出值的当前行
          auto values_row = values_accessor[i];  // 获取稀疏张量的当前行
          auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);  // 在梯度偏移量中查找输出偏移量的下界
          auto j = low - grad_offsets.begin();  // 计算对应的梯度索引位置

          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {  // 如果找到匹配的梯度偏移量
            auto grad_values_row = grad_values_accessor[j];  // 获取梯度值的当前行
            for (const auto k : c10::irange(nvalues)) {  // 遍历每个元素值
              if (LogSoftMax) {  // 如果使用 LogSoftMax 操作
                values_row[k] = grad_values_row[k] + std::exp(out_values_row[k]) * tmp_row[k];  // 更新稀疏张量的当前行
              } else {
                values_row[k] = out_values_row[k] * (grad_values_row[k] + tmp_row[k]);  // 更新稀疏张量的当前行
              }
            }
          } else {
            for (const auto k : c10::irange(nvalues)) {  // 如果未找到匹配的梯度偏移量，则仅考虑输出值和临时向量
              if (LogSoftMax) {  // 如果使用 LogSoftMax 操作
                values_row[k] = std::exp(out_values_row[k]) * tmp_row[k];  // 更新稀疏张量的当前行
              } else {
                values_row[k] = out_values_row[k] * (tmp_row[k]);  // 更新稀疏张量的当前行
              }
            }
          }
        }
      }
    });
}

} // anonymous namespace

// 计算稀疏张量在指定维度上的 softmax
Tensor softmax_sparse_cpu(
    const Tensor& input_,      // 输入稀疏张量
    const int64_t dim_,        // 指定的维度
    const bool half_to_float)  // 是否将 Half 类型转换为 Float
{
  Tensor input, output;       // 输入张量和输出张量
  int64_t dim;                // 维度
  // 预处理输入并获取输出和维度
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "softmax");
  // 如果输入张量元素数量为 0，则直接返回输出张量
  if (input.numel() == 0) {
    return output;
  }
  // 根据输入张量的数据类型调度对应的 CPU 稀疏 COO softmax 函数
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
    cpu_sparse_coo_softmax<scalar_t, false>(output, input, dim);
  });
  // 返回计算后的输出张量
  return output;
}

// 计算稀疏张量在指定维度上的 log_softmax
Tensor log_softmax_sparse_cpu(
    const Tensor& input_,      // 输入稀疏张量
    const int64_t dim_,        // 指定的维度
    const bool half_to_float)  // 是否将 Half 类型转换为 Float
{
  Tensor input, output;       // 输入张量和输出张量
  int64_t dim;                // 维度
  // 预处理输入并获取输出和维度
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "log_softmax");
  // 如果输入张量元素数量为 0，则直接返回输出张量
  if (input.numel() == 0) {
    return output;
  }
  // 根据输入张量的数据类型调度对应的 CPU 稀疏 COO log_softmax 函数
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
    cpu_sparse_coo_softmax<scalar_t, true>(output, input, dim);
  });
  // 返回计算后的输出张量
  return output;
}

// 计算稀疏张量在指定维度上的 softmax 反向传播
Tensor softmax_backward_sparse_cpu(
    const Tensor& grad_,       // 梯度张量
    const Tensor& output_,     // 输出张量
    int64_t dim_,              // 指定的维度
    const Tensor& input_)      // 输入张量
{
  Tensor grad_input, grad, output;  // 梯度输入张量、梯度张量和输出张量
  int64_t dim;                      // 维度
  // 预处理输入并获取梯度输入、梯度和输出以及维度
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "softmax_backward");
  // 如果输出张量元素数量为 0，则直接返回梯度输入张量
  if (output.numel() == 0) {
    return grad_input;
  }
  // 根据梯度张量的数据类型调度对应的 CPU 稀疏 COO softmax 反向传播函数
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
    cpu_sparse_coo_softmax_backward<scalar_t, false>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  // 返回计算后的梯度输入张量
  return grad_input;
}

// 计算稀疏张量在指定维度上的 log_softmax 反向传播
Tensor log_softmax_backward_sparse_cpu(
    const Tensor& grad_,       // 梯度张量
    const Tensor& output_,     // 输出张量
    int64_t dim_,              // 指定的维度
    const Tensor& input_)      // 输入张量
{
  Tensor grad_input, grad, output;  // 梯度输入张量、梯度张量和输出张量
  int64_t dim;                      // 维度
  // 预处理输入并获取梯度输入、梯度和输出以及维度
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "log_softmax_backward");
  // 如果输出张量元素数量为 0，则直接返回梯度输入张量
  if (output.numel() == 0) {
    return grad_input;
  }
  // 根据梯度张量的数据类型调度对应的 CPU 稀疏 COO log_softmax 反向传播函数
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "log_softmax_backward", [&] {
    cpu_sparse_coo_softmax_backward<scalar_t, true>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  // 返回计算后的梯度输入张量
  return grad_input;
}

// 计算稀疏张量在指定维度上的 softmax
Tensor _sparse_softmax(
    const Tensor& input_,       // 输入张量
    const int64_t dim_,         // 指定的维度
    std::optional<ScalarType> dtype) // 可选的数据类型
{
  // 使用 lambda 函数计算结果
  auto result = [&]() {
    NoNamesGuard guard;
    // 如果输入张量在 CUDA 上且数据类型为 Half，且需要转换为 Float，则调用对应函数
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_sparse_softmax(input_, dim_, true);
    } else {
        // 否则，根据是否指定数据类型转换输入张量
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        // 调用对应函数计算 softmax
        return at::_sparse_softmax(converted, dim_, false);
    }
  }();
  // 将输入张量的命名信息传播到结果张量
  namedinference::propagate_names(result, input_);
  // 返回计算后的结果张量
  return result;
}

// 计算稀疏张量在指定维度上的 softmax
Tensor _sparse_softmax(
    const Tensor& self,        // 输入张量
    Dimname dim,               // 维度名
    optional<ScalarType> dtype) // 可选的数据类型
{
  // 调用第一个 _sparse_softmax 函数，将维度名转换为维度位置
  return at::_sparse_softmax(self, dimname_to_position(self, dim), dtype);
}
// 定义函数 _sparse_log_softmax，接受输入张量 input_，维度 dim_，以及可选的数据类型 dtype
Tensor _sparse_log_softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  // 使用 lambda 表达式定义 result 变量，用于根据条件生成不同的输出张量
  auto result = [&]() {
    // 定义 NoNamesGuard 对象 guard，用于暂时禁用命名推断
    NoNamesGuard guard;
    // 如果输入张量在 CUDA 上，并且标量类型为 ScalarType::Half，并且 dtype 指定为 ScalarType::Float
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        // 返回对输入张量进行稀疏对数 softmax 操作的结果
        return at::_sparse_log_softmax(input_, dim_, true);
    } else {
        // 否则，将输入张量转换为指定的数据类型 dtype，或者保持不变
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        // 返回对转换后的张量进行稀疏对数 softmax 操作的结果
        return at::_sparse_log_softmax(converted, dim_, false);
    }
  }(); // 立即调用 lambda 表达式，得到 result 变量的值

  // 将 result 的命名信息从 input_ 中传播到 result 上
  namedinference::propagate_names(result, input_);

  // 返回执行稀疏对数 softmax 操作后得到的结果张量 result
  return result;
}

// 定义函数 _sparse_log_softmax，接受输入张量 self，维度 dim 的命名，以及可选的数据类型 dtype
Tensor _sparse_log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  // 调用 _sparse_log_softmax 函数，将 dimname 转换为在 self 张量中的位置索引，然后进行稀疏对数 softmax 操作
  return at::_sparse_log_softmax(self, dimname_to_position(self, dim), dtype);
}

// 结束 namespace at::native
} // namespace at::native
```