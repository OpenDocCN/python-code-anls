# `.\pytorch\aten\src\ATen\native\sparse\SparseBinaryOpIntersectionKernel.cpp`

```
    // 定义一个宏，指示仅包含方法操作符
    #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
    // 包含稀疏张量相关的头文件
    #include <ATen/native/sparse/SparseStubs.h>
    #include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
    // 包含CPU循环相关的头文件
    #include <ATen/native/cpu/Loops.h>
    // 包含张量迭代器相关的头文件
    #include <ATen/native/TensorIterator.h>
    // 包含积累类型相关的头文件
    #include <ATen/AccumulateType.h>

    // 命名空间：at::native
    namespace at::native {

    // 匿名命名空间，定义一些局部使用的辅助结构和函数
    namespace {

    // CPUKernelLauncher 结构模板
    template <typename func_t>
    struct CPUKernelLauncher {
      // launch 方法：执行传入的函数 func_t 在迭代器 iter 上
      static void launch(TensorIteratorBase& iter, const func_t& f) {
        cpu_kernel(iter, f);
      }
    };

    // MulOp 结构：乘法操作符结构
    struct MulOp {
      // apply 方法模板：应用于标量类型 scalar_t 的乘法操作
      template <typename scalar_t>
      static scalar_t apply(scalar_t a, scalar_t b) {
        return a * b;
      }
    };

    // 特化 MulOp 结构对布尔类型的 apply 方法
    template <>
    bool MulOp::apply(bool a, bool b) {
      return a && b;
    }

    // RhsProjOp 结构：右侧投影操作符结构
    struct RhsProjOp {
      // apply 方法模板：应用于标量类型 scalar_t 的右侧投影操作
      template <typename scalar_t>
      static scalar_t apply(scalar_t a, scalar_t b) {
        return b;
      }
    };

    // LhsProjOp 结构：左侧投影操作符结构
    struct LhsProjOp {
      // apply 方法模板：应用于标量类型 scalar_t 的左侧投影操作
      template <typename scalar_t>
      static scalar_t apply(scalar_t a, scalar_t b) {
        return a;
      }
    };

    // CPUValueSelectionIntersectionKernel 结构模板
    template <typename binary_op_t>
    struct CPUValueSelectionIntersectionKernel {
      // apply 方法：执行值选择交集的计算
      static Tensor apply(
          const Tensor& lhs_values,
          const Tensor& lhs_select_idx,
          const Tensor& rhs_values,
          const Tensor& rhs_select_idx,
          const Tensor& intersection_counts,
          const Tensor& argsort,
          const bool accumulate_matches) {
        // 创建值选择交集迭代器
        auto iter = make_value_selection_intersection_iter(
            lhs_values,
            lhs_select_idx,
            rhs_values,
            rhs_select_idx,
            intersection_counts);
        // 获取结果张量
        auto res_values = iter.tensor(0);

        // 计算左侧非零元素的步长
        auto lhs_nnz_stride = lhs_values.stride(0);
        // 计算右侧非零元素的步长
        auto rhs_nnz_stride = rhs_values.stride(0);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, at::ScalarType::ComplexHalf,
        res_values.scalar_type(),
        "binary_op_intersection_cpu", [&] {
            // COO indices are only 64-bit for now.
            // 定义索引类型为 int64_t
            using index_t = int64_t;
            // 定义循环函数 loop，接收数据指针和步长数组
            auto loop = [&](char** data, const int64_t* strides, int64_t n) {
              // 解析数据指针
              auto* ptr_res_values_bytes = data[0];
              const auto* ptr_lhs_values_bytes = data[1];
              const auto* ptr_lhs_select_idx_bytes = data[2];
              const auto* ptr_rhs_values_bytes = data[3];
              const auto* ptr_rhs_select_idx_bytes = data[4];
              const auto* ptr_intersection_counts_bytes = data[5];
              const auto* ptr_argsort = argsort.const_data_ptr<index_t>();

              // 循环处理每个元素
              for (int64_t i = 0; i < n; ++i) {
                // 提取数据
                auto* ptr_res_values = reinterpret_cast<scalar_t*>(ptr_res_values_bytes);
                const auto* ptr_lhs_values = reinterpret_cast<const scalar_t*>(ptr_lhs_values_bytes);
                const auto lhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_lhs_select_idx_bytes);
                const auto* ptr_rhs_values = reinterpret_cast<const scalar_t*>(ptr_rhs_values_bytes);
                const auto rhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_rhs_select_idx_bytes);
                const auto count = *reinterpret_cast<const int64_t*>(ptr_intersection_counts_bytes);

                // 计算指针偏移量
                const auto* ptr_lhs_begin = ptr_lhs_values + lhs_nnz_idx * lhs_nnz_stride;
                const auto* ptr_rhs_sorted_nnz_idx = ptr_argsort + rhs_nnz_idx;

                // 定义累加类型 accscalar_t，非 GPU
                using accscalar_t = at::acc_type<scalar_t, /*is_gpu=*/false>;
                accscalar_t res_values = 0;
                accscalar_t lhs_values = static_cast<accscalar_t>(*ptr_lhs_begin);
                accscalar_t rhs_values;
                index_t rhs_sorted_nnz_idx;
                // 计算匹配数量
                const auto match_count = accumulate_matches ? count : std::min<int64_t>(count, 1);
                // 循环处理匹配项
                for (int64_t c = 0; c < match_count; ++c) {
                  rhs_sorted_nnz_idx = *ptr_rhs_sorted_nnz_idx++;
                  rhs_values = static_cast<accscalar_t>(*(ptr_rhs_values + rhs_sorted_nnz_idx * rhs_nnz_stride));
                  res_values += binary_op_t::apply(lhs_values, rhs_values);
                }
                // 将结果保存到 res_values 中
                *ptr_res_values = static_cast<scalar_t>(res_values);

                // 更新指针位置
                ptr_res_values_bytes += strides[0];
                ptr_lhs_values_bytes += strides[1];
                ptr_lhs_select_idx_bytes += strides[2];
                ptr_rhs_values_bytes += strides[3];
                ptr_rhs_select_idx_bytes += strides[4];
                ptr_intersection_counts_bytes += strides[5];
              }
            };
            // 使用迭代器 iter 执行循环函数 loop，每次处理的元素数量为 GRAIN_SIZE
            iter.for_each(loop, at::internal::GRAIN_SIZE);
        });

    // 返回计算结果 res_values
    return res_values;
  }
};

using OptTensor = std::optional<Tensor>;

// 定义一个函数 mul_sparse_sparse_out_cpu_kernel，用于执行稀疏张量之间的乘法操作
void mul_sparse_sparse_out_cpu_kernel(
    Tensor& result,                 // 输出结果张量
    const Tensor& x,                // 输入稀疏张量 x
    const Tensor& y) {              // 输入稀疏张量 y
  // 定义 CPUValueSelectionMulKernel 类型，用于执行值选择与 MulOp 操作的交集
  using CPUValueSelectionMulKernel = CPUValueSelectionIntersectionKernel<MulOp>;
  // 调用 _sparse_binary_op_intersection_kernel_out 函数，使用 CPUKernelLauncher 启动器，执行乘法操作
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueSelectionMulKernel>(
      result, x, y
  );
}

// 定义一个函数 sparse_mask_intersection_out_cpu_kernel，用于执行稀疏掩码与张量的交集操作
void sparse_mask_intersection_out_cpu_kernel(
    Tensor& result,                         // 输出结果张量
    const Tensor& x,                        // 输入稀疏张量 x
    const Tensor& y,                        // 输入张量 y
    const OptTensor& x_hash_opt = c10::nullopt) {  // 可选的稀疏张量 x 的哈希值
  // 定义 CPUValueRhsProjKernel 类型，用于执行值选择与 RhsProjOp 操作的交集
  using CPUValueRhsProjKernel = CPUValueSelectionIntersectionKernel<RhsProjOp>;
  // 调用 _sparse_binary_op_intersection_kernel_out 函数，使用 CPUKernelLauncher 启动器，执行交集操作
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueRhsProjKernel>(
      result, x, y, x_hash_opt
  );
}

// 定义一个函数 sparse_mask_projection_out_cpu_kernel，用于执行稀疏掩码与张量的投影操作
void sparse_mask_projection_out_cpu_kernel(
    Tensor& result,                         // 输出结果张量
    const Tensor& x,                        // 输入稀疏张量 x
    const Tensor& y,                        // 输入张量 y
    const OptTensor& x_hash_opt,            // 可选的稀疏张量 x 的哈希值
    bool accumulate_matches) {              // 是否累加匹配项
  // 定义 CPUValueLhsProjKernel 类型，用于执行值选择与 LhsProjOp 操作的交集
  using CPUValueLhsProjKernel = CPUValueSelectionIntersectionKernel<LhsProjOp>;
  // 调用 _sparse_binary_op_intersection_kernel_out 函数，使用 CPUKernelLauncher 启动器，执行投影操作
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueLhsProjKernel>(
      result, x, y, x_hash_opt, c10::nullopt, accumulate_matches
  );
}

// 注册稀疏乘法操作的架构分发器
REGISTER_ARCH_DISPATCH(mul_sparse_sparse_out_stub, DEFAULT, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX512_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX2_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_VSX_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);

// 注册稀疏掩码与张量交集操作的架构分发器
REGISTER_ARCH_DISPATCH(sparse_mask_intersection_out_stub, DEFAULT, &sparse_mask_intersection_out_cpu_kernel);
REGISTER_AVX512_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel);
REGISTER_AVX2_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel);
REGISTER_VSX_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel);

// 注册稀疏掩码与张量投影操作的架构分发器
REGISTER_ARCH_DISPATCH(sparse_mask_projection_out_stub, DEFAULT, &sparse_mask_projection_out_cpu_kernel);
REGISTER_AVX512_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel);
REGISTER_AVX2_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel);
REGISTER_VSX_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel);
}
```