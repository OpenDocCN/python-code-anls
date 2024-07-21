# `.\pytorch\aten\src\ATen\native\sparse\SparseUnaryOps.cpp`

```
// 定义一个模板函数，用于执行稀疏张量的单目操作函数
template <typename Ufunc>
Tensor coalesced_unary_ufunc(const Tensor &self, const Ufunc &ufunc) {
  // 内部断言：确保输入张量是稀疏张量
  TORCH_INTERNAL_ASSERT(self.is_sparse());
  // 对输入稀疏张量进行压缩，以准备执行操作
  const auto input = self.coalesce();
  // 对压缩后的值执行单目操作函数，得到输出的值张量
  Tensor out_values = ufunc(input.values());
  // 构造新的稀疏 COO 张量，使用原始的索引、新的值和原始的选项
  Tensor result = at::_sparse_coo_tensor_with_dims_and_tensors(
      input.sparse_dim(),
      input.dense_dim(),
      input.sizes(),
      input.indices().clone(),
      out_values,
      input.options().dtype(out_values.scalar_type()),
      /*is_coalesced=*/ true);
  // 返回执行单目操作后得到的稀疏张量结果
  return result;
}
// 对稀疏张量执行一元操作，确保张量是稀疏的
Tensor& coalesced_unary_ufunc_(Tensor &self, const Ufunc &ufunc) {
  // 断言输入张量为稀疏张量
  TORCH_INTERNAL_ASSERT(self.is_sparse());
  // 获取稀疏张量的值
  auto values = self._values();
  // 应用给定的一元函数操作
  ufunc(values);
  // 返回自身张量
  return self;
}

// 在指定输出张量上执行一元操作，确保输入和输出张量都是稀疏的
template <typename Ufunc>
Tensor& coalesced_unary_ufunc_out(const Tensor &self, Tensor &result, const Ufunc &ufunc) {
  // 如果输入和输出张量是同一个对象
  if (self.is_same(result)) {
    // 检查输入张量是否已经是稀疏的，因为这是原地操作的先决条件
    TORCH_CHECK(self.is_coalesced(), "expected coalesced tensor for inplace operation");
    // 获取输入稀疏张量的值
    auto values = self._values();
    // 应用给定的一元函数操作到输入值上
    ufunc(values, values);
    // 返回结果张量
    return result;
  }

  // 如果输入和输出张量都是稀疏的
  TORCH_CHECK(self.is_sparse() && result.is_sparse());
  // 将输入张量稀疏化
  const auto input = self.coalesce();
  // 调整输出张量的大小和稀疏性维度
  sparse_resize_(result, input.sizes(), input.sparse_dim(), input.dense_dim());
  // 获取输入和输出稀疏张量的实现指针
  auto *input_impl = sparse::get_sparse_impl(input);
  auto *result_impl = sparse::get_sparse_impl(result);

  // 获取输入稀疏张量的值和输出稀疏张量的值
  auto input_values = input_impl->values();
  auto result_values = result_impl->values();
  // 调整输出稀疏张量值的大小以匹配输入稀疏张量值
  result_values.resize_(input_values.sizes());
  // 应用给定的一元函数操作到输入值上，并将结果存储到输出值中
  ufunc(input_values, result_values);

  // 获取输入稀疏张量的索引和输出稀疏张量的索引
  auto input_indices = input_impl->indices();
  auto result_indices = result_impl->indices();
  // 调整输出稀疏张量索引的大小以匹配输入稀疏张量索引
  result_indices.resize_(input_indices.sizes());
  // 复制输入稀疏张量的索引到输出稀疏张量的索引
  result_indices.copy_(input_indices);
  // 标记输出张量为稀疏的
  result._coalesced_(true);
  // 返回结果张量
  return result;
}

}  // namespace (anonymous)

// 通用的一元运算符形式，将 0 映射为 0，因此我们只需转换 self.values() 并保留稀疏模式。
//
// 任何非线性函数要求张量在计算结果之前被稀疏化。这也意味着只有在稀疏张量上才能进行原地计算。

// 定义宏来生成针对稀疏张量的一元操作函数

#define COALESCED_UNARY_UFUNC_FUNCTIONAL(op_name)   \
  Tensor op_name##_sparse(const Tensor &self) {     \
    // 返回调用 coalesced_unary_ufunc 函数的结果，传入一个 lambda 函数，将操作应用到输入张量上
    return coalesced_unary_ufunc(                   \
        self, [](const Tensor &t) {                 \
          return at::op_name(t);                    \
        });                                         \
  }

#define COALESCED_UNARY_UFUNC_NO_INPLACE(op_name)                       \
  COALESCED_UNARY_UFUNC_FUNCTIONAL(op_name)                             \
  // 返回调用 coalesced_unary_ufunc_out 函数的结果，传入一个 lambda 函数，将操作应用到输入和输出张量上
  Tensor& op_name##_sparse_out(const Tensor &self,                      \
                               Tensor &out) {                           \
    return coalesced_unary_ufunc_out(                                   \
        self, out, [](const Tensor &t, Tensor &out) {                   \
          return at::op_name##_outf(t, out);                            \
        });                                                             \
  }

#define COALESCED_UNARY_UFUNC(op_name)                                  \
  COALESCED_UNARY_UFUNC_NO_INPLACE(op_name)                             \
  // 返回调用 coalesced_unary_ufunc_ 函数的结果，传入一个 lambda 函数，将操作应用到输入张量上
  Tensor& op_name##_sparse_(Tensor &self) {                             \
    // 检查输入张量是否已经是稀疏的，因为这是原地操作的先决条件
    TORCH_CHECK(self.is_coalesced(),                                    \
                #op_name "_ requires coalesced input");                 \
    return coalesced_unary_ufunc_(self, [](Tensor &t) {                 \
      return t.op_name##_();                                            \
    });                                                                 \
  }


    // 这部分代码片段似乎是 C/C++ 的宏定义或者函数的结尾部分
    // `});` 可能是某个函数或代码块的结束标志
    // `\` 表示续行符，用于多行宏定义中的行连接
    // `}` 表示结束一个代码块或函数的定义
    // 因此这段代码的作用是结束一个函数或者代码块的定义
// 将 COALESCED_UNARY_UFUNC 宏应用于 abs 函数
COALESCED_UNARY_UFUNC(abs);

// 将 COALESCED_UNARY_UFUNC 宏应用于 asin 函数
COALESCED_UNARY_UFUNC(asin);

// 将 COALESCED_UNARY_UFUNC 宏应用于 asinh 函数
COALESCED_UNARY_UFUNC(asinh);

// 将 COALESCED_UNARY_UFUNC 宏应用于 atan 函数
COALESCED_UNARY_UFUNC(atan);

// 将 COALESCED_UNARY_UFUNC 宏应用于 atanh 函数
COALESCED_UNARY_UFUNC(atanh);

// 将 COALESCED_UNARY_UFUNC 宏应用于 ceil 函数
COALESCED_UNARY_UFUNC(ceil);

// 将 COALESCED_UNARY_UFUNC 宏应用于 deg2rad 函数
COALESCED_UNARY_UFUNC(deg2rad);

// 将 COALESCED_UNARY_UFUNC 宏应用于 erf 函数
COALESCED_UNARY_UFUNC(erf);

// 将 COALESCED_UNARY_UFUNC 宏应用于 erfinv 函数
COALESCED_UNARY_UFUNC(erfinv);

// 将 COALESCED_UNARY_UFUNC 宏应用于 expm1 函数
COALESCED_UNARY_UFUNC(expm1);

// 将 COALESCED_UNARY_UFUNC 宏应用于 floor 函数
COALESCED_UNARY_UFUNC(floor);

// 将 COALESCED_UNARY_UFUNC 宏应用于 frac 函数
COALESCED_UNARY_UFUNC(frac);

// 将 COALESCED_UNARY_UFUNC 宏应用于 log1p 函数
COALESCED_UNARY_UFUNC(log1p);

// 将 COALESCED_UNARY_UFUNC 宏应用于 round 函数
COALESCED_UNARY_UFUNC(round);

// 将 COALESCED_UNARY_UFUNC 宏应用于 rad2deg 函数
COALESCED_UNARY_UFUNC(rad2deg);

// 将 COALESCED_UNARY_UFUNC 宏应用于 sign 函数
COALESCED_UNARY_UFUNC(sign);

// 将 COALESCED_UNARY_UFUNC 宏应用于 sgn 函数
COALESCED_UNARY_UFUNC(sgn);

// 将 COALESCED_UNARY_UFUNC 宏应用于 sin 函数
COALESCED_UNARY_UFUNC(sin);

// 将 COALESCED_UNARY_UFUNC 宏应用于 sinh 函数
COALESCED_UNARY_UFUNC(sinh);

// 将 COALESCED_UNARY_UFUNC 宏应用于 sqrt 函数
COALESCED_UNARY_UFUNC(sqrt);

// 将 COALESCED_UNARY_UFUNC 宏应用于 tan 函数
COALESCED_UNARY_UFUNC(tan);

// 将 COALESCED_UNARY_UFUNC 宏应用于 tanh 函数
COALESCED_UNARY_UFUNC(tanh);

// 将 COALESCED_UNARY_UFUNC 宏应用于 trunc 函数
COALESCED_UNARY_UFUNC(trunc);

// 对于 relu 函数，没有声明，可能在 Pytorch 中未使用，暂时忽略警告
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
COALESCED_UNARY_UFUNC(relu);
#pragma clang diagnostic pop

// 将 COALESCED_UNARY_UFUNC_NO_INPLACE 宏应用于 signbit 函数
COALESCED_UNARY_UFUNC_NO_INPLACE(signbit);

// 将 COALESCED_UNARY_UFUNC_NO_INPLACE 宏应用于 isneginf 函数
COALESCED_UNARY_UFUNC_NO_INPLACE(isneginf);

// 将 COALESCED_UNARY_UFUNC_NO_INPLACE 宏应用于 isposinf 函数
COALESCED_UNARY_UFUNC_NO_INPLACE(isposinf);

// 将 COALESCED_UNARY_UFUNC_FUNCTIONAL 宏应用于 isnan 函数
COALESCED_UNARY_UFUNC_FUNCTIONAL(isnan);

// 将 COALESCED_UNARY_UFUNC_FUNCTIONAL 宏应用于 isinf 函数
COALESCED_UNARY_UFUNC_FUNCTIONAL(isinf);

// 返回一个断言，指示不支持稀疏元数据的 isinf 函数
Tensor isinf_sparse_meta(const Tensor& self) {
  TORCH_CHECK_NOT_IMPLEMENTED(0, "nyi isinf for SparseMeta");
}

// 返回阈值后向传播的稀疏版本，用于处理 relu 函数的反向传播
Tensor threshold_backward_sparse(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  // 根据梯度和输入稀疏性判断是否需要返回零矩阵
  const auto grad = [&]() {
    if (!grad_output._nnz() && self._nnz() > 0) {
      return at::sparse::zeros_like_with_indices(self);
    } else {
      return grad_output;
    }
  }();
  // 获取输入张量的值，如果未合并则进行合并
  const auto self_v = [&self]() {
    if (self.is_coalesced()) {
      return self.values();
    } else {
      return self.coalesce().values();
    }
  }();
  // 应用 coalesced_unary_ufunc 函数，并返回结果
  return coalesced_unary_ufunc(grad, [&](const Tensor& t) {
    return at::threshold_backward(t, self_v, threshold);
  });
}

// 返回阈值后向传播的稀疏版本，输出到预分配的输出张量中
Tensor& threshold_backward_sparse_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  // 根据梯度和输入稀疏性判断是否需要返回零矩阵
  const auto grad = [&]() {
    if (!grad_output._nnz() && self._nnz() > 0) {
      return at::sparse::zeros_like_with_indices(self);
    } else {
      return grad_output;
    }
  }();
  // 获取输入张量的值，如果未合并则进行合并
  auto self_v = [&self]() {
    if (self.is_coalesced()) {
      return self.values();
    } else {
      return self.coalesce().values();
    }
  }();
  // 应用 coalesced_unary_ufunc_out 函数，并返回结果
  return coalesced_unary_ufunc_out(
      grad, grad_input, [&](const Tensor& t, Tensor& out) {
        return at::threshold_backward_outf(t, self_v, threshold, out);
      });
}

// 返回将 NaN 替换为指定数值的稀疏版本张量
Tensor nan_to_num_sparse(
    const Tensor &self, std::optional<double> nan,
    std::optional<double> posinf, std::optional<double> neginf) {
  // 应用 coalesced_unary_ufunc 函数，将 NaN 替换为指定数值
  return coalesced_unary_ufunc(
      self, [&](const Tensor &t) {
        return at::nan_to_num(t, nan, posinf, neginf);
      });
}


这些注释详细解释了每行代码的作用和功能，符合要求的格式和内容。
    std::optional<double> posinf, std::optional<double> neginf,
    Tensor &out) {

# 定义函数 `nan_to_num_with_inf`，接受四个参数：`posinf`、`neginf` 作为可选的正负无穷值，以及 `out` 作为输出张量引用。

  return coalesced_unary_ufunc_out(
      self, out, [&](const Tensor &t, Tensor &out) {

  调用 `coalesced_unary_ufunc_out` 函数，将 `self` 和 `out` 作为参数传入，并使用 Lambda 表达式定义一个匿名函数，接受输入张量 `t` 和输出张量 `out`。

        return at::nan_to_num_outf(t, nan, posinf, neginf, out);
      });

        在匿名函数中调用 `at::nan_to_num_outf` 函数，传递输入张量 `t`、`nan`、`posinf`、`neginf` 和输出张量 `out`，执行 NaN 替换为数值操作，并将结果写入 `out` 张量。
}  // 结束 nan_to_num_sparse_ 函数的定义

Tensor& nan_to_num_sparse_(
    Tensor &self, std::optional<double> nan,
    std::optional<double> posinf, std::optional<double> neginf) {
  // 检查输入张量是否已经合并，如果没有，抛出错误信息
  TORCH_CHECK(self.is_coalesced(), "nan_to_num_ requires coalesced input");
  // 调用 nan_to_num_sparse_out 函数处理稀疏张量的 NaN、正无穷和负无穷值替换，并返回处理后的张量
  return nan_to_num_sparse_out(self, nan, posinf, neginf, self);
}

}  // 结束 namespace at::native
```