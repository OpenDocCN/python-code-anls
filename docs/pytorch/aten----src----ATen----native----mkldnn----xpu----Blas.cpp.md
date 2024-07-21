# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\Blas.cpp`

```
// 引入 ATen 库中的头文件，用于张量操作和计算
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

// 命名空间声明，定义了 ATen 库中 XPU 相关的功能
namespace at::native::xpu {

// 计算 result = beta * self + alpha * (mat1 * mat2)，并存储在 result 引用中
Tensor& addmm_out(
    const Tensor& self,       // 输入张量 self
    const Tensor& mat1,       // 输入张量 mat1
    const Tensor& mat2,       // 输入张量 mat2
    const Scalar& beta,       // 标量 beta
    const Scalar& alpha,      // 标量 alpha
    at::Tensor& result) {     // 输出张量 result

  // 检查张量的后端是否为 XPU
  checkBackend("addmm_out", {result, self, mat1, mat2}, Backend::XPU);

  // 检查 mat1 是否为二维矩阵
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");

  // 检查 mat2 是否为二维矩阵
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");

  // 检查 mat1 和 mat2 是否可以相乘
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  // 设定 result 的形状为 mat1 的行数和 mat2 的列数
  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};
  result.resize_(result_shape);

  // 获取 result 的尺寸
  IntArrayRef result_sizes = result.sizes();

  // 如果 result 的任何一个维度为 0，则直接返回 result
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  // 如果 mat1 的元素个数为 0
  if (mat1.numel() == 0){
    // 如果 beta 转为 float 类型为 0，则将 result 全部置为 0
    if(beta.to<float>() == 0.f){
      return result.zero_();
    }
    // 否则，使用 self 扩展为 result 尺寸，乘以 beta，并存储在 result 中
    return at::mul_out(
      result,
      self.expand(result.sizes()),
      at::native::scalar_tensor(
        beta,
        self.scalar_type(),
        c10::nullopt,
        at::kCPU,
        c10::nullopt
      )
    );
  }

  // 检查 self 是否可以扩展到 result_shape
  TORCH_CHECK(
      are_expandable(self.sizes(), result_shape),
      "addmm_out input must be expanable to:",
      result_shape,
      " but got:",
      self.sizes());

  // 如果 mat1 是复数或双精度类型，则报错，因为 oneDNN 不支持这些类型的矩阵乘法
  if (mat1.is_complex() || mat1.scalar_type() == ScalarType::Double) {
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }

  // 初始化偏置张量 bias
  Tensor bias = Tensor();

  // 初始化 oneDNN 属性 attr
  onednn::Attr attr;

  // 将 beta 转为 float 类型
  float beta_ = beta.to<float>();

  // 如果 beta_ 等于 0
  if (beta_ == 0.f) {
    // 如果 alpha 不等于 1，则在后处理中添加 alpha 乘法操作
    if (alpha.to<float>() != 1.f) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
    }
  } else {
    // 如果 alpha 等于 1 并且 beta_ 等于 1，则将 self 赋给 bias
    if (alpha.to<float>() == 1.f && beta_ == 1.f) {
      bias = self;
    } else {
      // 如果 self 的维度为 1，则在第 0 维度上添加一个维度，构成 binary
      Tensor binary = self.dim() == 1 ? self.unsqueeze(0) : self;
      // 计算 alpha_ = alpha / beta_
      float alpha_ = alpha.to<float>() / beta_;
      // 如果 alpha_ 不等于 1，则在后处理中添加 alpha_ 乘法操作
      if (alpha_ != 1.f)
        attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      // 添加 binary 到后处理中的二进制加法
      attr.append_post_binary(attr.kind_with_binary_add, binary);
      // 如果 beta_ 不等于 1，则在后处理中添加 beta_ 乘法操作
      if (beta_ != 1.f)
        attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }

  // 调用 oneDNN 的矩阵乘法函数 matmul
  onednn::matmul(result, mat1, mat2, bias, true, attr);

  // 返回结果张量 result
  return result;
}

// 下划线方法，计算带有激活函数的 addmm，但未提供完整实现代码
Tensor& _addmm_activation_out(
    const Tensor& self,       // 输入张量 self
    const Tensor& mat1,
    # 使用 addmm_out 函数计算矩阵乘法 mat1 * mat2 + beta * self，结果存储到 result 中
    addmm_out(self, mat1, mat2, beta, alpha, result);
    
    # 根据 use_gelu 参数选择激活函数，如果为 True，则使用 GeLU 激活函数
    if (use_gelu):
        # 在 result 上应用原地 GeLU 激活函数
        at::gelu_(result);
    else:
        # 在 result 上应用原地 ReLU 激活函数
        at::relu_(result);
    
    # 返回计算后的结果张量 result
    return result;
}

// 在给定的张量 self 和 mat2 上执行矩阵乘法，结果存储在 result 中
Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& result) {
  // 检查张量的后端是否为 XPU，确保张量都是在同一后端上
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  // 检查 self 张量是否为二维矩阵
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  // 检查 mat2 张量是否为二维矩阵
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  // 检查矩阵尺寸是否允许矩阵乘法操作
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0],
      "x",
      self.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  // 调整 result 的尺寸以匹配矩阵乘法的输出形状
  result.resize_({self.size(0), mat2.size(1)});
  // 若任意输入张量为空，则直接返回全零的 result
  if (self.numel() == 0 || mat2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  // 如果 self 张量是复数类型或者是双精度浮点数，则抛出错误
  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }

  // 调用 oneDNN 库执行矩阵乘法，并将结果存储在 result 中
  onednn::matmul(result, self, mat2, Tensor(), true, onednn::Attr());
  return result;
}

// 对两个输入张量进行矩阵乘法操作，并返回结果张量
Tensor mm(const Tensor& self, const Tensor& mat2) {
  // 创建一个空张量 result，用于存储矩阵乘法的结果
  auto result = at::empty({0}, self.options());
  // 调用 mm_out 函数执行矩阵乘法，结果存储在 result 中
  xpu::mm_out(self, mat2, result);
  return result;
}

// 对给定的张量 self 和向量 vec 执行矩阵-向量乘法操作
Tensor mv(const Tensor& self, const Tensor& vec) {
  // 创建一个空张量 result，用于存储矩阵-向量乘法的结果
  Tensor result = at::empty({self.size(0)}, self.options());
  // 调用 addmv_ 函数执行矩阵-向量乘法，结果存储在 result 中
  return at::addmv_(result, self, vec, 0, 1);
}


// result = beta * input + alpha * (batch1 @ batch2)
// 在给定的输入 input、batch1 和 batch2 上执行批量矩阵乘法，结果存储在 result 中
Tensor& baddbmm_out(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // 检查张量的后端是否为 XPU，确保张量都是在同一后端上
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  // 检查 batch1 张量是否为三维张量
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  // 检查 batch2 张量是否为三维张量
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  // 根据 batch1 和 batch2 的尺寸确定 result 的形状，并调整 result 的尺寸
  std::vector<int64_t> result_shape = {
      batch1.size(0), batch1.size(1), batch2.size(2)};
  result.resize_(result_shape);
  
  // 若 result 张量为空，则直接返回空的 result
  if (result.numel() == 0){
    return result;
  } else if (batch1.size(2) == 0){
    // 如果 batch1 的第三维度为零，则根据 beta 的值判断如何处理 result
    if (beta.to<c10::complex<double>>() == 0.0){
      return result.zero_();
    }else{
      // 否则，使用 alpha 和 input 来更新 result 的值
      at::mul_out(result, input, beta);
      return result;
    }
  }

  // 检查输入 input 的形状是否可以扩展到 result_shape
  TORCH_CHECK(
      are_expandable(input.sizes(), result_shape),
      "baddbmm_out input must be expanable to:",
      result_shape,
      " but got:",
      input.sizes());

  // 如果 batch1 张量是复数类型或者 batch2 张量是双精度浮点数类型，则抛出错误
  if (batch1.is_complex() || batch2.scalar_type() == ScalarType::Double) {
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }

  // 为执行 oneDNN 操作配置属性
  onednn::Attr attr;
  float beta_ = beta.to<float>();
  Tensor binary;
  
  // 根据 beta 的值来确定执行的操作
  if (beta_ == 0.f) {
    if (alpha.to<float>() != 1.f) {
      // 如果 beta 为零且 alpha 不为一，则设置属性以执行相应的操作
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
    }
  } else {
    // 如果 beta 不为零，则根据 alpha 和 beta 的比例设置属性
    binary = input.dim() < 3 ? input.unsqueeze(0) : input;
    binary = binary.dim() < 3 ? binary.unsqueeze_(0) : binary;
    float alpha_ = alpha.to<float>() / beta_;
    if (alpha_ != 1.f)
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
    attr.append_post_binary(attr.kind_with_binary_add, binary);
    # 如果 beta_ 不等于 1.0（浮点数），则执行以下操作
    if (beta_ != 1.f)
      # 向属性列表 attr 中添加一个后续的元素运算，参数为 1.0、beta_、0.0 和 attr.kind_with_linear
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
  }
  # 使用 onednn 库的 matmul 函数进行矩阵乘法计算
  onednn::matmul(result, batch1, batch2, at::Tensor(), true, attr);
  # 返回计算结果 result
  return result;
}

// 原地操作，将 batch1 和 batch2 进行批矩阵乘法，并加到 self 中
Tensor& baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  // 检查输入张量的数据类型必须一致
  TORCH_CHECK(self.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", self.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
  // 调用 xpu 原生方法执行 baddbmm_out 操作
  return at::native::xpu::baddbmm_out(
      self, batch1, batch2, beta, alpha, self);
}

// 执行批矩阵乘法并返回结果张量
Tensor baddbmm(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  // 创建一个空张量作为结果
  Tensor r = at::empty({0}, input.options());
  // 检查输入张量的数据类型必须一致
  TORCH_CHECK(input.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", input.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
  // 调用 xpu 原生方法执行 baddbmm_out 操作
  r = at::native::xpu::baddbmm_out(input, batch1, batch2, beta, alpha, r);
  return r;
}

// 在给定的结果张量上执行批矩阵乘法，并返回结果张量的引用
Tensor& addbmm_out(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  // 检查张量的后端是否为 XPU
  checkBackend("addbmm_out", {out, self, batch1, batch2}, Backend::XPU);
  // 检查 batch1 和 batch2 是否为 3D 张量
  TORCH_CHECK(
      batch1.dim() == 3 && batch2.dim() == 3,
      "Batch tensors should be 3D, got dimensions ",
      batch1.dim(),
      " and ",
      batch2.dim());

  // 调整结果张量的大小为 batch1 的第二维和 batch2 的第三维
  out.resize_({batch1.size(1), batch2.size(2)});
  // 如果 alpha 等于 0 或者 batch1 或 batch2 为空，则直接返回结果张量
  if (alpha.to<float>() == 0.f || batch1.numel() == 0 || batch2.numel() == 0) {
    // 重新调整结果张量的大小为 batch1 的第二维和 batch2 的第三维
    out.resize_({batch1.size(1), batch2.size(2)});
    if (out.numel() == 0)
      return out;

    // 如果 self 已定义且 beta 不等于 0，则将结果张量与 self 相乘
    if (self.defined() && beta.to<float>() != 0.f) {
      out = at::mul_out(
          out, self, at::native::wrapped_scalar_tensor(at::Scalar(beta)));
    } else {
      // 否则，将结果张量置零
      out.zero_();
    }
    return out;
  }

  // 根据 batch1 的大小选择不同的操作
  Tensor b1;
  if (batch1.size(0) > 1) {
    // 如果 batch1 的第一维大于 1，则先转置，再展平为二维张量
    b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
  } else {
    // 否则，直接展平为二维张量
    b1 = batch1.contiguous().view({batch1.size(1), -1});
  }
  // 展平 batch2 为二维张量
  auto b2 = batch2.contiguous().view({-1, batch2.size(2)});
  // 调用 xpu 原生方法执行 addmm_out 操作
  at::native::xpu::addmm_out(self, b1, b2, beta, alpha, out);

  return out;
}

// 原地操作，将 batch1 和 batch2 进行批矩阵乘法，并加到 self 中
Tensor& addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  // 调用 xpu 原生方法执行 addbmm_out 操作
  at::native::xpu::addbmm_out(self, batch1, batch2, beta, alpha, self);
  return self;
}

// 执行批矩阵乘法并返回结果张量
Tensor addbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  // 创建一个空张量作为结果
  Tensor out = at::empty({0}, self.options());
  // 调用 xpu 原生方法执行 addbmm_out 操作
  at::native::xpu::addbmm_out(self, batch1, batch2, beta, alpha, out);
  return out;
}

// 在给定的结果张量上执行批矩阵乘法，并返回结果张量的引用
Tensor& bmm_out(const Tensor& self, const Tensor& batch2, Tensor& result) {
  // 检查张量的后端是否为 XPU
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  // 检查 self 和 batch2 是否为 3D 张量
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  // 调整结果张量的大小为 self 的第一维、self 的第二维和 batch2 的第三维
  result.resize_({self.size(0), self.size(1), batch2.size(2)});
  // 如果 self 或 batch2 为空，则将结果张量置零
  if (self.numel() == 0 || batch2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    // 返回结果张量的引用
    return result;

    // 返回结果张量的引用
    return result;
  }
  //```cpp
    // 如果不满足上述条件，则调用 xpu 原生方法执行 bmm_out 操作
    at::native::xpu::bmm_out(self, batch2, result);
  // 返回结果张量的引用
  return result;
}

// 以下是未完整的代码，无法添加注释。
    return result;
  }



    // 返回 result 变量，结束当前函数的执行并返回结果
    return result;
  }



  if (self.is_complex() || self.scalar_type() == ScalarType::Double) {
    // 检查 self 张量是否为复数类型或者数据类型为双精度浮点型
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }



    // 如果 self 张量是复数类型或者数据类型为双精度浮点型，则抛出错误信息
    AT_ERROR(
        "Double and complex datatype matmul is not supported in oneDNN");
  }



  onednn::matmul(result, self, batch2, at::Tensor(), true, onednn::Attr());



    // 调用 oneDNN 库的 matmul 函数执行矩阵乘法运算，将结果存储在 result 中
    onednn::matmul(result, self, batch2, at::Tensor(), true, onednn::Attr());



  return result;



    // 返回存储矩阵乘法运算结果的 result 变量
    return result;
  }
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  // 创建一个空的张量 result，使用 self 张量的选项（数据类型和设备）
  auto result = at::empty({0}, self.options());
  // 调用本地的 xpu::bmm_out 函数，将 self 和 batch2 张量的批量矩阵乘积计算结果存入 result
  at::native::xpu::bmm_out(self, batch2, result);
  // 返回计算结果的张量 result
  return result;
}

Tensor& addmv_out(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  // 定义一个用于临时存储 self 张量的变量 self_v
  Tensor self_v;
  // 检查矩阵 mat、向量 vec 和 self 张量的维度满足条件
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());
  if (self.dim() == 1 && self.size(0) != 1) {
    // 如果 self 是一维且大小不为 1，则将其视图调整为二维（self.size(0) x 1）
    TORCH_CHECK(
        (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
        "size mismatch, get ",
        self.size(0),
        ", ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self.view({self.size(0), 1});
  } else {
    // 否则直接使用 self
    TORCH_CHECK(
        (mat.size(1) == vec.size(0)),
        "size mismatch, get ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
    self_v = self;
  }

  // 将 vec 张量视图调整为二维（vec.size(0) x 1）
  Tensor vec_v = vec.view({vec.size(0), 1});
  // 调用本地的 addmm_out 函数，将 self_v 、mat 和 vec_v 的矩阵乘积结果加权并存入 out
  at::native::xpu::addmm_out(self_v, mat, vec_v, beta, alpha, out);
  // 调整 out 张量的大小为 mat 的第一维大小
  out.resize_({mat.size(0)});
  // 返回计算结果的张量 out
  return out;
}

Tensor& tensordot_out(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2,
    Tensor& result) {
  // 调用 at::tensordot 函数计算 input1 和 input2 张量在指定维度上的张量点积，存入 result_tmp
  Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
  // 获取 result_tmp 张量的数据类型
  auto result_dtype = result_tmp.scalar_type();
  // 获取 result 张量的数据类型
  auto output_tensor_dtype = result.scalar_type();
  // 获取 result 张量的设备
  auto output_device = result.device();
  // 获取 input1 张量的设备
  auto input1_device = input1.device();
  // 获取 input2 张量的设备
  auto input2_device = input2.device();
  // 检查输入和输出张量是否在同一个设备上
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "tensordot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device,
      ", input tensor a on ",
      input1_device,
      ", and input tensor b on ",
      input2_device);
  // 检查计算结果的数据类型是否与输出张量的数据类型相同（因为 tensordot 不支持类型提升）
  TORCH_CHECK(
      result_dtype == output_tensor_dtype,
      "tensordot",
      ": Expected the output tensor to have dtype ",
      result_dtype,
      ", but got an output tensor with dtype ",
      output_tensor_dtype);
  // 调整输出张量 result 的大小以匹配 result_tmp
  at::native::resize_output(result, result_tmp.sizes());
  // 将 result_tmp 的内容复制到输出张量 result
  result.copy_(result_tmp);
  // 返回计算结果的张量 result
  return result;
}
// 注册 ATen 库的实现函数到 XPU 设备上
TORCH_LIBRARY_IMPL(aten, XPU, m){
  // 注册 addmm 函数的实现函数 addmm_out
  m.impl("addmm.out", TORCH_FN(addmm_out));
  // 注册 _addmm_activation 函数的实现函数 _addmm_activation_out
  m.impl("_addmm_activation.out", TORCH_FN(_addmm_activation_out));
  // 注册 mm 函数的实现函数 mm_out
  m.impl("mm.out", TORCH_FN(mm_out));
  // 注册 mm 函数的实现函数 mm
  m.impl("mm", TORCH_FN(mm));
  // 注册 baddbmm 函数的实现函数 baddbmm_out
  m.impl("baddbmm.out", TORCH_FN(baddbmm_out));
  // 注册 baddbmm_ 函数的实现函数 baddbmm_
  m.impl("baddbmm_", TORCH_FN(baddbmm_));
  // 注册 baddbmm 函数的实现函数 baddbmm
  m.impl("baddbmm", TORCH_FN(baddbmm));
  // 注册 addbmm 函数的实现函数 addbmm_out
  m.impl("addbmm.out", TORCH_FN(addbmm_out));
  // 注册 addbmm_ 函数的实现函数 addbmm_
  m.impl("addbmm_", TORCH_FN(addbmm_));
  // 注册 addbmm 函数的实现函数 addbmm
  m.impl("addbmm", TORCH_FN(addbmm));
  // 注册 bmm 函数的实现函数 bmm_out
  m.impl("bmm.out", TORCH_FN(bmm_out));
  // 注册 bmm 函数的实现函数 bmm
  m.impl("bmm", TORCH_FN(bmm));
  // 注册 addmv 函数的实现函数 addmv_out
  m.impl("addmv.out", TORCH_FN(addmv_out));
  // 注册 tensordot 函数的实现函数 tensordot_out
  m.impl("tensordot.out", TORCH_FN(tensordot_out));
}

} // namespace at::native::xpu
```