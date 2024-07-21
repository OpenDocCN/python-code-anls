# `.\pytorch\aten\src\ATen\native\quantized\cpu\qrelu.cpp`

```
// 定义宏，用于仅支持方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <c10/util/irange.h>
// 包含线程池相关的头文件
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含一般函数和原生函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定操作符的头文件
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/relu_native.h>
#endif

#include <algorithm>

// 命名空间开始：at::native
namespace at {
namespace native {

// 定义 qrelu_stub、qrelu_leaky_stub 和 qprelu_stub 的调度分发
DEFINE_DISPATCH(qrelu_stub);
DEFINE_DISPATCH(qrelu_leaky_stub);
DEFINE_DISPATCH(qprelu_stub);

// 如果使用 PyTorch QNNPACK
#ifdef USE_PYTORCH_QNNPACK
// 定义 qnnpack_relu 函数，接收一个 Tensor 类型的输入并返回一个 Tensor
static Tensor qnnpack_relu(Tensor input) {
  Tensor qy;
  // 断言输入张量的维度大于 0
  TORCH_CHECK(
      input.ndimension() > 0, "qnnpack_relu(): Got empty input tensor");
  // 断言输入张量的数据类型为 c10::kQUInt8
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_relu(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));

  // 将输入张量进行内存格式连续化处理
  Tensor input_contig = input.contiguous(input.suggest_memory_format());

  // 获取输入张量的量化零点
  const auto zero_point = input_contig.q_zero_point();

  // 初始化 QNNPACK
  initQNNPACK();

  // 计算输入张量的元素总数
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {


继续完成余下部分的注释。这样可以帮助确保代码在阅读和理解时更加清晰和有条理。
    `
        # 更新 num_elems 变量的值，乘以输入张量的第 i 维的大小
        num_elems *= input_contig.size(i);
      }
    
      # 定义一个 QNNPACK 操作指针，初始值为 nullptr
      pytorch_qnnp_operator_t qnnpack_operator{nullptr};
    
      # 尝试创建一个 QNNPACK clamp 操作，返回状态保存在 createStatus 中
      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
          num_elems /* channels */,
          zero_point /* output min */,
          std::numeric_limits<uint8_t>::max() /* output max */,
          0 /* flags */,
          &qnnpack_operator);
    
      # 使用智能指针管理 QNNPACK 操作对象，自动释放资源
      std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(qnnpack_operator);
    
      # 检查创建操作是否成功，如果失败则抛出异常
      TORCH_INTERNAL_ASSERT(
          createStatus == pytorch_qnnp_status_success,
          "failed to create QNNPACK Relu operator");
    
      # 创建一个新的量化张量，尺寸与输入张量相同，数据类型和量化参数也与输入相同
      qy = at::_empty_affine_quantized(
          input_contig.sizes(),
          at::device(kCPU).dtype(input.scalar_type()),
          input_contig.q_scale(),
          input_contig.q_zero_point(),
          input.suggest_memory_format());
    
      # 尝试设置 QNNPACK clamp 操作，返回状态保存在 setupStatus 中
      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
          qnnpack_operator, /* clamp */
          input_contig.size(0) /* batch size */,
          (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
          num_elems /* input stride */,
          (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
          num_elems /* output stride */);
      
      # 检查设置操作是否成功，如果失败则抛出异常
      TORCH_INTERNAL_ASSERT(
          setupStatus == pytorch_qnnp_status_success,
          "failed to setup QNNPACK Relu operator");
    
      # 获取线程池，供 QNNPACK 操作使用
      pthreadpool_t threadpool = caffe2::pthreadpool_();
    
      # 尝试运行 QNNPACK 操作，返回状态保存在 runStatus 中
      const pytorch_qnnp_status runStatus =
          pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
    
      # 检查操作是否成功执行，如果失败则抛出异常
      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Relu operator");
      
      # 返回处理后的量化张量
      return qy;
}
#endif

// 定义一个函数 relu_quantized_cpu，接收一个常量引用 qx，并返回一个 Tensor
Tensor relu_quantized_cpu(const Tensor& qx) {
  // 如果定义了 USE_PYTORCH_QNNPACK 宏，并且全局上下文的量化引擎为 QNNPACK，并且 qx 的标量类型为 kQUInt8
  #ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8) {
    // 调用 qnnpack_relu 函数处理 qx，并返回处理后的 Tensor
    return qnnpack_relu(qx);
  }
  #endif
  // 声明一个 Tensor qy
  Tensor qy;
  // 调用 qrelu_stub 函数处理 qx，结果存储在 qy 中
  qrelu_stub(qx.device().type(), qx, qy);
  // 返回结果 Tensor qy
  return qy;
}

// 定义一个函数 relu_quantized_cpu_，接收一个 Tensor 引用 qx，并返回一个 Tensor 引用
Tensor& relu_quantized_cpu_(Tensor& qx) {
  // 获取 qx 的量化零点值
  const auto zero_point = qx.q_zero_point();
  // 根据 qx 的标量类型进行分发处理
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    // 定义使用的向量类型 Vec
    using Vec = Vectorized<scalar_t>;
    // 创建一个 Tensor 迭代器，处理 qx
    auto iter = TensorIterator::unary_op(qx, qx);
    // 创建一个零点值的向量
    auto zero_point_vec = Vec(scalar_t(zero_point));
    // 执行 CPU 内核向量化处理
    cpu_kernel_vec(
        iter,
        // 对每个元素执行操作，保证不小于零点值
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        // 对向量执行 ReLU 操作，保证不小于零点值
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
  // 返回处理后的 Tensor 引用 qx
  return qx;
}

// 定义一个函数 leaky_relu_out_quantized_cpu，接收一个常量 Tensor self、标量 negval 和一个 Tensor 引用 result，返回一个 Tensor 引用
Tensor& leaky_relu_out_quantized_cpu(const Tensor& self,
                                 const Scalar& negval, Tensor& result) {
  // 调用 qrelu_leaky_stub 函数处理 self，并将结果存储在 result 中
  qrelu_leaky_stub(self.device().type(), result, self, negval);
  // 返回结果 Tensor 引用 result
  return result;
}

// 定义一个函数 leaky_relu_quantized_cpu，接收一个常量 Tensor self 和一个标量 negval，返回一个 Tensor
Tensor leaky_relu_quantized_cpu(const Tensor& self, const Scalar& negval) {
  // 使 self 连续化，并建议适当的内存格式
  const auto qx = self.contiguous(self.suggest_memory_format());
  // 根据 qx 的属性创建一个新的量化 Tensor qy
  auto qy = at::_empty_affine_quantized(qx.sizes(),
      at::device(kCPU).dtype(self.scalar_type()),
      qx.q_scale(),
      qx.q_zero_point(),
      self.suggest_memory_format());
  // 调用 qrelu_leaky_stub 函数处理 self，并将结果存储在 qy 中
  qrelu_leaky_stub(self.device().type(), qy, qx, negval);
  // 返回结果 Tensor qy
  return qy;
}

// 定义一个函数 leaky_relu_quantized_cpu_，接收一个 Tensor 引用 self 和一个标量 negval，返回一个 Tensor 引用
Tensor& leaky_relu_quantized_cpu_(Tensor& self, const Scalar& negval) {
  // 调用 qrelu_leaky_stub 函数处理 self，并将结果存储在 self 中
  qrelu_leaky_stub(self.device().type(), self, self, negval);
  // 返回处理后的 Tensor 引用 self
  return self;
}

// 定义一个静态函数 _prelu_kernel_quantized_cpu_impl，接收两个常量 Tensor self 和 weight，以及两个数值类型参数 output_scale 和 output_zero_point，返回一个 Tensor
static Tensor _prelu_kernel_quantized_cpu_impl(const Tensor& self, const Tensor& weight,
                                double output_scale, int64_t output_zero_point) {
  // 获取 self 的维度数
  auto ndim = self.dim();
  // 如果 ndim 小于 1 或大于 5，则执行参考路径
  if (ndim > 5 || ndim < 1) {
    // 对 self 进行去量化操作，得到 x
    auto x = self.dequantize();
    // 对 x 和 weight 执行 prelu 操作，得到 y
    auto y = at::prelu(x, weight);
    // 对 y 进行按张量量化操作，使用 output_scale 和 output_zero_point，类型为 c10::kQUInt8
    return at::quantize_per_tensor(y, output_scale, output_zero_point, c10::kQUInt8);
  }

  // 根据 self 的属性创建一个新的量化 Tensor qy
  auto qy = at::_empty_affine_quantized(self.sizes(),
      at::device(kCPU)
        .dtype(self.scalar_type()),
      output_scale,
      output_zero_point,
      self.suggest_memory_format());

  // 调用 qprelu_stub 函数处理 self 和 weight，并将结果存储在 qy 中
  qprelu_stub(self.device().type(), qy, self, weight);

  // 返回结果 Tensor qy
  return qy;
}

// 定义一个函数 _prelu_kernel_quantized_cpu，接收两个常量 Tensor self 和 weight，返回一个 Tensor
Tensor _prelu_kernel_quantized_cpu(const Tensor& self, const Tensor& weight) {
  // 调用 _prelu_kernel_quantized_cpu_impl 函数处理 self 和 weight，同时使用 self 的量化比例和零点值
  return _prelu_kernel_quantized_cpu_impl(self, weight, self.q_scale(), self.q_zero_point());
}

// 匿名命名空间
namespace {
// 定义一个函数 quantized_relu6，接收一个常量 Tensor qx，返回一个 Tensor
Tensor quantized_relu6(const Tensor& qx) {
  // 声明一个 Tensor qy
  Tensor qy;
  // 调用 hardtanh_quantized_cpu 函数处理 qx，设定下界 0.0f，上界 6.0f，并将结果存储在 qy 中
  qy = hardtanh_quantized_cpu(qx, 0.0f, 6.0f);
  // 返回结果 Tensor qy
  return qy;
}

// 定义一个函数 quantized_relu6_，接收一个 Tensor 引用 qx，返回一个 Tensor 引用
Tensor quantized_relu6_(Tensor& qx) {
  // 调用 hardtanh_quantized_cpu_ 函数处理 qx，设定下界 0.0f，上界 6.0f
  hardtanh_quantized_cpu_(qx, 0.0f, 6.0f);
  // 返回处理后的 Tensor 引用 qx
  return qx;
}

// 定义一个类 QRelu6
class QRelu6 final {
 public:
  // 定义一个静态方法 run，接收一个 Tensor qx 和一个布尔值 inplace，返回一个 Tensor
  static Tensor run(Tensor qx, bool inplace) {
    // 如果 inplace 为 true，则调用 quantized_relu6_ 处理 qx，并返回处理结果
    if (inplace) {
      return quantized_relu6_(qx);
    } else {
      // 否则调用 quantized_relu6 处理 qx，并返回处理结果
      return quantized_relu6(qx);
    }
  }
};
# 定义一个名为 QLeakyRelu 的最终类
class QLeakyRelu final {
 public:
  // 静态方法，执行量化 Leaky ReLU 操作
  static Tensor run(Tensor self, const Scalar& negative_slope, bool inplace, double output_scale, int64_t output_zero_point) {
    // 如果 inplace 参数为 true，则忽略，TODO: 支持 inplace 操作
    if (inplace) {
      // 输出警告信息，因为 quantized::leaky_relu 尚不支持 inplace 操作
      TORCH_WARN("inplace=True is not supported for quantized::leaky_relu yet");
    }
    // 将输入张量 qx 进行内存连续性操作并采用推荐的内存格式
    const auto qx = self.contiguous(self.suggest_memory_format());
    // 创建一个与 qx 相同大小的量化张量 qy，使用指定的输出量化比例和零点，存储在 CPU 上
    auto qy = at::_empty_affine_quantized(qx.sizes(),
      at::device(kCPU).dtype(self.scalar_type()),
      output_scale,
      output_zero_point,
      self.suggest_memory_format());
    // 调用量化的 Leaky ReLU 内核函数 qrelu_leaky_stub
    qrelu_leaky_stub(self.device().type(), qy, qx, negative_slope);
    // 返回量化张量 qy
    return qy;
  }
};

# 定义一个名为 QPRelu 的最终类
class QPRelu final {
 public:
  // 静态方法，执行量化 PReLU 操作
  static Tensor run(Tensor self, const Tensor& weight, double output_scale, int64_t output_zero_point) {
    // 调用内部的量化 PReLU CPU 实现函数 _prelu_kernel_quantized_cpu_impl
    return _prelu_kernel_quantized_cpu_impl(self, weight, output_scale, output_zero_point);
  }
};

# 注册量化操作的实现，包括 quantized::relu6、quantized::leaky_relu 和 quantized::prelu
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::relu6"), TORCH_FN(QRelu6::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::leaky_relu"), TORCH_FN(QLeakyRelu::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::prelu"), TORCH_FN(QPRelu::run));
}

# 结束命名空间定义，at::native
} // namespace at::native

# 结束命名空间定义，全局命名空间
}}  // namespace
```