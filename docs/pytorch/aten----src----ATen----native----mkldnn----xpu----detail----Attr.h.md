# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Attr.h`

```py
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件，提供张量操作支持

#include <oneapi/dnnl/dnnl.hpp>
// 包含 oneDNN 库的头文件，提供深度学习网络的加速计算功能

#include <oneapi/dnnl/dnnl_types.h>
// 包含 oneDNN 库的类型定义头文件，定义了与计算引擎相关的类型和常量

#include <ATen/native/mkldnn/xpu/detail/Utils.h>
// 包含 ATen 库的 mkldnn 描述处理工具的头文件

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
// 包含 ATen 库的 mkldnn 描述处理上下文的头文件

namespace at::native::onednn {
// 命名空间定义：ATen 库的本地命名空间和 oneDNN 加速器命名空间

/* oneDNN quantization usage:
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_quantization.html#

   src_fp32 = scale_src * (src_int8 - zero_point)
   wei_fp32 = scale_wei * (wei_int8 - zero_point)
   dst_fp32 = scale_dst * (dst_int8 - zero_point)
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   Int8 Convolution: dst_int8 = 1 / scale_dst * dst_fp32;

   Considering zero-point (asymmetric):
   dst_fp32 = (src_int8 - src_zp) * src_sc * wei_int8 * wei_sc
   dst_sc * (dst_int8 - dst_zp) = (src_int8 - src_zp) * wei_int8  * src_sc *
                                 wei_sc
   dst_int8 = (src_int8 - src_zp) * wei_int8 * src_sc * wei_sc / dst_sc +
              dst_zp

   considering bias:
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32 + bias
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   + bias Int8 Convolution: dst_fp32 = (src_int8 * wei_int8 + bias/(scale_src *
   scale_wei)) * (scale_src * scale_wei) Int8 Convolution: dst_int8 = 1 /
   scale_dst * dst_fp32;
*/
// oneDNN 量化使用的注释，详细描述了量化计算过程和考虑的因素

/*
   oneDNN postops usage:
   Currently, oneDNN supports 5 kinds of post ops. More details can be refered
to oneDNN doc.
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_post_ops.html#doxid-dev-guide-attributes-post-ops-1dev-guide-attributes-post-ops-eltwise

0. without post ops
   dst = Conv(src, wei) + bias;
   dst_int8 = 1/q_scale * dst; q_scale is the op output quantization scale
   fp32 API: Attr attr;
   int8 API: Attr attr(q_scale);

1. append eltwise post op
   dst = elt_scale * Eltwise{conv_scale * [Conv(src, wei) + bias], alpha, beta}
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)

2. append sum post op
   dst = conv_scale * Conv(src, wei) + sum_scale * (dst - zp)
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)

3. append binary post op
   dst = Binary[Conv(src, wei)]

*/
// oneDNN 后处理操作的使用注释，介绍了支持的不同后处理操作类型及其应用场景

using kind_t = dnnl::primitive::kind;
// 使用 dnnl 命名空间的 primitive 类型别名定义
// 定义了一个名为 PostOpParam 的结构体，用于表示不同类型的后操作参数

struct PostOpParam {
  // eltwise 后操作的构造函数，初始化 scale、alpha、beta、算法类型和类型
  PostOpParam(float scale, float alpha, float beta, dnnl::algorithm algo, kind_t kind)
      : scale_(scale), alpha_(alpha), beta_(beta), algo_(algo), kind_(kind) {}

  // sum 后操作的构造函数，初始化 scale 和类型
  PostOpParam(float scale, kind_t kind) : scale_(scale), kind_(kind) {}

  // binary 后操作的构造函数，初始化 binary 引用、内存描述、预期内存描述、算法类型和类型
  PostOpParam(
      at::Tensor& binary,
      dnnl::memory::desc& binary_md,
      dnnl::memory::desc& expected_md,
      dnnl::algorithm algo,
      kind_t kind)
      : binary_(binary),
        meta_(binary_md),
        expected_meta_(expected_md),
        algo_(algo),
        kind_(kind) {}

  // prelu 后操作的构造函数，初始化 mask 和类型
  PostOpParam(int mask, kind_t kind) : mask_(mask), kind_(kind) {}

  // post sum 或 binary 后操作的构造函数，初始化 binary 引用、scale、算法类型和类型
  PostOpParam(at::Tensor& binary, float scale, dnnl::algorithm algo, kind_t kind)
      : scale_(scale), binary_(binary), algo_(algo), kind_(kind) {}

  // int8 sum/eltwise 的 scale 参数，默认为 1.0
  float scale_ = 1.0;
  // eltwise 的 alpha 和 beta 参数，默认为 0.0
  float alpha_ = 0.0;
  float beta_ = 0.0;
  // binary 操作的 tensor 和预期 tensor，默认为空 tensor
  at::Tensor binary_ = at::Tensor();
  at::Tensor expected_binary_ = at::Tensor();
  // binary 操作的指针，默认为空指针
  void* binary_ptr_ = nullptr;
  // 用于描述内存的 meta 数据，默认为空描述
  dnnl::memory::desc meta_ = dnnl::memory::desc();
  dnnl::memory::desc expected_meta_ = dnnl::memory::desc();
  // prelu 操作的掩码参数，默认为 0
  int mask_ = 0;
  // 共享的算法参数，默认为 eltwise_relu
  dnnl::algorithm algo_ = dnnl::algorithm::eltwise_relu;
  // 类型参数，默认为 eltwise
  kind_t kind_ = kind_t::eltwise;
};
class Attr {
 public:
  // 默认构造函数，初始化量化比例为1.0，量化零点为0
  Attr() : q_scale_(1.f), q_zero_point_(0) {}
  
  // 带参数的构造函数，初始化量化比例和量化零点
  Attr(float q_scale, int64_t zp = 0) : q_scale_(q_scale), q_zero_point_(zp) {}

  /***** eltwise *****/
  // 使用 ReLU 的算法
  dnnl::algorithm kind_with_relu = dnnl::algorithm::eltwise_relu;
  // 使用 Sigmoid 的算法
  dnnl::algorithm kind_with_sigmoid = dnnl::algorithm::eltwise_logistic;
  // 使用 GELU (Tanh 版本) 的算法
  dnnl::algorithm kind_with_gelu_tanh = dnnl::algorithm::eltwise_gelu_tanh;
  // 使用 GELU (Erf 版本) 的算法
  dnnl::algorithm kind_with_gelu_erf = dnnl::algorithm::eltwise_gelu_erf;
  // 使用 Mish 的算法
  dnnl::algorithm kind_with_mish = dnnl::algorithm::eltwise_mish;
  // 使用 Linear 的算法
  dnnl::algorithm kind_with_linear = dnnl::algorithm::eltwise_linear;
  // 使用 Swish 的算法
  dnnl::algorithm kind_with_swish = dnnl::algorithm::eltwise_swish;
  // 使用 Sqrt 的算法
  dnnl::algorithm kind_with_sqrt = dnnl::algorithm::eltwise_sqrt;
  // 使用 Tanh 的算法
  dnnl::algorithm kind_with_tanh = dnnl::algorithm::eltwise_tanh;
  // 使用 Square 的算法
  dnnl::algorithm kind_with_square = dnnl::algorithm::eltwise_square;
  // 使用 Abs 的算法
  dnnl::algorithm kind_with_abs = dnnl::algorithm::eltwise_abs;
  // 使用 Exp 的算法
  dnnl::algorithm kind_with_exp = dnnl::algorithm::eltwise_exp;
  // 使用 Log 的算法
  dnnl::algorithm kind_with_log = dnnl::algorithm::eltwise_log;
  // 使用 Round 的算法
  dnnl::algorithm kind_with_round = dnnl::algorithm::eltwise_round;
  // 使用 Hardswish 的算法
  dnnl::algorithm kind_with_hardswish = dnnl::algorithm::eltwise_hardswish;
  // 使用 Soft ReLU 的算法
  dnnl::algorithm kind_with_soft_relu = dnnl::algorithm::eltwise_soft_relu;
  // 使用 ELU 的算法
  dnnl::algorithm kind_with_elu = dnnl::algorithm::eltwise_elu;
  // 使用 Pow 的算法
  dnnl::algorithm kind_with_pow = dnnl::algorithm::eltwise_pow;
  // 使用 Clip 的算法
  dnnl::algorithm kind_with_clip = dnnl::algorithm::eltwise_clip;
  // 注意：hardsigmoid 算法目前似乎还不被 oneDNN 支持
  dnnl::algorithm kind_with_hardsigmoid = dnnl::algorithm::eltwise_hardsigmoid;

  /***** binary *****/
  // 使用二元乘法的算法
  dnnl::algorithm kind_with_binary_mul = dnnl::algorithm::binary_mul;
  // 使用二元加法的算法
  dnnl::algorithm kind_with_binary_add = dnnl::algorithm::binary_add;
  // 使用二元减法的算法
  dnnl::algorithm kind_with_binary_sub = dnnl::algorithm::binary_sub;
  // 使用二元除法的算法
  dnnl::algorithm kind_with_binary_div = dnnl::algorithm::binary_div;
  // 使用二元等于的算法
  dnnl::algorithm kind_with_binary_eq = dnnl::algorithm::binary_eq;
  // 使用二元不等于的算法
  dnnl::algorithm kind_with_binary_ne = dnnl::algorithm::binary_ne;
  // 使用二元大于等于的算法
  dnnl::algorithm kind_with_binary_ge = dnnl::algorithm::binary_ge;
  // 使用二元大于的算法
  dnnl::algorithm kind_with_binary_gt = dnnl::algorithm::binary_gt;
  // 使用二元小于等于的算法
  dnnl::algorithm kind_with_binary_le = dnnl::algorithm::binary_le;
  // 使用二元小于的算法
  dnnl::algorithm kind_with_binary_lt = dnnl::algorithm::binary_lt;
  // 使用二元最大值的算法
  dnnl::algorithm kind_with_binary_max = dnnl::algorithm::binary_max;
  // 使用二元最小值的算法
  dnnl::algorithm kind_with_binary_min = dnnl::algorithm::binary_min;

  // 添加 sum 后操作
  Attr& append_post_sum(
      float sum_scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(
        // 添加 sum 后操作参数，包括乘积的规模和类型（这里是 sum）
        PostOpParam(/*scale_sum*/ sum_scale * sum_q_scale, kind_t::sum));
    return *this;
  }

  // 添加 eltwise 后操作
  Attr& append_post_eltwise(
      float scale,
      float alpha,
      float beta,
      dnnl::algorithm algo) {
    ops_params_.push_back(
        // 添加 eltwise 后操作参数，包括规模、alpha、beta、算法类型（这里是 eltwise）
        PostOpParam(scale, alpha, beta, algo, kind_t::eltwise));
    return *this;
  }

  // 省略了其他成员函数
  // 返回当前对象的引用，用于链式调用
  return *this;
}

// 添加二进制后操作
Attr& append_post_binary(dnnl::algorithm algo, const at::Tensor& binary) {
  // 如果输入张量是量化的，则将其反量化
  auto binary_ = binary.is_quantized() ? at::dequantize(binary) : binary;
  // 检查输入张量是否按照ChannelsLast格式存储
  bool binary_is_channels_last = (binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                                    binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);

  // 如果不是ChannelsLast格式，则将其转换为连续存储格式
  binary_ = binary_is_channels_last ? binary_ : binary_.contiguous();
  // 获取输入张量的内存描述符
  dnnl::memory::desc md = get_onednn_md(binary_);
  // 创建预期的内存描述符，任意格式
  auto expected_md = dnnl::memory::desc(
      md.get_dims(), md.get_data_type(), dnnl::memory::format_tag::any);
  // 将后操作参数添加到ops_params_列表中
  ops_params_.push_back(
      PostOpParam(binary_, md, expected_md, algo, kind_t::binary));
  // 返回当前对象的引用，用于链式调用
  return *this;
}

// 添加带有二进制加法的比例后操作
Attr& append_scale_binary(
    dnnl::algorithm algo,
    at::Tensor binary,
    float scale,
    float sum_q_scale = 1.f,
    int64_t zp = 0) {
  // 将后操作参数添加到ops_params_列表中
  ops_params_.push_back(PostOpParam(
      binary, /*scale_sum*/ scale * sum_q_scale, algo, kind_t::binary));
  // 返回当前对象的引用，用于链式调用
  return *this;
}

// 添加带有偏置的后操作（仅用于QConv）
template <int N>
Attr& append_bias(const at::Tensor& binary) {
  // 在PyTorch中，偏置的形状为[OC]，根据卷积维度进行扩展
  // Conv1d [OC, 1, 1], Conv2d [1, OC, 1, 1], Conv3d [1, OC, 1, 1, 1]
  at::Tensor binary_ = binary.contiguous();
  dnnl::memory::desc binary_md;
  // 根据N的不同值设置不同的内存描述符
  switch (N) {
    case 1:
      binary_md = dnnl::memory::desc(
          {binary.size(0), 1, 1},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::abc);
      break;
    case 2:
      binary_md = dnnl::memory::desc(
          {1, binary.size(0), 1, 1},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::abcd);
      break;
    case 3:
      binary_md = dnnl::memory::desc(
          {1, binary.size(0), 1, 1, 1},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::abcde);
      break;
    default:
      TORCH_INTERNAL_ASSERT(0,
          "XPU only supports append_bias for Conv1d, Conv2d and Conv3d.");
  }
  // 预期的内存描述符与binary_md相同
  // 将后操作参数添加到ops_params_列表中
  ops_params_.push_back(PostOpParam(
      binary_, binary_md, binary_md, kind_with_binary_add, kind_t::binary));
  // 返回当前对象的引用，用于链式调用
  return *this;
}

// 添加PReLU后操作
Attr& append_post_prelu(int mask) {
  // 将后操作参数添加到ops_params_列表中
  ops_params_.push_back(PostOpParam(mask, kind_t::prelu));
  // 返回当前对象的引用，用于链式调用
  return *this;
}

dnnl::post_ops extract_post_ops(const at::Tensor& dst){
  // 该函数用于从ops_params_中提取后操作参数，并将它们放入onednn的后操作中
    // 遍历 ops_params_ 中的操作参数列表
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      // 获取当前操作的类型
      kind_t kind = ops_params_[i].kind_;
      // 根据操作类型进行分支处理
      switch (kind) {
        case kind_t::eltwise: {
          // 如果是 eltwise 操作，获取算法、alpha 和 beta 参数
          dnnl::algorithm algo = ops_params_[i].algo_;
          float alpha = ops_params_[i].alpha_;
          float beta = ops_params_[i].beta_;
          // 将 eltwise 操作附加到 dnnl_post_ops_ 中
          dnnl_post_ops_.append_eltwise(algo, alpha, beta);
          break;
        }
        case kind_t::sum: {
          // 如果是 sum 操作，获取 scale 参数
          float scale = ops_params_[i].scale_;
          // 添加 sum 操作到 dnnl_post_ops_ 中
          // 注意：当前 GPU 不支持后续的 post-sum zp
          dnnl_post_ops_.append_sum(scale);
          break;
        }
        case kind_t::binary: {
          // 如果是 binary 操作，获取算法和预期的内存描述
          dnnl::algorithm algo = ops_params_[i].algo_;
          auto expected_md = ops_params_[i].expected_meta_;
          // 添加 binary 操作到 dnnl_post_ops_ 中，使用 format_tag::any 来确保性能优化
          dnnl_post_ops_.append_binary(algo, expected_md);
          break;
        }
        default:
          break;
      }
    }

    // 如果输出量化，则附加 eltwise linear 操作以调整输出的 scale 和 zero_point
    if (dst.is_quantized()) {
      // 添加 eltwise 操作来调整量化输出的 scale 和 zero_point
      // 注意：这里的除以 2 是为了处理 oneDNN 和 PyTorch 中 u8 量化的差异
      dnnl_post_ops_.append_eltwise(
          kind_with_linear, 1.f / q_scale_, q_zero_point_);
    }
    // 返回构建好的 dnnl_post_ops_
    return dnnl_post_ops_;
  }

  // 检查 ops_params_ 中是否包含 sum 操作
  bool with_sum() {
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::sum) {
        return true;
      }
    }
    return false;
  }

  // 检查 ops_params_ 中是否包含 binary 操作
  bool with_binary() {
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::binary) {
        return true;
      }
    }
    return false;
  }

  // 构建 binary post ops 中的 primitive_desc 和 args
  void construct_post_binary(
      dnnl::primitive_desc& pd,
      std::unordered_map<int, dnnl::memory>& args) {
    // 该函数用于构建 binary post ops 中的二进制内存描述
    // 根据 oneDNN 文档，二进制张量可以采用以下形状
    // 循环遍历 ops_params_ 中的每个操作参数
    auto engine =
        GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      // 获取当前操作的类型
      kind_t kind = ops_params_[i].kind_;
      // 如果当前操作是二元操作
      if (kind == kind_t::binary) {
        // 创建一个 DNNL 内存对象用于存储二元操作的数据
        dnnl::memory binary_m;
        // 获取当前操作的二元数据和元数据
        auto binary = ops_params_[i].binary_;
        auto md = ops_params_[i].meta_;
        // 查询预期的元数据以获得最佳性能
        auto expected_md = pd.query_md(
            dnnl::query::exec_arg_md,
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1);

        // 使用 onednn::make_onednn_memory 函数创建 DNNL 内存对象
        binary_m = at::native::onednn::make_onednn_memory(
          md, engine, binary.data_ptr()
        );

        // 将当前操作的信息插入参数列表 args 中
        args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, binary_m});
      }
    }
  }

  // 设置量化比例 q_scale_，用于将融合结果从 FP32 量化为 INT8，仅适用于 INT8 情况
  float q_scale_ = 1.0;
  // 设置量化零点 q_zero_point_
  int64_t q_zero_point_ = 0;
  // 存储一系列后操作参数的向量 ops_params_
  std::vector<PostOpParam> ops_params_;
  // 定义 DNNL 后操作序列对象 dnnl_post_ops_
  dnnl::post_ops dnnl_post_ops_;
};

// 结束 at::native::onednn 命名空间的定义
} // namespace at::native::onednn
```