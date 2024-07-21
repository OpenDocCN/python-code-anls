# `.\pytorch\aten\src\ATen\native\quantized\cpu\conv_serialization.h`

```
// 预编译指令，确保头文件只被包含一次
#pragma once

// 包含 ATen 库中的 Tensor 和 List 头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>

// 包含量化相关的 CPU 实现工具函数头文件
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>

// 包含用于 C++ 中范围迭代的头文件
#include <c10/util/irange.h>

// 根据宏定义条件编译 CPU 特定的头文件
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

// 根据宏定义条件包含不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
#endif

// 包含元组头文件，用于使用 std::tuple
#include <tuple>

/* Convolution prepacked parameters serialization.
 *
 * Version 1
 *
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim
 *  4. padding x kSpatialDim
 *  5. dilation x kSpatialDim
 *  6. groups
 *
 * Version 2
 *
 * - Fields:
 *  0. version (string)
 *  1. list of non-optional tensors
 *    0: packed parameters (int16_t)
 *      - kSpatialDim
 *      - stride x kSpatialDim
 *      - padding x kSpatialDim
 *      - dilation x kSpatialDim
 *      - output_padding x kSpatialDim
 *      - groups
 *      - transpose (0 or 1)
 *    1: weight
 *  2. list of optional tensors
 *    0: bias
 *
 * Version 3
 *
 * - Fields:
 *  0. version (int64_t)
 *  1. list of int64_t configuration values
 *    - kSpatialDim
 *    - stride x kSpatialDim
 *    - padding x kSpatialDim
 *    - dilation x kSpatialDim
 *    - output_padding x kSpatialDim
 *    - groups
 *    - flags (bitmask)
 *      - (1 << 0) transpose (1 = yes)
 *  2. list of optional tensors
 *    0: None (helps with type inference)
 *    1: weight (this must be present)
 *    2: bias
 */

// 使用别名定义版本 2 的序列化类型，包含字符串版本号、非可选张量列表和可选张量列表
using ConvParamsSerializationTypeV2 = std::tuple<
  std::string,                           // 版本号
  std::vector<at::Tensor>,               // 非可选张量列表
  std::vector<std::optional<at::Tensor>> // 可选张量列表
>;

// 使用别名定义版本 3 的序列化类型，包含整数版本号、配置值列表和可选张量列表
using ConvParamsSerializationTypeV3 = std::tuple<
  int64_t,                               // 版本号
  std::vector<int64_t>,                  // 配置值列表
  std::vector<std::optional<at::Tensor>> // 可选张量列表
>;

// 解析任何历史卷积打包参数格式到当前格式的模板函数
template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV3 parse_conv_serialized_state(c10::IValue v) {

  // 根据 IValue 的内容确定版本号
  int version = -1;
  if (v.isTuple()) {
    const auto& elements = v.toTupleRef().elements();
    if (!elements.empty()) {
      auto firstElement = elements[0];
      if (firstElement.isTensor()) {
        version = 1;
      } else if (firstElement.isString()) {
        const std::string& version_str = firstElement.toStringRef();
        // 注意：未解析字符串以自动处理错误的输入
        if (version_str == "2") {
          version = 2;
        }
      } else if (firstElement.isInt()) {
        auto raw_version = firstElement.toInt();
        if (raw_version == 3) {
          version = 3;
        }
      }
    }
  }
  
  // 断言确保版本号已正确解析
  TORCH_INTERNAL_ASSERT(version != -1, "Unable to parse serialization version");

  // 如果版本号为 1，执行以下代码块
  if (version == 1) {
    // version 1 - convert to version 3 manually

    // 获取元组中的所有元素
    const auto& elements = v.toTupleRef().elements();

    // 从元组中获取权重张量
    at::Tensor weight = elements[0].toTensor();
    
    // 从元组中获取可选的偏置张量
    std::optional<at::Tensor> bias = elements[1].toOptional<at::Tensor>();

    // 从元组中获取包含多个张量的列表，用于步长（stride）
    torch::List<at::Tensor> stride_x_kSpatialDim = elements[2].toTensorList();

    // 从元组中获取包含多个张量的列表，用于填充（padding）
    torch::List<at::Tensor> padding_x_kSpatialDim = elements[3].toTensorList();

    // 从元组中获取包含多个张量的列表，用于膨胀（dilation）
    torch::List<at::Tensor> dilation_x_kSpatialDim = elements[4].toTensorList();

    // 从元组中获取分组数张量
    at::Tensor groups = elements[5].toTensor();

    // 创建存储配置值的向量，并预留足够的空间
    std::vector<int64_t> config_vals;
    config_vals.reserve(
        stride_x_kSpatialDim.size() + padding_x_kSpatialDim.size() +
        dilation_x_kSpatialDim.size() + kSpatialDim + 3);
    
    // 添加空间维度的数量到配置值向量
    config_vals.push_back(kSpatialDim);
    
    // 遍历步长列表，并将第一个步长的整数部分添加到配置值向量
    for (const auto i : c10::irange(stride_x_kSpatialDim.size())) {
      auto stride = stride_x_kSpatialDim.get(i);
      config_vals.push_back(stride[0].item<int16_t>());
    }
    
    // 遍历填充列表，并将第一个填充的整数部分添加到配置值向量
    for (const auto i : c10::irange(padding_x_kSpatialDim.size())) {
      auto padding = padding_x_kSpatialDim.get(i);
      config_vals.push_back(padding[0].item<int16_t>());
    }
    
    // 遍历膨胀列表，并将第一个膨胀的整数部分添加到配置值向量
    for (const auto i : c10::irange(dilation_x_kSpatialDim.size())) {
      auto dilation = dilation_x_kSpatialDim.get(i);
      config_vals.push_back(dilation[0].item<int16_t>());
    }
    
    // 在版本 1 中不存在输出填充，因此使用默认值填充配置值向量
    for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
      config_vals.push_back(0);
    }
    
    // 添加分组数到配置值向量
    config_vals.push_back(groups[0].item<int16_t>());
    
    // 在版本 1 中不存在转置，因此使用默认值添加到配置值向量
    config_vals.push_back(0);

    // 创建存储张量的可选值的向量，并初始化为默认值
    std::vector<std::optional<at::Tensor>> tensors;
    tensors.emplace_back(); // 添加默认空值
    tensors.emplace_back(weight); // 添加权重张量
    tensors.emplace_back(bias); // 添加偏置张量

    // 设置版本号为 3
    int64_t version = 3;
    
    // 返回版本号、配置值向量和张量向量的元组
    return std::tie(version, config_vals, tensors);
  } else if (version == 2) {
    // version 2
    
    // 获取元组中的所有元素
    const auto& elements = v.toTupleRef().elements();
    
    // 从元组中获取非可选张量列表的第二个元素
    std::vector<at::Tensor> non_optional = elements[1].toTensorList().vec();
    
    // 创建存储可选张量的向量
    std::vector<std::optional<at::Tensor>> optional;

    // 如果第三个元素是张量列表，则将其转换为可选张量添加到向量中
    if (elements[2].isTensorList()) {
      for (const auto& elem : elements[2].toTensorList()) {
        optional.emplace_back(static_cast<at::Tensor>(elem));
      }
    } else {
      // 否则，将每个元素转换为可选张量并添加到向量中
      for (const auto& elem : elements[2].toList()) {
        optional.emplace_back(static_cast<c10::IValue>(elem).toOptional<at::Tensor>());
      }
    }
    
    // 如果可选张量向量为空，则添加默认值
    if (optional.empty()) {
      optional.emplace_back();
    }

    // 访问非可选张量列表中第一个张量的访问器，类型为 int16_t，1 维度
    auto config_a = non_optional[0].accessor<int16_t, 1>();
    
    // 创建存储配置值的向量，并预留足够的空间
    std::vector<int64_t> config_vals;
    config_vals.reserve(config_a.size(0));
    
    // 将配置值从访问器中添加到配置值向量中
    for (const auto i : c10::irange(config_a.size(0))) {
      config_vals.emplace_back(config_a[i]);
    }

    // 获取非可选张量列表中的权重和可选张量向量中的偏置
    auto weight = non_optional[1];
    auto bias = optional[0];

    // 创建存储张量的可选值的向量，并初始化为默认值
    std::vector<std::optional<at::Tensor>> tensors;
    tensors.emplace_back(); // 添加默认空值
    tensors.emplace_back(weight); // 添加权重张量
    tensors.emplace_back(bias); // 添加偏置张量
    # 定义一个 int64_t 类型的变量 version，赋值为 3
    int64_t version = 3;
    # 如果 version 等于 3，则返回一个 tuple，包含 version、config_vals 和 tensors
    return std::tie(version, config_vals, tensors);
  } else if (version == 3) {
    # 如果 version 等于 3，则将变量 v 转换为 ConvParamsSerializationTypeV3 类型，并返回结果
    return v.to<ConvParamsSerializationTypeV3>();
  } else {
    # 如果 version 不等于 3，则触发内部断言错误，并输出相关信息，此分支应不会执行到
    TORCH_INTERNAL_ASSERT(false, "Unexpected serialized qconv version: ",
        version);
  }
}

#define QCONV_SERIALIZATION_VERSION 2

#if QCONV_SERIALIZATION_VERSION == 2
// 定义一个类型别名 ConvParamsSerializationType，指向 ConvParamsSerializationTypeV2 类型
using ConvParamsSerializationType = ConvParamsSerializationTypeV2;

// 序列化卷积参数为 ConvParamsSerializationTypeV2 类型对象
template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV2 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {

  // 设定版本号为字符串 "2"
  std::string version = "2";
  // 非可选的张量数组
  std::vector<at::Tensor> non_optional;
  // 可选的张量数组
  std::vector<std::optional<at::Tensor>> optional;

  // 创建一个用于存储卷积参数的压缩 int8_t 张量
  std::vector<int16_t> params_vec;
  // 将空间维度 kSpatialDim 添加到 params_vec
  params_vec.push_back(kSpatialDim);
  // 添加 stride 到 params_vec
  auto stride = params->stride().vec();
  params_vec.insert(params_vec.end(), stride.begin(), stride.end());
  // 添加 padding 到 params_vec
  auto padding = params->padding().vec();
  params_vec.insert(params_vec.end(), padding.begin(), padding.end());
  // 添加 dilation 到 params_vec
  auto dilation = params->dilation().vec();
  params_vec.insert(params_vec.end(), dilation.begin(), dilation.end());
  // 添加 output_padding 到 params_vec
  auto output_padding = params->output_padding().vec();
  params_vec.insert(params_vec.end(), output_padding.begin(),
                    output_padding.end());
  // 添加 groups 到 params_vec
  params_vec.push_back(params->groups());
  // 添加 transpose 到 params_vec
  params_vec.push_back(params->transpose());
  // 获取 params_vec 的大小
  int64_t vec_size = params_vec.size();
  // 创建 params_tensor，从 params_vec 的数据创建张量，类型为 int16_t
  at::Tensor params_tensor = at::from_blob(
      params_vec.data(), {vec_size},
      at::TensorOptions().dtype(at::kShort))
    // 克隆以保留数据的所有权
    .clone();

  // 解压 params 得到 weight 和 bias
  auto [weight, bias] = params->unpack();

  // 将 params_tensor 添加到 non_optional
  non_optional.emplace_back(std::move(params_tensor));
  // 将 weight 添加到 non_optional
  non_optional.emplace_back(std::move(weight));
  // 将 bias 添加到 optional
  optional.emplace_back(std::move(bias));

  // 返回包含版本号、non_optional 和 optional 的元组
  return std::tie(version, non_optional, optional);
}

#elif QCONV_SERIALIZATION_VERSION == 3
// 定义一个类型别名 ConvParamsSerializationType，指向 ConvParamsSerializationTypeV3 类型
using ConvParamsSerializationType = ConvParamsSerializationTypeV3;

// 序列化卷积参数为 ConvParamsSerializationTypeV3 类型对象
template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV3 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {
  // 创建一个存储配置值的 int64_t 数组
  std::vector<int64_t> config_vals;
  // 将空间维度 kSpatialDim 添加到 config_vals
  config_vals.push_back(kSpatialDim);
  // 添加 stride 到 config_vals
  auto stride = params->stride().vec();
  config_vals.insert(config_vals.end(), stride.begin(), stride.end());
  // 添加 padding 到 config_vals
  auto padding = params->padding().vec();
  config_vals.insert(config_vals.end(), padding.begin(), padding.end());
  // 添加 dilation 到 config_vals
  auto dilation = params->dilation().vec();
  config_vals.insert(config_vals.end(), dilation.begin(), dilation.end());
  // 添加 output_padding 到 config_vals
  auto output_padding = params->output_padding().vec();
  config_vals.insert(config_vals.end(), output_padding.begin(),
                    output_padding.end());
  // 添加 groups 到 config_vals
  config_vals.push_back(params->groups());
  // 添加 transpose 到 config_vals
  config_vals.push_back(params->transpose());

  // 解压 params 得到 weight 和 bias
  auto [weight, bias] = params->unpack();

  // 创建一个包含可选张量的数组 tensors
  std::vector<std::optional<at::Tensor>> tensors;
  tensors.emplace_back();
  tensors.emplace_back(weight);
  tensors.emplace_back(bias);

  // 设定版本号为 3
  int64_t version = 3;
  // 返回包含版本号、config_vals 和 tensors 的元组
  return std::tie(version, config_vals, tensors);
}

#else
// 如果 QCONV_SERIALIZATION_VERSION 既不是 2 也不是 3，则抛出错误
#error "Invalid qconv serialization version."
#endif

// 反序列化卷积参数为 c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> 对象
template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv(
    // 解析序列化的卷积参数，从给定状态对象中提取版本号、配置值和张量数据
    ConvParamsSerializationTypeV3 state) {
      // 使用结构化绑定获取版本号、配置值和张量数据
      auto [version, config_vals, tensors] = state;
      // 断言版本号为3，验证序列化的卷积版本是否符合预期
      TORCH_INTERNAL_ASSERT(version == 3, "Unexpected serialized qconv version: ", version);
    
      // 检查张量数据的数量是否为3
      TORCH_CHECK(tensors.size() == 3, "Wrong number of tensors", tensors.size());
      // 提取权重张量
      std::optional<at::Tensor> weight = tensors[1];
      // 提取偏置张量
      std::optional<at::Tensor> bias = tensors[2];
      // 断言权重张量不为空，序列化的卷积数据中应始终包含权重
      TORCH_INTERNAL_ASSERT(weight, "Weight should always be present in serialized qconv.");
    
      // 初始化存储卷积参数的列表：步长、填充、输出填充、扩张
      torch::List<int64_t> stride, padding, output_padding, dilation;
      // 跳过维度常量 kSpatialDim，从索引1开始读取配置值
      int idx = 1;
      // 读取并存储步长
      for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
        stride.emplace_back(config_vals.at(idx));
        idx++;
      }
      // 读取并存储填充
      for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
        padding.emplace_back(config_vals.at(idx));
        idx++;
      }
      // 读取并存储扩张
      for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
        dilation.emplace_back(config_vals.at(idx));
        idx++;
      }
      // 读取并存储输出填充
      for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
        // 断言索引未超出配置值的大小范围
        TORCH_INTERNAL_ASSERT(idx < static_cast<int64_t>(config_vals.size()),
            "Unexpected index = ", idx, " for config_vals of size ",
            config_vals.size());
        output_padding.emplace_back(config_vals.at(idx));
        idx++;
      }
      // 读取并存储组数
      int64_t groups = config_vals.at(idx);
      idx++;
      // 读取并存储标志位
      int64_t flags = config_vals.at(idx);
      idx++;
      // 断言已读取完所有配置值
      TORCH_INTERNAL_ASSERT(idx == static_cast<int64_t>(config_vals.size()),
          "Unexpected length of config_vals, expected ",
          idx,
          " got ",
          config_vals.size());
    
      // 检查标志位以确定是否转置卷积
      bool transpose = flags & (1 << 0);
    
      // 提取除了转置位外的其它标志位
      int64_t other_flags = flags & ~(1 << 0);
      // 断言其它标志位应为0，即除了转置位外不应设置其它标志
      TORCH_INTERNAL_ASSERT(other_flags == 0, "Unexpected flags set in ", flags, ".");
    
      // 获取全局上下文引用
      auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
  // 如果使用 FBGEMM 引擎
  if (ctx.qEngine() == at::QEngine::X86) {
    // 如果上下文的量化引擎是 X86
#if AT_MKLDNN_ENABLED()
    // 如果启用了 MKLDNN
    bool use_onednn = onednn_utils::should_use_onednn_quant(
        weight.value(), transpose, groups, output_padding);
    // 判断是否应该使用 OneDNN 进行量化
    if (use_onednn) {
      // 如果应该使用 OneDNN
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
        weight.value(),
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose
      );
    }
#endif // AT_MKLDNN_ENABLED()
    // 返回使用 X86 的预打包卷积权重
    return PackedConvWeight<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  } // x86
#endif // USE_FBGEMM

#ifdef USE_FBGEMM
  // 如果使用 FBGEMM 引擎
  if (ctx.qEngine() == at::QEngine::FBGEMM) {
    // 如果上下文的量化引擎是 FBGEMM
    return PackedConvWeight<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  }
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
  // 如果使用 PYTORCH_QNNPACK 引擎
  if (ctx.qEngine() == at::QEngine::QNNPACK) {
    // 如果上下文的量化引擎是 QNNPACK
    TORCH_CHECK(
        kSpatialDim == 2,
        "prepack/__setstate__: QNNPACK only supports Conv2d "
        "now.");
    // 检查空间维度是否为 2，QNNPACK 只支持 Conv2d
    return PackedConvWeightsQnnp<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  }
#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
  // 如果启用了 MKLDNN
  if (ctx.qEngine() == at::QEngine::ONEDNN) {
    // 如果上下文的量化引擎是 ONEDNN
    return PackedConvWeightsOnednn<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  }
#endif // AT_MKLDNN_ENABLED()

// 如果找不到适合的引擎则抛出错误
TORCH_CHECK(
  false,
  "Didn't find engine for when deserializing ConvPackedParams: ",
  toString(ctx.qEngine()));
}


这段代码是根据上下文中的量化引擎选择不同的预打包卷积权重操作。根据不同的引擎类型（X86、FBGEMM、QNNPACK、ONEDNN），选择相应的预打包函数。如果没有找到匹配的引擎，则会抛出错误信息。
```