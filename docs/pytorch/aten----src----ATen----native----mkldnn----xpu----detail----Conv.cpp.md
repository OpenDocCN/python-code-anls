# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Conv.cpp`

```py
#include <c10/xpu/XPUFunctions.h>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <c10/core/MemoryFormat.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

// 定义输入尺寸和卷积核尺寸的默认维度常量
constexpr int src_batch_size_dim = 0;
constexpr int weight_dst_channels_dim = 0;

// 计算卷积操作后的目标输出尺寸
dnnl::memory::dims conv_dst_size(
    int64_t ndim,
    IntArrayRef src_size,
    IntArrayRef weight_size,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  dnnl::memory::dims dst_size(ndim);
  dst_size[0] = src_size[src_batch_size_dim]; // 设置批次维度大小
  dst_size[1] = weight_size[weight_dst_channels_dim]; // 设置卷积核输出通道维度大小
  for (int d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (weight_size[d] - 1) + 1;
    // 计算当前维度下的输出大小
    dst_size[d] =
        (src_size[d] +
         (padding_front_top_left[d - 2] + padding_back_bottom_right[d - 2]) -
         kernel) /
            stride[d - 2] +
        1;
  }
  return dst_size; // 返回计算得到的输出尺寸
}

// 调整卷积操作中的膨胀参数，将每个元素减去1
static inline dnnl::memory::dims compatible_dilation(IntArrayRef& dilation) {
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret; // 返回调整后的膨胀参数
}

// 返回卷积操作的输入数据格式标签，根据维度和通道顺序确定
static inline dnnl::memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? dnnl::memory::format_tag::ncw
        : ((ndim == 4) ? dnnl::memory::format_tag::nchw
                       : ((ndim == 5) ? dnnl::memory::format_tag::ncdhw
                                      : dnnl::memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? dnnl::memory::format_tag::nwc
        : ((ndim == 4) ? dnnl::memory::format_tag::nhwc
                       : ((ndim == 5) ? dnnl::memory::format_tag::ndhwc
                                      : dnnl::memory::format_tag::undef));
  }
}

// 返回卷积操作的卷积核格式标签，根据维度、是否分组和通道顺序确定
static inline dnnl::memory::format_tag conv_weight_fmt(
    const int64_t ndim,
    const bool grouped = false,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? (grouped ? dnnl::memory::format_tag::goiw : dnnl::memory::format_tag::oiw)
        : (ndim == 4)
        ? (grouped ? dnnl::memory::format_tag::goihw : dnnl::memory::format_tag::oihw)
        : ((ndim == 5) ? (grouped ? dnnl::memory::format_tag::goidhw
                                  : dnnl::memory::format_tag::oidhw)
                       : dnnl::memory::format_tag::undef);
  } else {
    // channels_last 的情况下选择相应的格式标签
    return (ndim == 3)
        ? dnnl::memory::format_tag::nwc
        : ((ndim == 4) ? dnnl::memory::format_tag::nhwc
                       : ((ndim == 5) ? dnnl::memory::format_tag::ndhwc
                                      : dnnl::memory::format_tag::undef));
  }
}
    # 返回一个布尔值，判断输入的张量维度是否为3
    return (ndim == 3)
        # 如果是3维，根据grouped标志返回不同的内存格式标签
        ? (grouped ? dnnl::memory::format_tag::gowi : dnnl::memory::format_tag::owi)
        # 如果不是3维，进一步判断是否为4维
        : (ndim == 4)
        # 如果是4维，根据grouped标志返回不同的内存格式标签
        ? (grouped ? dnnl::memory::format_tag::gohwi : dnnl::memory::format_tag::ohwi)
        # 如果不是4维，再判断是否为5维
        : ((ndim == 5) 
            # 如果是5维，根据grouped标志返回不同的内存格式标签
            ? (grouped ? dnnl::memory::format_tag::godhwi : dnnl::memory::format_tag::odhwi)
            # 如果既不是3、4、5维，则返回未定义的内存格式标签
            : dnnl::memory::format_tag::undef);
}

// 定义一个静态内联函数，用于确定兼容的权重维度
static inline dnnl::memory::dims compatible_weight_dims(
    const int64_t ndim,                     // 权重张量的维度
    const int64_t groups,                   // 分组卷积的组数
    const int64_t oc,                       // 输出通道数
    const int64_t ic,                       // 输入通道数
    const IntArrayRef wsizes) {             // 权重张量的大小数组
  if (ndim == 3) {                          // 如果权重张量是3维的
    auto kw = wsizes[2];                    // 获取权重张量的第3维度（宽度）
    return (groups != 1) ?                  // 如果有分组卷积
        dnnl::memory::dims({groups, oc / groups, ic / groups, kw})  // 返回分组卷积的权重维度
        : dnnl::memory::dims({oc, ic, kw});  // 否则返回常规卷积的权重维度
  } else if (ndim == 4) {                   // 如果权重张量是4维的
    auto kh = wsizes[2];                    // 获取权重张量的第3维度（高度）
    auto kw = wsizes[3];                    // 获取权重张量的第4维度（宽度）
    return (groups != 1)                    // 如果有分组卷积
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})  // 返回分组卷积的权重维度
        : dnnl::memory::dims({oc, ic, kh, kw});  // 否则返回常规卷积的权重维度
  } else if (ndim == 5) {                   // 如果权重张量是5维的
    auto kd = wsizes[2];                    // 获取权重张量的第3维度（深度）
    auto kh = wsizes[3];                    // 获取权重张量的第4维度（高度）
    auto kw = wsizes[4];                    // 获取权重张量的第5维度（宽度）
    return (groups != 1)                    // 如果有分组卷积
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})  // 返回分组卷积的权重维度
        : dnnl::memory::dims({oc, ic, kd, kh, kw});  // 否则返回常规卷积的权重维度
  }

  return {};  // 如果维度不匹配，返回空维度数组
}

// 定义一个函数，用于获取卷积操作的内存描述符
static std::tuple<
    dnnl::memory::desc,
    dnnl::memory::desc,
    dnnl::memory::desc>
conv_get_md(
    const at::Tensor& src,                  // 输入张量
    const at::Tensor& weight,               // 权重张量
    const at::Tensor& dst,                  // 输出张量
    int64_t groups,                         // 分组数
    bool is_channels_last) {                // 是否通道最后
  // 创建来自输入/权重/输出张量的内存描述符
  dnnl::memory::desc src_usr_md, weight_usr_md, dst_usr_md;
  auto ndim = src.ndimension();            // 获取输入张量的维度数
  auto fmt_src = conv_src_fmt(ndim, is_channels_last);  // 获取输入格式

  auto src_size = src.sizes().vec();       // 获取输入张量的大小
  auto src_data_t = get_onednn_dtype_include_double(src);  // 获取包括双精度的数据类型
  src_usr_md = dnnl::memory::desc(src_size, src_data_t, fmt_src);  // 创建输入张量的内存描述符

  auto dst_size = dst.sizes().vec();       // 获取输出张量的大小
  auto dst_data_t = get_onednn_dtype_include_double(dst);  // 获取包括双精度的数据类型
  dst_usr_md = dnnl::memory::desc(dst_size, dst_data_t, fmt_src);  // 创建输出张量的内存描述符

  auto ic = src.size(1);                   // 获取输入通道数
  auto oc = dst.size(1);                   // 获取输出通道数
  auto wei_data_t = get_onednn_dtype_include_double(weight);  // 获取包括双精度的权重数据类型
  dnnl::memory::dims weight_size = 
      compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());  // 获取兼容的权重维度
  auto fmt_weight = conv_weight_fmt(
      ndim,
      groups != 1,
      is_channels_last);                   // 获取权重格式
  weight_usr_md = dnnl::memory::desc(weight_size, wei_data_t, fmt_weight);  // 创建权重张量的内存描述符

  return {src_usr_md, weight_usr_md, dst_usr_md};  // 返回内存描述符元组
}

// 定义一个函数，执行卷积操作
sycl::event convolution(
    at::Tensor& dst,                        // 输出张量
    const at::Tensor& src,                  // 输入张量
    const at::Tensor& weight,               // 权重张量
    const at::Tensor& bia,                  // 偏置张量
    IntArrayRef padding_front_top_left,     // 前置填充的上左边界
    IntArrayRef padding_back_bottom_right,  // 后置填充的下右边界
    IntArrayRef stride,                     // 步长
    IntArrayRef dilation,                   // 空洞卷积的扩张
    int64_t groups,                         // 分组数
    Attr& attr,                             // 属性对象
    // 获取当前设备的 GPU 引擎
    const std::vector<sycl::event>& deps) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  // 获取当前设备的 GPU 流
  auto stream = GpuStreamManager::Instance().get_stream();

  // 判断是否使用 channels_last 来进行卷积操作
  bool is_channels_last = use_channels_last_for_conv(src, weight, false);

  // 创建用于存储张量的用户内存描述 (usr_md)，以及卷积原语的内存描述 (src_md, weight_md, dst_md)
  dnnl::memory::desc src_md, weight_md, dst_md;
  std::tie(src_md, weight_md, dst_md) = conv_get_md(src, weight, dst, groups, is_channels_last);

  // 定义偏置的内存格式
  auto bia_fmt = dnnl::memory::format_tag::x;
  // 如果定义了偏置张量 bia，则创建其内存描述
  auto bia_md = bia.defined()
      ? dnnl::memory::desc(
            {dst.size(1)}, get_onednn_dtype_include_double(bia), bia_fmt)
      : dnnl::memory::desc();

  // 创建卷积原语描述符
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding_front_top_left = padding_front_top_left.vec();
  dnnl::memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  dnnl::memory::dims _dilation = compatible_dilation(dilation);

  // 提取后处理操作
  dnnl::primitive_attr pattr;
  dnnl::post_ops po = attr.extract_post_ops(dst);
  pattr.set_post_ops(po);

  // 设置内存管理方式为用户自定义
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 如果支持确定性操作，则设置为确定性模式
  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn()){
        pattr.set_deterministic(true);
    }
  #endif

  // 创建卷积前向原语描述符
  auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  // 创建 OneDNN 内存对象用于存储输入、权重、输出、偏置
  dnnl::memory src_m, weight_m, dst_m, bia_m;
  at::Tensor src_blocked, weight_blocked, dst_blocked = dst;

  // 为输入张量创建 OneDNN 内存对象
  src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  // 为权重张量创建 OneDNN 内存对象
  weight_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  // 为输出张量创建 OneDNN 内存对象
  dst_m = make_onednn_memory(dst_md, engine, dst.data_ptr());

  // 创建参数映射，将 OneDNN 内存对象映射到对应的卷积操作参数
  std::unordered_map<int, dnnl::memory> args;
  if (bia.defined()) {
    // 如果定义了偏置张量，则为其创建 OneDNN 内存对象，并添加到参数映射中
    bia_m = make_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }
  // 获取预期的输出张量描述符
  auto expected_dst_md = conv_fwd_pd.dst_desc();
  // 如果属性中包含二进制后处理，则构建对应的二进制后处理
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, args);

  // 将输入、权重、输出 OneDNN 内存对象插入参数映射中
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, weight_m});
  args.insert({DNNL_ARG_DST, dst_m});

  // 获取卷积前向操作需要的临时空间大小
  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  // 创建用于卷积操作的临时缓冲区张量
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  // 为临时空间创建 OneDNN 内存对象
  auto scratchpad_m = make_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  // 将临时空间内存对象插入参数映射中
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // 创建卷积前向操作对象
  auto conv_forward = dnnl::convolution_forward(conv_fwd_pd);
  // 在给定的流上执行卷积前向操作，返回事件对象
  auto conv_fwd_event = dnnl::sycl_interop::execute(conv_forward, stream, args, deps);

  // 返回卷积前向操作事件对象
  return conv_fwd_event;
}
}

// 反向权重卷积操作，计算卷积层的权重梯度
sycl::event convolution_backward_weights(
    at::Tensor& diff_weight,  // 输出：权重的梯度
    at::Tensor& diff_bia,    // 输出：偏置的梯度
    const at::Tensor& diff_dst,  // 输入：上一层的误差梯度
    const at::Tensor& src,    // 输入：当前层的输入数据
    IntArrayRef diff_weight_aten_size,  // 输入：权重张量的大小
    IntArrayRef padding_front_top_left,  // 输入：前置填充尺寸
    IntArrayRef padding_back_bottom_right,  // 输入：后置填充尺寸
    IntArrayRef stride,       // 输入：卷积核的步长
    IntArrayRef dilation,     // 输入：卷积核的扩展
    int64_t groups,           // 输入：卷积组数
    const std::vector<sycl::event>& deps) {  // 输入：依赖的事件

  // 获取当前设备的 GPU 引擎
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  // 获取 GPU 流
  auto stream = GpuStreamManager::Instance().get_stream();

  // 判断是否使用通道最后格式
  bool is_channels_last = use_channels_last_for_conv(src, diff_dst, /*is_transposed=*/false);

  // 创建 DNNL 内存描述符
  dnnl::memory::desc src_md, weight_md, dst_md;
  std::tie(src_md, weight_md, dst_md) =
      conv_get_md(src, diff_weight, diff_dst, groups, is_channels_last);
  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bia_md = diff_bia.defined()
      ? dnnl::memory::desc({diff_dst.size(1)}, src_md.get_data_type(), bia_fmt)
      : dnnl::memory::desc();

  // 创建前向卷积的基本属性
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding_front_top_left = padding_front_top_left.vec();
  dnnl::memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  dnnl::memory::dims _dilation = compatible_dilation(dilation);
  dnnl::primitive_attr pattr;

  // 如果支持确定性算法，则设置为确定性模式
  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn()){
        pattr.set_deterministic(true);
    }
  #endif

  // 设置为用户模式的临时缓冲区
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 创建前向卷积的原语描述
  auto conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  // 创建反向权重卷积的原语描述
  auto conv_bwd_w_pd = dnnl::convolution_backward_weights::primitive_desc(
      engine,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      conv_fwd_pd,
      pattr);

  // 创建反向卷积需要的内存
  at::Tensor expected_src, expected_diff_dst, expected_diff_weight;
  dnnl::memory src_m, diff_dst_m, diff_weight_m;

  // 使用 OneDNN 接口创建对应的内存对象
  src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());
  diff_weight_m = make_onednn_memory(weight_md, engine, diff_weight.data_ptr());

  // 插入需要的参数
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_weight_m});

  // 如果定义了偏置，则添加偏置参数
  if (diff_bia.defined()) {
    dnnl::memory diff_bia_m =
        make_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});

    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }
    // 将 {DNNL_ARG_DIFF_BIAS, diff_bia_m} 插入到 args 中
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }

  // 计算反向卷积权重操作需要的临时内存大小
  size_t scratchpad_size = conv_bwd_w_pd.scratchpad_desc().get_size();
  // 创建一个 ATen 张量作为临时内存，用于存储 scratchpad_size 大小的字节
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  // 在 OneDNN 中创建一个内存对象，使用 scratchpad_tensor 的数据指针
  auto scratchpad_m = make_onednn_memory(
      conv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  // 将 {DNNL_ARG_SCRATCHPAD, scratchpad_m} 插入到 args 中
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // 执行反向卷积权重的原语操作
  auto conv_bwd_w = dnnl::convolution_backward_weights(conv_bwd_w_pd);
  // 使用 SYCL 执行 OneDNN 的反向卷积权重操作，并返回 SYCL 事件
  sycl::event conv_bwd_w_event = dnnl::sycl_interop::execute(conv_bwd_w, stream, args, deps);

  // 返回执行反向卷积权重操作的 SYCL 事件
  return conv_bwd_w_event;
  // 创建 SYCL 事件以表示反向数据卷积操作
  sycl::event convolution_backward_data(
      // 输出张量的引用，用于存储反向传播的梯度
      at::Tensor& diff_src,
      // 输入梯度张量，包含前一层传播回来的梯度信息
      const at::Tensor& diff_dst,
      // 权重张量，用于卷积计算
      const at::Tensor& weight,
      // 前部、顶部和左侧填充的大小
      IntArrayRef padding_front_top_left,
      // 后部、底部和右侧填充的大小
      IntArrayRef padding_back_bottom_right,
      // 卷积步长
      IntArrayRef stride,
      // 卷积扩展率
      IntArrayRef dilation,
      // 分组数量
      int64_t groups,
      // 是否定义了偏置项
      bool bias_defined,
      // 依赖事件的向量，表示操作的依赖关系
      const std::vector<sycl::event>& deps) {
    // 获取当前设备的 XPU 引擎
    auto engine =
        GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
    // 获取 GPU 流以执行计算任务
    auto stream = GpuStreamManager::Instance().get_stream();

    // 检查是否使用通道最后的布局进行卷积操作
    bool is_channels_last = use_channels_last_for_conv(diff_dst, weight, /*is_transposed=*/false);

    // 创建内存描述符
    dnnl::memory::desc src_md, weight_md, dst_md;
    // 获取输入、权重和输出的内存描述符
    std::tie(src_md, weight_md, dst_md) =
        conv_get_md(diff_src, weight, diff_dst, groups, is_channels_last);
    // 设置偏置项的内存描述符格式为单一维度
    dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
    auto bia_md = bias_defined
        ? dnnl::memory::desc({diff_dst.size(1)}, weight_md.get_data_type(), bia_fmt)
        : dnnl::memory::desc();

    // 创建前向传播原始操作的属性
    dnnl::primitive_attr pattr;

    // 如果支持 ONEDNN 的确定性算法，则设置为确定性计算
    #if ONEDNN_SUPPORT_DETERMINISTIC
      if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn()){
          pattr.set_deterministic(true);
  // 结束条件判断
  }
  // 结束宏定义条件

  // 设置 scratchpad 模式为用户定义
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 获取 stride 的维度
  dnnl::memory::dims _stride = stride.vec();
  // 获取前部分 padding 的维度
  dnnl::memory::dims _padding_front_top_left = padding_front_top_left.vec();
  // 获取后部分 padding 的维度
  dnnl::memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  // 获取 dilation 的维度
  dnnl::memory::dims _dilation = compatible_dilation(dilation);

  // 创建前向卷积的原语描述对象
  auto conv_forward_pd = dnnl::convolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      pattr);

  // 创建反向卷积数据的原语描述对象
  auto conv_backward_data_pd = dnnl::convolution_backward_data::primitive_desc(
      engine,
      dnnl::algorithm::convolution_direct,
      src_md,
      weight_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      conv_forward_pd,
      pattr);

  // 创建内存对象
  at::Tensor expected_src, expected_wei, expected_dst;
  dnnl::memory diff_dst_m, wei_m, diff_src_m;

  // 使用数据指针创建 OneDNN 内存对象
  diff_src_m = make_onednn_memory(src_md, engine, diff_src.data_ptr());
  wei_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());

  // 插入参数
  std::unordered_map<int, dnnl::memory> args;
  // 计算 scratchpad 的大小
  size_t scratchpad_size = conv_backward_data_pd.scratchpad_desc().get_size();
  // 创建与 scratchpad 相关联的 Tensor
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  // 创建与 scratchpad 相关的 OneDNN 内存对象
  auto scratchpad_memory = make_onednn_memory(
      conv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
  // 插入参数：SCRATCHPAD
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
  // 插入参数：DIFF_DST
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  // 插入参数：WEIGHTS
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  // 插入参数：DIFF_SRC
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // 执行原语
  auto conv_backward_data =
      dnnl::convolution_backward_data(conv_backward_data_pd);
  // 执行原语并返回事件
  auto conv_backward_data_event = dnnl::sycl_interop::execute(conv_backward_data, stream, args, deps);
  // 返回反向卷积数据的事件
  return conv_backward_data_event;
}

} // namespace at::native::onednn
```