# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Deconv.cpp`

```py
// 包含C10库中的XPU函数声明和ATen库
#include <c10/xpu/XPUFunctions.h>
#include <ATen/ATen.h>

// 包含oneDNN库的头文件
#include <oneapi/dnnl/dnnl.hpp>
// 包含ATen库中MKLDNN相关的头文件
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>

// 命名空间at::native::onednn，包含了以下函数和类的声明
namespace at::native::onednn {

// 返回与反卷积兼容的扩展后的膨胀参数
static inline dnnl::memory::dims deconv_compatible_dilation(IntArrayRef& dilation) {
  // 将传入的dilation转换为DNNL的格式，并减去1
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

// 返回与反卷积兼容的步幅参数
static inline std::vector<int64_t> compatible_groups_deconv_strides(
    const at::Tensor& weight,
    dnnl::memory::dims group_size) {
  // 获取weight张量的步幅
  std::vector<int64_t> strides = weight.strides().vec();
  // 调整步幅顺序以符合反卷积的要求
  strides[0] = weight.strides()[1];
  strides[1] = weight.strides()[0];
  // 在步幅数组的开头插入计算得到的值，以适应反卷积操作
  strides.insert(strides.begin(), group_size[2] * weight.strides()[0]);
  return strides;
}

// 计算反卷积的输出尺寸
dnnl::memory::dims deconv_dst_size(
    IntArrayRef src_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef dst_padding,
    int64_t groups) {
  // 获取输入张量的维度
  auto dim = src_size.size();
  dnnl::memory::dims dst_size(dim);
  // 获取卷积核的尺寸
  auto kernel_size = weight_size.slice(2);

  // 计算输出尺寸的各个维度
  dst_size[0] = src_size[0];
  dst_size[1] = weight_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    dst_size[d] = (src_size[d] - 1) * stride[d - 2] - 2 * padding[d - 2] +
        (dilation[d - 2] * (kernel_size[d - 2] - 1) + 1) + dst_padding[d - 2];
  }
  return dst_size;
}

// 返回反卷积的输入数据格式标签
static inline dnnl::memory::format_tag deconv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  // 根据维度和通道是否在最后的标志，返回对应的DNNL格式标签
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

// 返回反卷积的权重数据格式标签
static inline std::vector<int64_t> deconv_weight_fmt(
    const at::Tensor& weight,
    const int64_t ndim,
    dnnl::memory::dims weight_size,
    const bool grouped = false,
    const bool is_channels_last = false) {
  // 获取权重张量的步幅
  auto strides_ = weight.strides().vec();
  std::vector<int64_t> strides;
  // 如果是分组卷积，则调整步幅以适应反卷积的要求
  if (grouped) {
    strides = compatible_groups_deconv_strides(weight, weight_size);
  } else {
    strides = strides_;
    std::swap(strides[0], strides[1]);
  }
  return strides;
}

} // namespace at::native::onednn
// 定义函数，计算适用于反卷积的权重维度
static inline dnnl::memory::dims deconv_compatible_weight_dims(
    int64_t ndim,                            // 输入张量的维度
    int64_t groups,                          // 分组数
    int64_t oc,                              // 输出通道数
    int64_t ic,                              // 输入通道数
    IntArrayRef weight_size) {                // 权重张量的大小
  // 如果输入张量是3维
  if (ndim == 3) {
    auto kw = weight_size[2];                // 获取权重的宽度
    // 返回适合反卷积的权重维度
    return (groups != 1) ? dnnl::memory::dims({groups, oc / groups, ic / groups, kw})
                         : dnnl::memory::dims({oc, ic, kw});
  }
  // 如果输入张量是4维
  else if (ndim == 4) {
    auto kh = weight_size[2];                // 获取权重的高度
    auto kw = weight_size[3];                // 获取权重的宽度
    // 返回适合反卷积的权重维度
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : dnnl::memory::dims({oc, ic, kh, kw});
  }
  // 如果输入张量是5维
  else if (ndim == 5) {
    auto kd = weight_size[2];                // 获取权重的深度
    auto kh = weight_size[3];                // 获取权重的高度
    auto kw = weight_size[4];                // 获取权重的宽度
    // 返回适合反卷积的权重维度
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : dnnl::memory::dims({oc, ic, kd, kh, kw});
  }
  // 如果输入张量维度不支持，则抛出错误
  else {
    TORCH_CHECK(0, "unsupported dimension in xpu oneDNN deconvolution...");
  }
}

// 定义函数，获取反卷积操作的内存描述符
static std::tuple<
    dnnl::memory::desc,                      // 输入张量的内存描述符
    dnnl::memory::desc,                      // 权重张量的内存描述符
    dnnl::memory::desc>                      // 输出张量的内存描述符
deconv_get_plain_md(
    const at::Tensor& src,                   // 输入张量
    const at::Tensor& weight,                // 权重张量
    const at::Tensor& dst,                   // 输出张量
    int64_t groups,                          // 分组数
    bool is_channels_last_suggested) {       // 是否建议使用通道最后格式
  auto ndim = src.ndimension();              // 获取输入张量的维度
  auto src_data_t = get_onednn_dtype_include_double(src);  // 获取输入张量的数据类型
  auto fmt_src = deconv_src_fmt(ndim, is_channels_last_suggested);  // 获取输入张量的格式
  // 创建输入张量的内存描述符
  auto src_usr_md = dnnl::memory::desc(src.sizes().vec(), src_data_t, fmt_src);

  auto dst_data_t = get_onednn_dtype_include_double(dst);  // 获取输出张量的数据类型
  // 创建输出张量的内存描述符，与输入张量使用相同的格式
  auto dst_usr_md = dnnl::memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);

  auto ic = src.size(1);                     // 获取输入通道数
  auto oc = dst.size(1);                     // 获取输出通道数
  // 计算适用于反卷积的权重张量维度
  dnnl::memory::dims weight_size =
      deconv_compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  auto weight_dt = get_onednn_dtype_include_double(weight);  // 获取权重张量的数据类型
  auto fmt_weight = deconv_weight_fmt(
      weight, ndim, weight_size, groups != 1, is_channels_last_suggested);  // 获取权重张量的格式
  // 创建权重张量的内存描述符
  dnnl::memory::desc weight_usr_md = dnnl::memory::desc(weight_size, weight_dt, fmt_weight);

  return {src_usr_md, weight_usr_md, dst_usr_md};  // 返回输入、权重和输出张量的内存描述符元组
}

// 定义函数，执行反卷积操作
sycl::event deconvolution(
    at::Tensor& dst,                         // 输出张量
    const at::Tensor& src,                   // 输入张量
    const at::Tensor& weight,                // 权重张量
    const at::Tensor& bia,                   // 偏置张量
    IntArrayRef stride,                      // 步幅数组
    IntArrayRef padding,                     // 填充数组
    IntArrayRef dst_padding,                 // 输出填充数组
    IntArrayRef dilation,                    // 膨胀数组
    int64_t groups,                          // 分组数
    Attr& attr,                              // 属性对象
    // 获取当前设备的 GPU 引擎
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  // 获取当前设备的 GPU 流
  auto stream = GpuStreamManager::Instance().get_stream();

  // 检查是否推荐使用 channels last 格式进行反卷积操作
  bool is_channels_last_suggested = use_channels_last_for_conv(src, weight, /*is_transposed=*/true);

  // 创建用于张量的用户内存描述符和反卷积原语的内存描述符
  dnnl::memory::desc src_md, weight_md, dst_md;

  // 获取张量的内存描述符，以及反卷积操作的相关描述符
  std::tie(src_md, weight_md, dst_md) =
      deconv_get_plain_md(src, weight, dst, groups, is_channels_last_suggested);

  // 设置偏置的内存格式为 x（未指定特定格式）
  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bia_md = bia.defined()
      ? dnnl::memory::desc(
            {dst.size(1)}, get_onednn_dtype_include_double(bia), bia_fmt)
      : dnnl::memory::desc();

  // 创建卷积操作的原语描述符
  dnnl::memory::dims _stride = stride.vec();
  dnnl::memory::dims _padding = padding.vec();
  dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);

  // 构造原语属性
  dnnl::primitive_attr pattr;
  // 从属性中提取后操作，并设置到原语属性中
  dnnl::post_ops po = attr.extract_post_ops(dst);
  pattr.set_post_ops(po);
  // 如果支持确定性算法，则根据全局上下文中的设定设置确定性属性
  #if ONEDNN_SUPPORT_DETERMINISTIC
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        pattr.set_deterministic(true);
  #endif

  // 设置内存模式为用户定义
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 创建反卷积前向传播的原语描述符
  auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  // 创建内存对象以及阻塞张量对象，用于后续的内存分配
  dnnl::memory src_m, weight_m, dst_m, bia_m;
  at::Tensor src_blocked, weight_blocked, dst_blocked = dst;

  // 为源张量、权重张量和目标张量分配内存对象
  src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  weight_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  dst_m = make_onednn_memory(dst_md, engine, dst.data_ptr());

  // 创建用于存储操作参数的无序映射
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, weight_m});
  args.insert({DNNL_ARG_DST, dst_m});

  // 如果有偏置，则创建偏置内存对象并添加到参数中
  if (bia.defined()) {
    auto bia_m = make_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }
  
  // 如果属性包含二进制操作，则构造二进制后操作并添加到参数中
  if (attr.with_binary())
    attr.construct_post_binary(deconv_fwd_pd, args);

  // 获取所需的临时缓冲区大小，并为其分配张量内存
  size_t scratchpad_size = deconv_fwd_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = make_onednn_memory(
      deconv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // 创建反卷积前向传播的实际原语对象
  auto deconv_fwd = dnnl::deconvolution_forward(deconv_fwd_pd);
  // 执行反卷积操作并返回与此操作关联的事件对象
  sycl::event deconv_event = dnnl::sycl_interop::execute(deconv_fwd, stream, args, deps);
  // 返回反卷积操作关联的事件对象
  return deconv_event;
  // 调用 GpuEngineManager 实例获取当前设备的引擎对象
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
  // 调用 GpuStreamManager 实例获取当前设备上的流对象
  auto stream = GpuStreamManager::Instance().get_stream();

  // 根据反向卷积的输出、权重和输入数据，推断是否建议使用通道末尾存储布局
  bool is_channels_last_suggested =
      use_channels_last_for_conv(diff_dst, weight, /*is_transposed=*/true);

  // 创建用于内存描述的对象：源数据、权重、目标数据
  dnnl::memory::desc src_md, weight_md, dst_md;
  std::tie(src_md, weight_md, dst_md) =
      deconv_get_plain_md(
          diff_src, weight, diff_dst, groups, is_channels_last_suggested);

  // 定义偏置的内存描述，如果定义了偏置，使用相应的内存格式标签
  dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
  auto bias_md = bias_defined
      ? dnnl::memory::desc({diff_dst.size(1)}, weight_md.get_data_type(), bia_fmt)
      : dnnl::memory::desc();

  // 创建前向原语属性描述
  dnnl::primitive_attr pattr;
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 如果支持 ONEDNN 的确定性行为，编译以下代码
  #if ONEDNN_SUPPORT_DETERMINISTIC
    // 检查全局上下文中是否启用了确定性算法或者确定性的 MKLDNN
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        // 如果是，则设置当前操作为确定性操作
        pattr.set_deterministic(true);
  #endif

  // 将步长转换为 DNNL 需要的格式
  dnnl::memory::dims _stride = stride.vec();
  // 将填充转换为 DNNL 需要的格式
  dnnl::memory::dims _padding = padding.vec();
  // 将膨胀（dilation）转换为兼容的 DNNL 格式
  dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);
  // 创建反卷积前向传播的原始描述符
  auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      bias_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  // 创建反卷积反向传播数据的原始描述符
  auto deconv_backward_data_pd = dnnl::deconvolution_backward_data::primitive_desc(
      engine,
      dnnl::algorithm::deconvolution_direct,
      src_md,
      weight_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      deconv_fwd_pd);

  // 创建存储器对象：输入梯度（diff_src_m）、权重（wei_m）、输出梯度（diff_dst_m）
  dnnl::memory diff_dst_m, wei_m, diff_src_m;

  // 使用 make_onednn_memory 函数创建 OneDNN 内存对象
  diff_src_m = make_onednn_memory(src_md, engine, diff_src.data_ptr());
  wei_m = make_onednn_memory(weight_md, engine, weight.data_ptr());
  diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());

  // 插入参数到 unordered_map 中
  std::unordered_map<int, dnnl::memory> args;
  // 获取反卷积反向传播数据的 scratchpad 大小
  size_t scratchpad_size = deconv_backward_data_pd.scratchpad_desc().get_size();
  // 创建用于 scratchpad 的 Tensor
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  // 使用 make_onednn_memory 函数创建 scratchpad 对象
  auto scratchpad_memory = make_onednn_memory(
      deconv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
  // 插入 scratchpad 参数到 args 中
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
  // 插入输入梯度参数到 args 中
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  // 插入权重参数到 args 中
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  // 插入输出梯度参数到 args 中
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // 执行反卷积反向传播数据的原始操作
  auto deconv_backward_data =
      dnnl::deconvolution_backward_data(deconv_backward_data_pd);
  // 使用 SYCL 执行 OneDNN 操作，并返回事件对象
  sycl::event deconv_bwd_data_event = dnnl::sycl_interop::execute(deconv_backward_data, stream, args, deps);
  // 返回执行事件
  return deconv_bwd_data_event;
  // 定义函数 deconvolution_backward_weights，用于反向权重卷积计算
  sycl::event deconvolution_backward_weights(
      // 输出参数：权重梯度、偏置梯度、输入梯度、输入数据、步幅、填充、扩展、分组、依赖事件列表
      at::Tensor& diff_weight,
      at::Tensor& diff_bia,
      const at::Tensor& diff_dst,
      const at::Tensor& src,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups,
      const std::vector<sycl::event>& deps) {
    // 获取当前设备的 GPU 引擎
    auto engine =
        GpuEngineManager::Instance().get_engine({c10::kXPU, c10::xpu::current_device()});
    // 获取 GPU 流
    auto stream = GpuStreamManager::Instance().get_stream();

    // 根据推荐情况确定是否使用通道最后存储格式进行反向卷积计算
    bool is_channels_last_suggested =
        use_channels_last_for_conv(src, diff_dst, /*is_transposed=*/true);

    // 创建内存描述符：输入、权重和输出
    dnnl::memory::desc src_md, weight_md, dst_md;
    std::tie(src_md, weight_md, dst_md) = deconv_get_plain_md(
            src, diff_weight, diff_dst, groups, is_channels_last_suggested);

    // 定义偏置的内存描述符
    dnnl::memory::format_tag bia_fmt = dnnl::memory::format_tag::x;
    auto bia_md = diff_bia.defined()
        ? dnnl::memory::desc({diff_dst.size(1)}, src_md.get_data_type(), bia_fmt)
        : dnnl::memory::desc();

    // 创建前向传播的原始描述的提示
    dnnl::memory::dims _stride = stride.vec();
    dnnl::memory::dims _padding = padding.vec();
    dnnl::memory::dims _dilation = deconv_compatible_dilation(dilation);
    dnnl::primitive_attr pattr;

    // 如果支持 ONEDNN 的确定性算法，则设置确定性选项
    #if ONEDNN_SUPPORT_DETERMINISTIC
      if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
          pattr.set_deterministic(true);
    #endif
    // 设置内存模式为用户指定的临时存储区域模式
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // 创建反向权重卷积的原语描述符
    auto deconv_fwd_pd = dnnl::deconvolution_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward,
        dnnl::algorithm::deconvolution_direct,
        src_md,
        weight_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding,
        _padding,
        pattr);

    // 创建反向权重卷积的原语描述符
    auto deconv_bwd_w_pd = dnnl::deconvolution_backward_weights::primitive_desc(
        engine,
        dnnl::algorithm::deconvolution_direct,
        src_md,
        weight_md,
        bia_md,
        dst_md,
        _stride,
        _dilation,
        _padding,
        _padding,
        deconv_fwd_pd,
        pattr);

    // 创建 ONEDNN 的内存对象：输入、输出梯度、权重梯度
    dnnl::memory src_m, diff_dst_m, diff_weight_m;

    // 创建输入内存对象
    src_m = make_onednn_memory(src_md, engine, src.data_ptr());
    // 创建输出梯度内存对象
    diff_dst_m = make_onednn_memory(dst_md, engine, diff_dst.data_ptr());
    // 创建权重梯度内存对象
    diff_weight_m = make_onednn_memory(weight_md, engine, diff_weight.data_ptr());

    // 插入参数
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
    args.insert({DNNL_ARG_SRC, src_m});
    args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_weight_m});

    // 如果定义了偏置，创建偏置梯度内存对象并插入参数
    if (diff_bia.defined()) {
      dnnl::memory diff_bia_m =
          make_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    // 将参数插入参数集合，用于反向偏置梯度计算
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }

  // 计算反向权重卷积操作的临时存储空间大小
  size_t scratchpad_size = deconv_bwd_w_pd.scratchpad_desc().get_size();
  // 创建用于存储临时数据的张量，数据类型为字节，大小为 scratchpad_size
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, src.options().dtype(at::kByte), c10::nullopt);
  // 利用 OneDNN 内存工具创建存储器，关联张量的数据指针
  auto scratchpad_m = make_onednn_memory(
      deconv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  // 将临时存储器插入参数集合，用于反向权重卷积操作
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});

  // 执行反向权重卷积的基本操作
  auto deconv_bwd_w = dnnl::deconvolution_backward_weights(deconv_bwd_w_pd);

  // 使用 SYCL 接口执行反向权重卷积操作，返回事件对象
  sycl::event deconv_bwd_w_event = dnnl::sycl_interop::execute(deconv_bwd_w, stream, args, deps);
  // 返回反向权重卷积操作的事件对象
  return deconv_bwd_w_event;
}

// 结束 at::native::onednn 命名空间
} // namespace at::native::onednn
```