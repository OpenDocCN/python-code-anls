# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions.cpp`

```py
// 引入 Torch 库中的头文件，用于 JIT 引擎的张量表达式外部函数
#include <torch/csrc/jit/tensorexpr/external_functions.h>

// 引入 ATen 库中各种需要的头文件
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/BinaryOps.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/conv_serialization.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <Aten/quantized/QTensorImpl.h>  // 注意：这里的大小写有误，应为 <at/quantized/QTensorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <utility>

// 定义命名空间 torch::jit::tensorexpr 中的静态函数 deduce_memory_format，根据步幅和维度推断内存格式
namespace torch::jit::tensorexpr {

static c10::MemoryFormat deduce_memory_format(
    c10::IntArrayRef strides,
    c10::IntArrayRef dims) {
  // 检查步幅和维度的长度和特定条件来推断内存格式
  if (strides.size() == 4 && strides[3] == dims[1] && strides[1] == 1l) {
    return c10::MemoryFormat::ChannelsLast;
  }
  // 默认返回连续内存格式
  return c10::MemoryFormat::Contiguous;
}

// 重载 deduce_memory_format 函数，接受 std::vector<int64_t> 类型的步幅和维度
static c10::MemoryFormat deduce_memory_format(
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dims) {
  // 调用前面定义的 deduce_memory_format 函数，将 std::vector 转换为 c10::IntArrayRef
  return deduce_memory_format(
      c10::IntArrayRef(strides), c10::IntArrayRef(dims));
}

// 定义 from_blob_quantized 函数，从给定的数据创建量化张量
static at::Tensor from_blob_quantized(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    double qscale,
    int64_t qzero,
    at::ScalarType dtype) {
  // 推断内存格式
  auto memory_format = deduce_memory_format(strides, sizes);
  // 创建一个空的量化张量 qx
  auto qx = at::_empty_affine_quantized(
      sizes,
      dtype,
      c10::kStrided,
      at::kCPU,
      false,
      qscale,
      qzero,
      memory_format);
  // 获取 qx 的 QTensorImpl，并设置其外部数据指针和大小信息
  auto qtensor_impl = static_cast<at::QTensorImpl*>(qx.unsafeGetTensorImpl());
  auto typeMeta = c10::scalarTypeToTypeMeta(dtype);
  std::size_t size = 1;
  for (std::int64_t s : sizes) {
    size *= static_cast<std::size_t>(s);
  }
  // 使用 ShareExternalPointer 方法共享外部数据指针
  qtensor_impl->ShareExternalPointer(
      c10::InefficientStdFunctionContext::makeDataPtr(
          data, [](void*) {}, at::kCPU),
      typeMeta,
      size * typeMeta.itemsize());
  // 设置 qx 的大小和步幅信息，并返回该量化张量 qx
  qtensor_impl->set_sizes_and_strides(sizes, strides);
  return qx;
}

// 定义 constructTensors 函数，构造张量的向量，这些张量可以来自多个外部数据缓冲区
std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg) {
  std::vector<void*> buf_data_vec;
  std::vector<std::vector<int64_t>> buf_dims_vec;
  std::vector<std::vector<int64_t>> buf_strides_vec;
  std::vector<c10::ScalarType> buf_dtypes_vec;
  int64_t buf_dims_idx = 0;
  int64_t buf_strides_idx = 0;
  // 遍历每个缓冲区的索引范围，构建相应的数据结构
  for (const auto i : c10::irange(bufs_num)) {
    buf_data_vec.push_back(buf_data[i]);
    buf_dims_vec.emplace_back();
    buf_strides_vec.emplace_back();
    // 对 buf_strides_vec 添加一个空的子向量，用于存储当前维度的步长信息

    for (const auto dim : c10::irange(buf_ranks[i])) {
      (void)dim;
      // 遍历当前张量的维度数量（buf_ranks[i]），这里使用 (void)dim 确保不使用 dim 变量
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
      // 将当前维度的大小（buf_dims[buf_dims_idx++]）添加到 buf_dims_vec[i] 中
      buf_strides_vec[i].push_back(buf_strides[buf_strides_idx++]);
      // 将当前维度的步长（buf_strides[buf_strides_idx++]）添加到 buf_strides_vec[i] 中
    }
    // 将当前张量的数据类型转换为 c10::ScalarType 类型，并添加到 buf_dtypes_vec 中
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  // 创建存储张量的向量
  std::vector<at::Tensor> tensors;

  // 如果 qdataArg 中没有值
  if (!qdataArg.has_value()) {
    // 遍历 buf_data_vec 中的每个元素
    for (const auto i : c10::irange(buf_data_vec.size())) {
      // 定义张量的选项
      auto options = at::TensorOptions()
                         // NOLINTNEXTLINE
                         .dtype(buf_dtypes_vec[i]) // 设置张量的数据类型为 buf_dtypes_vec[i]
                         .layout(at::kStrided)     // 设置张量的布局为 kStrided
                         .device(at::kCPU)         // 设置张量的设备为 CPU
                         .memory_format(deduce_memory_format(
                             // NOLINTNEXTLINE
                             buf_strides_vec[i], // 推断张量的内存格式
                             // NOLINTNEXTLINE
                             buf_dims_vec[i]))   // 推断张量的维度格式
                         .requires_grad(false);  // 设置张量不需要梯度计算

      // 从给定的内存块 buf_data_vec[i] 创建张量，使用指定的维度和步长
      auto tensor = at::from_blob(
          // NOLINTNEXTLINE
          buf_data_vec[i],      // 内存块的指针
          buf_dims_vec[i],      // 张量的维度
          buf_strides_vec[i],   // 张量的步长
          options);             // 张量的选项

      // 将创建的张量添加到 tensors 向量中
      tensors.emplace_back(tensor);
    }
  } else {
    // 处理量化的情况

    // 创建存储可选的量化数据的向量 qdata，初始化为 c10::nullopt
    std::vector<std::optional<QIData>> qdata(bufs_num, c10::nullopt);

    // 将 qdataArg 中的量化数据填充到 qdata 中
    for (const auto& qd : *qdataArg) {
      qdata[qd.first] = qd.second;
    }

    // 遍历 buf_data_vec 中的每个元素
    for (const auto i : c10::irange(buf_data_vec.size())) {
      // 定义张量的选项
      auto options = at::TensorOptions()
                         // NOLINTNEXTLINE
                         .dtype(buf_dtypes_vec[i]) // 设置张量的数据类型为 buf_dtypes_vec[i]
                         .layout(at::kStrided)     // 设置张量的布局为 kStrided
                         .device(at::kCPU)         // 设置张量的设备为 CPU
                         .memory_format(deduce_memory_format(
                             // NOLINTNEXTLINE
                             buf_strides_vec[i], // 推断张量的内存格式
                             // NOLINTNEXTLINE
                             buf_dims_vec[i]))   // 推断张量的维度格式
                         .requires_grad(false);  // 设置张量不需要梯度计算

      // 如果 qdata[i] 不为空，则创建量化的张量
      if (auto qd = qdata[i]) {
        // 使用量化的方式从内存块 buf_data_vec[i] 创建张量
        auto tensor = from_blob_quantized(
            // NOLINTNEXTLINE
            buf_data_vec[i],  // 内存块的指针
            buf_dims_vec[i],  // 张量的维度
            buf_strides_vec[i],  // 张量的步长
            qd->scale,      // 量化的缩放因子
            qd->zero,       // 量化的零点
            qd->scalarType); // 量化的数据类型

        // 将创建的张量添加到 tensors 向量中
        tensors.emplace_back(tensor);
      } else {
        // 否则，从内存块 buf_data_vec[i] 创建普通张量
        auto tensor = at::from_blob(
            // NOLINTNEXTLINE
            buf_data_vec[i],  // 内存块的指针
            buf_dims_vec[i],  // 张量的维度
            buf_strides_vec[i],  // 张量的步长
            options);             // 张量的选项

        // 将创建的张量添加到 tensors 向量中
        tensors.emplace_back(tensor);
      }
    }
  }
  // 返回存储所有张量的向量 tensors
  return tensors;
}

// 构造张量的函数，接受多个参数来生成张量的向量
static std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,               // 缓冲区数量
    void** buf_data,                // 缓冲区数据的指针数组
    int64_t* buf_ranks,             // 缓冲区的秩数组
    int64_t* buf_dims,              // 缓冲区维度的数组
    int64_t* buf_strides,           // 缓冲区步长的数组
    int8_t* buf_dtypes,             // 缓冲区数据类型的数组
    std::vector<std::pair<size_t, QIData>> qdata) {  // 包含QIData的可选参数
  // 将qdata移动到std::optional中
  std::optional<std::vector<std::pair<size_t, QIData>>> opt = std::move(qdata);
  // 调用另一个构造张量的函数，并返回其结果
  return constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes, opt);
}

// 构造张量的函数，支持输出数量为 bufs_out_num
std::vector<at::Tensor> constructTensors2(
    int64_t bufs_in_num,                    // 输入缓冲区数量
    void** buf_data,                        // 缓冲区数据的指针数组
    int64_t* buf_ranks,                     // 缓冲区的秩数组
    int64_t* buf_dims,                      // 缓冲区维度的数组
    int64_t* buf_strides,                   // 缓冲区步长的数组
    int8_t* buf_dtypes,                     // 缓冲区数据类型的数组
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg,  // 可选的QIData参数
    size_t bufs_out_num) {                  // 输出缓冲区数量
  // 初始化各种向量
  std::vector<void*> buf_data_vec;
  std::vector<std::vector<int64_t>> buf_dims_vec;
  std::vector<std::vector<int64_t>> buf_strides_vec;
  std::vector<c10::ScalarType> buf_dtypes_vec;
  int64_t buf_dims_idx = 0;
  int64_t buf_strides_idx = 0;

  // 遍历输入缓冲区数量的范围
  for (const auto i : c10::irange(bufs_in_num)) {
    // 将缓冲区数据加入向量
    buf_data_vec.push_back(buf_data[bufs_out_num + i]);
    // 初始化维度和步长的向量
    buf_dims_vec.emplace_back();
    buf_strides_vec.emplace_back();
    // 遍历当前缓冲区的秩范围
    for (const auto dim : c10::irange(buf_ranks[i])) {
      (void)dim;  // 确保编译器不会发出未使用变量的警告
      // 添加维度和步长到相应的向量中
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
      buf_strides_vec[i].push_back(buf_strides[buf_strides_idx++]);
    }
    // 将数据类型转换为c10::ScalarType并加入向量
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  // 初始化张量向量
  std::vector<at::Tensor> tensors;
  at::Tensor und;
  // 遍历输出缓冲区数量的范围
  for (const auto i : c10::irange(bufs_out_num)) {
    (void)i;  // 确保编译器不会发出未使用变量的警告
    tensors.emplace_back(und);  // 将未定义的张量加入向量
  }

  // 如果没有提供QIData参数，则执行以下操作
  if (!qdataArg.has_value()) {
    // 遍历缓冲区数据向量的范围
    for (const auto i : c10::irange(buf_data_vec.size())) {
      // 构造张量选项
      auto options = at::TensorOptions()
                         .dtype(buf_dtypes_vec[i])  // 指定数据类型
                         .layout(at::kStrided)      // 使用步进布局
                         .device(at::kCPU)          // 指定设备为CPU
                         .memory_format(deduce_memory_format(
                             buf_strides_vec[i], buf_dims_vec[i]))  // 推断内存格式
                         .requires_grad(false);      // 不需要梯度

      // 从给定的缓冲区创建张量
      auto tensor = at::from_blob(
          buf_data_vec[i],    // 数据指针
          buf_dims_vec[i],    // 维度
          buf_strides_vec[i], // 步长
          options);           // 张量选项

      // 将生成的张量加入张量向量
      tensors.emplace_back(tensor);
    }
  } else {
    // 处理量化数据
    std::vector<std::optional<QIData>> qdata(bufs_in_num, c10::nullopt);
    // 遍历qdataArg中的每个QIData，并将其与相应的缓冲区关联
    for (const auto& qd : *qdataArg) {
      qdata[qd.first - bufs_out_num] = qd.second;
    }
    // 遍历 buf_data_vec 的索引范围，使用 auto 声明 i 为循环变量
    for (const auto i : c10::irange(buf_data_vec.size())) {
      // 创建 TensorOptions 对象，链式调用设置选项
      auto options = at::TensorOptions()
                         // NOLINTNEXTLINE
                         .dtype(buf_dtypes_vec[i])  // 设置数据类型
                         .layout(at::kStrided)      // 设置布局为 Strided
                         .device(at::kCPU)          // 设置设备为 CPU，TODO: 同时支持 GPU
                         .memory_format(deduce_memory_format(
                             // NOLINTNEXTLINE
                             buf_strides_vec[i],     // 推断内存格式需要的步长信息
                             // NOLINTNEXTLINE
                             buf_dims_vec[i]))       // 推断内存格式需要的维度信息
                         .requires_grad(false);     // 设置不需要梯度计算

      // 如果 qdata[i] 存在
      if (auto qd = qdata[i]) {
        // 使用 from_blob_quantized 创建量化后的张量
        // NOLINTNEXTLINE
        auto tensor = from_blob_quantized(
            buf_data_vec[i],          // 数据缓冲区
            buf_dims_vec[i],          // 张量维度
            buf_strides_vec[i],       // 张量步长
            qd->scale,                // 量化参数：缩放因子
            qd->zero,                 // 量化参数：零点偏移
            qd->scalarType);          // 量化参数：标量类型
        tensors.emplace_back(tensor); // 将张量加入到向量 tensors 中
      } else {
        // 使用 at::from_blob 创建张量
        // NOLINTNEXTLINE
        auto tensor = at::from_blob(
            buf_data_vec[i],          // 数据缓冲区
            buf_dims_vec[i],          // 张量维度
            buf_strides_vec[i],       // 张量步长
            options);                 // 使用之前设置的选项
        tensors.emplace_back(tensor); // 将张量加入到向量 tensors 中
      }
    }
  }
  // 返回张量向量 tensors
  return tensors;
// 如果给定的额外参数数量大于0，则检查是否为7个参数，并且缓冲区数量为4
if (args_num > 0) {
    // 断言确保参数数量为7且缓冲区数量为4，用于检验偏置张量的存在
    TORCH_INTERNAL_ASSERT(args_num == 7 && bufs_num == 4);
    // 获取偏置张量
    const at::Tensor& b = tensors[3];

    // 提取额外参数中的步幅值
    int64_t strideH = extra_args[0];
    int64_t strideW = extra_args[1];
}
    // 从 extra_args 数组中获取填充高度值
    int64_t paddingH = extra_args[2];
    // 从 extra_args 数组中获取填充宽度值
    int64_t paddingW = extra_args[3];
    // 从 extra_args 数组中获取扩展高度值
    int64_t dilationH = extra_args[4];
    // 从 extra_args 数组中获取扩展宽度值
    int64_t dilationW = extra_args[5];
    // 从 extra_args 数组中获取分组数
    int64_t groups = extra_args[6];

    try {
      // 尝试进行二维卷积操作，使用给定的参数和权重 w，偏置 b
      r = at::conv2d(
          x,
          w,
          b,
          {strideH, strideW},    // 使用指定的步长参数
          {paddingH, paddingW},  // 使用指定的填充参数
          {dilationH, dilationW},  // 使用指定的扩展参数
          groups);  // 使用指定的分组数
    } catch (...) {
    }
  } else {
    try {
      // 尝试进行简单的二维卷积操作，只传入输入张量 x 和权重 w
      r = at::conv2d(x, w);
    } catch (...) {
    }
  }

  // TODO: 是否可以使用 conv2d 的输出版本？
  // 将 r 张量的数据拷贝到 buf_data 数组的第一个缓冲区中
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
// 定义一个名为 nnc_aten_quantized_conv1d 的函数，用于执行量化的一维卷积操作
void nnc_aten_quantized_conv1d(
    int64_t bufs_num,                 // 输入缓冲区数量
    void** buf_data,                  // 输入缓冲区数据的指针数组
    int64_t* buf_ranks,               // 输入缓冲区的秩（维度）
    int64_t* buf_dims,                // 输入缓冲区的维度
    int64_t* buf_strides,             // 输入缓冲区的步幅
    int8_t* buf_dtypes,               // 输入缓冲区的数据类型
    int64_t,                          // 参数（未使用）
    int64_t* extra_args) {            // 额外的参数数组

  // 从额外参数中获取输入量化参数
  const double x_qscale = ((double*)extra_args)[0];
  const int64_t x_qzero = extra_args[1];
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);

  // 构建输入张量
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});

  // 从缓冲区中获取卷积操作的参数
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);

  // 从额外参数中获取输出量化参数
  const double out_qscale = ((double*)extra_args)[3];
  const int64_t out_qzero = extra_args[4];

  // 对输入张量进行操作，执行量化一维卷积
  // NOLINTNEXTLINE
  auto qx = tensors[1].unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
  auto r = convPackedParams->apply(qx, out_qscale, out_qzero);
  r = r.squeeze_(quant_utils::kConv1dSqueezeDim + 2);

  // 将结果拷贝回输出缓冲区
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

// 定义一个名为 nnc_aten_quantized_conv1d_out 的函数，用于执行输出版本的量化一维卷积操作
void nnc_aten_quantized_conv1d_out(
    int64_t bufs_in_num,              // 输入缓冲区数量
    void** buf_data,                  // 输入缓冲区数据的指针数组
    int64_t* buf_ranks,               // 输入缓冲区的秩（维度）
    int64_t* buf_dims,                // 输入缓冲区的维度
    int64_t* buf_strides,             // 输入缓冲区的步幅
    int8_t* buf_dtypes,               // 输入缓冲区的数据类型
    int64_t,                          // 参数（未使用）
    int64_t* extra_args) {            // 额外的参数数组

  // 确定输出缓冲区的数量
  const size_t bufs_out_num = 1u;

  // 从额外参数中获取输入量化参数
  const double x_qscale = ((double*)extra_args)[0];
  const int64_t x_qzero = extra_args[1];
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);

  // 构建输入张量
  auto tensors = constructTensors2(
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}},
      bufs_out_num);

  // 从缓冲区中获取卷积操作的参数
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);

  // 从额外参数中获取输出量化参数
  const double out_qscale = ((double*)extra_args)[3];
  const int64_t out_qzero = extra_args[4];

  // 对输入张量进行操作，执行输出版本的量化一维卷积
  // NOLINTNEXTLINE
  auto qx = tensors[1].unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
  auto r = convPackedParams->apply(qx, out_qscale, out_qzero);
  r = r.squeeze_(quant_utils::kConv1dSqueezeDim + 2);

  // 将结果拷贝回输出缓冲区，并增加引用计数
  buf_data[0] = r.data_ptr();
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

// 定义一个名为 nnc_aten_quantized_conv2d 的函数，用于执行量化的二维卷积操作
void nnc_aten_quantized_conv2d(
    int64_t bufs_num,                 // 输入缓冲区数量
    void** buf_data,                  // 输入缓冲区数据的指针数组
    int64_t* buf_ranks,               // 输入缓冲区的秩（维度）
    int64_t* buf_dims,                // 输入缓冲区的维度
    int64_t* buf_strides,             // 输入缓冲区的步幅
    int8_t* buf_dtypes,               // 输入缓冲区的数据类型
    int64_t,                          // 参数（未使用）
    // 定义一个函数，接受多个参数，包括 bufs_num、buf_data、buf_ranks、buf_dims、buf_strides、buf_dtypes 和 extra_args
    int64_t* extra_args) {
  // 从 extra_args 数组中提取 x_qscale，并转换为 double 类型
  const double x_qscale = ((double*)extra_args)[0];
  // 从 extra_args 数组中提取 x_qzero，并转换为 int64_t 类型
  const int64_t x_qzero = extra_args[1];
  // 从 extra_args 数组中提取 x_qdtype，并转换为 c10::ScalarType 类型
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  // 调用 constructTensors 函数，传入多个参数，其中包括一个包含额外参数的字典
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});
  // 将 buf_data[2] 强制转换为 ConvPackedParamsBase<2>* 类型，并赋值给 convPackedParams
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);
  // 从 extra_args 数组中提取 out_qscale，并转换为 double 类型
  const double out_qscale = ((double*)extra_args)[3];
  // 从 extra_args 数组中提取 out_qzero，并转换为 int64_t 类型
  const int64_t out_qzero = extra_args[4];
  // 使用 convPackedParams 对象的 apply 方法，对 tensors[1] 执行卷积操作，传入 out_qscale 和 out_qzero
  auto r = convPackedParams->apply(tensors[1], out_qscale, out_qzero);
  // 将 r 的数据拷贝到 buf_data[0] 中，拷贝的长度为 r.element_size() * r.numel()
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

// 函数定义：nnc_aten_quantized_conv2d_out
void nnc_aten_quantized_conv2d_out(
    int64_t bufs_in_num,            // 输入缓冲区数量
    void** buf_data,                // 输入缓冲区数据指针数组
    int64_t* buf_ranks,             // 输入缓冲区的秩数组
    int64_t* buf_dims,              // 输入缓冲区的维度数组
    int64_t* buf_strides,           // 输入缓冲区的步长数组
    int8_t* buf_dtypes,             // 输入缓冲区的数据类型数组
    int64_t,                        // 未使用的参数
    int64_t* extra_args) {          // 额外参数数组

  const size_t bufs_out_num = 1u;   // 输出缓冲区数量设为1
  const double x_qscale = ((double*)extra_args)[0];   // 从额外参数中获取输入的量化比例因子
  const int64_t x_qzero = extra_args[1];              // 从额外参数中获取输入的量化零点
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);   // 从额外参数中获取输入的量化数据类型
  auto tensors = constructTensors2(
      bufs_in_num,                  // 传入输入缓冲区数量
      buf_data,                     // 传入输入缓冲区数据指针数组
      buf_ranks,                    // 传入输入缓冲区的秩数组
      buf_dims,                     // 传入输入缓冲区的维度数组
      buf_strides,                  // 传入输入缓冲区的步长数组
      buf_dtypes,                   // 传入输入缓冲区的数据类型数组
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}},  // 传入输入的量化参数信息
      bufs_out_num);                // 传入输出缓冲区数量
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);   // 将第三个缓冲区数据解释为包含卷积参数的对象指针
  const double out_qscale = ((double*)extra_args)[3];   // 从额外参数中获取输出的量化比例因子
  const int64_t out_qzero = extra_args[4];              // 从额外参数中获取输出的量化零点
  // NOLINTNEXTLINE
  auto r = convPackedParams->apply(tensors[1], out_qscale, out_qzero);   // 使用卷积参数对象进行量化卷积操作
  buf_data[0] = r.data_ptr();     // 将结果的数据指针存入第一个输入缓冲区
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());   // 增加结果的引用计数
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();   // 将结果的引用指针存入最后一个缓冲区

}

// 函数定义：nnc_aten_quantized_conv2d_relu
void nnc_aten_quantized_conv2d_relu(
    int64_t bufs_num,               // 缓冲区数量
    void** buf_data,                // 缓冲区数据指针数组
    int64_t* buf_ranks,             // 缓冲区的秩数组
    int64_t* buf_dims,              // 缓冲区的维度数组
    int64_t* buf_strides,           // 缓冲区的步长数组
    int8_t* buf_dtypes,             // 缓冲区的数据类型数组
    int64_t,                        // 未使用的参数
    int64_t* extra_args) {          // 额外参数数组

  const double x_qscale = ((double*)extra_args)[0];   // 从额外参数中获取输入的量化比例因子
  const int64_t x_qzero = extra_args[1];              // 从额外参数中获取输入的量化零点
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);   // 从额外参数中获取输入的量化数据类型
  auto tensors = constructTensors(
      bufs_num,                     // 传入缓冲区数量
      buf_data,                     // 传入缓冲区数据指针数组
      buf_ranks,                    // 传入缓冲区的秩数组
      buf_dims,                     // 传入缓冲区的维度数组
      buf_strides,                  // 传入缓冲区的步长数组
      buf_dtypes,                   // 传入缓冲区的数据类型数组
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});  // 传入输入的量化参数信息
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);   // 将第三个缓冲区数据解释为包含卷积参数的对象指针
  const double out_qscale = ((double*)extra_args)[3];   // 从额外参数中获取输出的量化比例因子
  const int64_t out_qzero = extra_args[4];              // 从额外参数中获取输出的量化零点
  // NOLINTNEXTLINE
  auto r = convPackedParams->apply_relu(tensors[1], out_qscale, out_qzero);   // 使用卷积参数对象进行量化卷积+ReLU操作
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());   // 将结果复制到第一个缓冲区的内存中
}

// 函数定义：nnc_aten_quantized_conv2d_relu_out
void nnc_aten_quantized_conv2d_relu_out(
    int64_t bufs_in_num,            // 输入缓冲区数量
    void** buf_data,                // 输入缓冲区数据指针数组
    int64_t* buf_ranks,             // 输入缓冲区的秩数组
    int64_t* buf_dims,              // 输入缓冲区的维度数组
    int64_t* buf_strides,           // 输入缓冲区的步长数组
    int8_t* buf_dtypes,             // 输入缓冲区的数据类型数组
    int64_t,                        // 未使用的参数
    int64_t* extra_args) {          // 额外参数数组
    // 定义需要输出的缓冲区数量
    int64_t* extra_args) {
  // 设置输出缓冲区的数量为1
  const size_t bufs_out_num = 1u;
  // 从额外参数中获取输入张量的量化比例因子
  const double x_qscale = ((double*)extra_args)[0];
  // 从额外参数中获取输入张量的量化零点
  const int64_t x_qzero = extra_args[1];
  // 从额外参数中获取输入张量的数据类型
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  // 调用函数构造输入张量和输出张量
  auto tensors = constructTensors2(
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      // 创建包含量化参数的字典
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}},
      bufs_out_num);
  // 从缓冲区中解释获取卷积层的打包参数
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);
  // 从额外参数中获取输出张量的量化比例因子
  const double out_qscale = ((double*)extra_args)[3];
  // 从额外参数中获取输出张量的量化零点
  const int64_t out_qzero = extra_args[4];
  // 调用卷积层打包参数对象的应用ReLU函数
  // NOLINTNEXTLINE：忽略下一行的Lint检查
  auto r = convPackedParams->apply_relu(tensors[1], out_qscale, out_qzero);
  // 将ReLU结果的数据指针存储到第一个输出缓冲区中
  buf_data[0] = r.data_ptr();
  // 增加ReLU结果的引用计数
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
  // 将ReLU结果的指针存储到输入缓冲区后续位置中
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

void nnc_aten_quantized_linear(
    int64_t bufs_num,                                    // 定义函数 nnc_aten_quantized_linear，接受多个参数
    void** buf_data,                                     // 缓冲区数据的指针数组
    int64_t* buf_ranks,                                  // 缓冲区的秩（维度）
    int64_t* buf_dims,                                   // 缓冲区的维度大小
    int64_t* buf_strides,                                // 缓冲区的步长
    int8_t* buf_dtypes,                                  // 缓冲区的数据类型
    int64_t,                                              // 未命名的 int64_t 参数
    int64_t* extra_args) {                               // 额外的参数数组
  const double x_qscale = ((double*)extra_args)[0];      // 提取输入量化参数的比例因子
  const int64_t x_qzero = extra_args[1];                 // 提取输入量化参数的零点
  const c10::ScalarType x_qdtype =                       // 将输入量化参数的数据类型转换为 c10::ScalarType
      static_cast<c10::ScalarType>(extra_args[2]);
  auto tensors = constructTensors(                       // 构建输入张量集合
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}}); // 使用输入量化参数创建张量
  auto linearPackedParams =                              // 将第三个缓冲区数据解释为线性层打包参数
      reinterpret_cast<LinearPackedParamsBase*>(buf_data[2]);
  const double out_qscale = ((double*)extra_args)[3];    // 提取输出量化参数的比例因子
  const int64_t out_qzero = extra_args[4];               // 提取输出量化参数的零点
  // NOLINTNEXTLINE                                      // 禁止下一行的 lint 检查
  auto r = linearPackedParams->apply(                    // 调用线性层打包参数的 apply 方法
      tensors[1], out_qscale, out_qzero);                // 应用线性层操作并返回结果
  memcpy(buf_data[0], r.const_data_ptr(),                // 将结果数据复制到第一个缓冲区
         r.element_size() * r.numel());
}

void nnc_aten_quantized_linear_out(
    int64_t bufs_in_num,                                 // 定义函数 nnc_aten_quantized_linear_out，接受多个输入缓冲区的数量
    void** buf_data,                                     // 输入和输出缓冲区数据的指针数组
    int64_t* buf_ranks,                                  // 输入和输出缓冲区的秩（维度）
    int64_t* buf_dims,                                   // 输入和输出缓冲区的维度大小
    int64_t* buf_strides,                                // 输入和输出缓冲区的步长
    int8_t* buf_dtypes,                                  // 输入和输出缓冲区的数据类型
    int64_t,                                              // 未命名的 int64_t 参数
    int64_t* extra_args) {                               // 额外的参数数组
  const size_t bufs_out_num = 1u;                        // 输出缓冲区的数量为 1
  const double x_qscale = ((double*)extra_args)[0];      // 提取输入量化参数的比例因子
  const int64_t x_qzero = extra_args[1];                 // 提取输入量化参数的零点
  const c10::ScalarType x_qdtype =                       // 将输入量化参数的数据类型转换为 c10::ScalarType
      static_cast<c10::ScalarType>(extra_args[2]);
  auto tensors = constructTensors2(                      // 构建输入和输出张量集合
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}}, // 使用输入量化参数创建张量
      bufs_out_num);
  auto linearPackedParams =                              // 将第三个缓冲区数据解释为线性层打包参数
      reinterpret_cast<LinearPackedParamsBase*>(buf_data[2]);
  const double out_qscale = ((double*)extra_args)[3];    // 提取输出量化参数的比例因子
  const int64_t out_qzero = extra_args[4];               // 提取输出量化参数的零点
  // NOLINTNEXTLINE                                      // 禁止下一行的 lint 检查
  auto r = linearPackedParams->apply(                    // 调用线性层打包参数的 apply 方法
      tensors[1], out_qscale, out_qzero);                // 应用线性层操作并返回结果
  buf_data[0] = r.data_ptr();                            // 将结果数据的指针存储到第一个缓冲区
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get()); // 增加结果数据的引用计数
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get(); // 将结果数据的引用存储到输出缓冲区
}

void nnc_aten_quantized_linear_relu(
    int64_t bufs_num,                                    // 定义函数 nnc_aten_quantized_linear_relu，接受多个参数
    void** buf_data,                                     // 缓冲区数据的指针数组
    int64_t* buf_ranks,                                  // 缓冲区的秩（维度）
    int64_t* buf_dims,                                   // 缓冲区的维度大小
    int64_t* buf_strides,                                // 缓冲区的步长
    int8_t* buf_dtypes,                                  // 缓冲区的数据类型
    int64_t,                                              // 未命名的 int64_t 参数
    int64_t* extra_args) {                               // 额外的参数数组
  const double x_qscale = ((double*)extra_args)[0];      // 提取输入量化参数的比例因子
  const int64_t x_qzero = extra_args[1];                 // 提取输入量化参数的零点
  const c10::ScalarType x_qdtype =                       // 将输入量化参数的数据类型转换为 c10::ScalarType
      static_cast<c10::ScalarType>(extra_args[2]);
  auto tensors = constructTensors(                       // 构建输入张量集合
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}}); // 使用输入量化参数创建张量
  auto linearPackedParams =                              // 将第三个缓冲区数据解释为线性层打包参数
      reinterpret_cast<LinearPackedParamsBase*>(buf_data[2]);
  const double out_qscale = ((double*)extra_args)[3];    // 提取输出量化参数的比例因子
  const int64_t out_qzero = extra_args[4];               // 提取输出量化参数的零点
  // NOLINTNEXTLINE                                      // 禁止下一行的 lint 检查
  auto r = linearPackedParams->apply_relu(               // 调用线性层打包参数的 apply_relu 方法
      tensors[1], out_qscale, out_qzero);                // 应用带 ReLU 的线性层操作并返回结果
  memcpy(buf_data[0], r.const_data_ptr(),                // 将结果数据复制到第一个缓冲区
         r.element_size() * r.numel());
}

#ifndef _WIN32
void nnc_aten_quantized_add(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区数据的秩数组
    int64_t* buf_dims,                    // 参数：缓冲区数据的维度数组
    int64_t* buf_strides,                 // 参数：缓冲区数据的步长数组
    int8_t* buf_dtypes,                   // 参数：缓冲区数据的数据类型数组
    int64_t,                              // 参数：未命名的整数参数
    int64_t* extra_args) {                // 参数：额外参数数组的指针

  // TORCH_INTERNAL_ASSERT(tensors.size() == 3);
  
  // 从额外参数中提取第一个输入张量的量化参数
  const double a_qscale = ((double*)extra_args)[0];
  const int64_t a_qzero = extra_args[1];
  const c10::ScalarType a_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  
  // 从额外参数中提取第二个输入张量的量化参数
  const double b_qscale = ((double*)extra_args)[3];
  const int64_t b_qzero = extra_args[4];
  const c10::ScalarType b_qdtype = static_cast<c10::ScalarType>(extra_args[5]);
  
  // 构建输入张量
  auto tensors = constructTensors(
      bufs_num,                           // 使用缓冲区数量
      buf_data,                           // 使用缓冲区数据的指针数组
      buf_ranks,                          // 使用缓冲区数据的秩数组
      buf_dims,                           // 使用缓冲区数据的维度数组
      buf_strides,                        // 使用缓冲区数据的步长数组
      buf_dtypes,                         // 使用缓冲区数据的数据类型数组
      {{1u, {a_qscale, a_qzero, toQIntType(a_qdtype)}},  // 第一个张量的量化参数
       {2u, {b_qscale, b_qzero, toQIntType(b_qdtype)}}}); // 第二个张量的量化参数

  // 从额外参数中提取输出张量的量化参数
  const double out_qscale = ((double*)extra_args)[6];
  const int64_t out_qzero = extra_args[7];
  
  // 执行量化加法操作，并将结果复制回缓冲区
  // NOLINTNEXTLINE
  auto r = quantized_add(tensors[1], tensors[2], out_qscale, out_qzero);
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_quantized_mul(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区数据的秩数组
    int64_t* buf_dims,                    // 参数：缓冲区数据的维度数组
    int64_t* buf_strides,                 // 参数：缓冲区数据的步长数组
    int8_t* buf_dtypes,                   // 参数：缓冲区数据的数据类型数组
    int64_t,                              // 参数：未命名的整数参数
    int64_t* extra_args) {                // 参数：额外参数数组的指针

  // 从额外参数中提取第一个输入张量的量化参数
  const double a_qscale = ((double*)extra_args)[0];
  const int64_t a_qzero = extra_args[1];
  const c10::ScalarType a_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  
  // 从额外参数中提取第二个输入张量的量化参数
  const double b_qscale = ((double*)extra_args)[3];
  const int64_t b_qzero = extra_args[4];
  const c10::ScalarType b_qdtype = static_cast<c10::ScalarType>(extra_args[5]);
  
  // 构建输入张量
  auto tensors = constructTensors(
      bufs_num,                           // 使用缓冲区数量
      buf_data,                           // 使用缓冲区数据的指针数组
      buf_ranks,                          // 使用缓冲区数据的秩数组
      buf_dims,                           // 使用缓冲区数据的维度数组
      buf_strides,                        // 使用缓冲区数据的步长数组
      buf_dtypes,                         // 使用缓冲区数据的数据类型数组
      {{1u, {a_qscale, a_qzero, toQIntType(a_qdtype)}},  // 第一个张量的量化参数
       {2u, {b_qscale, b_qzero, toQIntType(b_qdtype)}}}); // 第二个张量的量化参数

  // 从额外参数中提取输出张量的量化参数
  const double out_qscale = ((double*)extra_args)[6];
  const int64_t out_qzero = extra_args[7];
  
  // 执行量化乘法操作，并将结果复制回缓冲区
  // NOLINTNEXTLINE
  auto r = quantized_mul(tensors[1], tensors[2], out_qscale, out_qzero);
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_quantized_mul_out(
    int64_t bufs_in_num,                  // 参数：输入缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区数据的秩数组
    int64_t* buf_dims,                    // 参数：缓冲区数据的维度数组
    int64_t* buf_strides,                 // 参数：缓冲区数据的步长数组
    int8_t* buf_dtypes,                   // 参数：缓冲区数据的数据类型数组
    int64_t,                              // 参数：未命名的整数参数
    int64_t* extra_args) {                // 参数：额外参数数组的指针
    // 定义额外参数的指针，指向一个 int64_t 类型的数组
    int64_t* extra_args) {
  // 定义输出缓冲区的数量为 1
  const size_t bufs_out_num = 1u;
  // 从额外参数中提取 a_qscale 值作为 double 类型
  const double a_qscale = ((double*)extra_args)[0];
  // 从额外参数中提取 a_qzero 值作为 int64_t 类型
  const int64_t a_qzero = extra_args[1];
  // 从额外参数中提取 a_qdtype 值作为 c10::ScalarType 枚举类型
  const c10::ScalarType a_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  // 从额外参数中提取 b_qscale 值作为 double 类型
  const double b_qscale = ((double*)extra_args)[3];
  // 从额外参数中提取 b_qzero 值作为 int64_t 类型
  const int64_t b_qzero = extra_args[4];
  // 从额外参数中提取 b_qdtype 值作为 c10::ScalarType 枚举类型
  const c10::ScalarType b_qdtype = static_cast<c10::ScalarType>(extra_args[5]);
  // 调用 constructTensors2 函数，构造输入和输出张量
  auto tensors = constructTensors2(
      bufs_in_num,  // 输入缓冲区的数量
      buf_data,     // 输入缓冲区数据数组
      buf_ranks,    // 输入缓冲区秩数组
      buf_dims,     // 输入缓冲区维度数组
      buf_strides,  // 输入缓冲区步幅数组
      buf_dtypes,   // 输入缓冲区数据类型数组
      {{1u, {a_qscale, a_qzero, toQIntType(a_qdtype)}},  // 第一个张量的量化参数
       {2u, {b_qscale, b_qzero, toQIntType(b_qdtype)}}}, // 第二个张量的量化参数
      1u);          // 输出张量的数量为 1
  // 从额外参数中提取 out_qscale 值作为 double 类型
  const double out_qscale = ((double*)extra_args)[6];
  // 从额外参数中提取 out_qzero 值作为 int64_t 类型
  const int64_t out_qzero = extra_args[7];
  // 调用 quantized_mul 函数进行量化乘法运算，得到结果 r
  // NOLINTNEXTLINE 是用来告诉 linter 忽略下一行的 lint 检查
  auto r = quantized_mul(tensors[1], tensors[2], out_qscale, out_qzero);
  // 将结果 r 的数据指针存入 buf_data 的第一个位置
  buf_data[0] = r.data_ptr();
  // 增加结果 r 的引用计数
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
  // 将结果 r 的智能指针存入 buf_data 的第 bufs_in_num + bufs_out_num 位置
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

void nnc_aten_quantized_mul_scalar(
    int64_t bufs_num,                             // 参数 bufs_num：表示缓冲区数量
    void** buf_data,                              // 参数 buf_data：指向缓冲区数据的指针数组
    int64_t* buf_ranks,                           // 参数 buf_ranks：指向缓冲区秩的指针
    int64_t* buf_dims,                            // 参数 buf_dims：指向缓冲区维度的指针
    int64_t* buf_strides,                         // 参数 buf_strides：指向缓冲区步长的指针
    int8_t* buf_dtypes,                           // 参数 buf_dtypes：指向缓冲区数据类型的指针
    int64_t,                                      // 参数（未命名）：未使用的整数参数
    int64_t* extra_args) {                        // 参数 extra_args：指向额外参数数组的指针
  const double x_qscale = ((double*)extra_args)[0];    // 获取量化比例因子 x_qscale
  const int64_t x_qzero = extra_args[1];               // 获取量化零点 x_qzero
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // 获取量化数据类型 x_qdtype
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});   // 构建量化后的张量数组
  const double scalar = ((double*)extra_args)[3];           // 获取标量值 scalar
  // NOLINTNEXTLINE
  auto r = quantized_mul_scalar(tensors[1], scalar);        // 对张量进行标量乘法运算
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());  // 将结果复制回缓冲区
}

void nnc_aten_quantized_mul_scalar_out(
    int64_t bufs_in_num,                          // 参数 bufs_in_num：输入缓冲区数量
    void** buf_data,                              // 参数 buf_data：指向缓冲区数据的指针数组
    int64_t* buf_ranks,                           // 参数 buf_ranks：指向缓冲区秩的指针
    int64_t* buf_dims,                            // 参数 buf_dims：指向缓冲区维度的指针
    int64_t* buf_strides,                         // 参数 buf_strides：指向缓冲区步长的指针
    int8_t* buf_dtypes,                           // 参数 buf_dtypes：指向缓冲区数据类型的指针
    int64_t,                                      // 参数（未命名）：未使用的整数参数
    int64_t* extra_args) {                        // 参数 extra_args：指向额外参数数组的指针
  const size_t bufs_out_num = 1u;                 // 输出缓冲区数量为 1
  const double x_qscale = ((double*)extra_args)[0];    // 获取量化比例因子 x_qscale
  const int64_t x_qzero = extra_args[1];               // 获取量化零点 x_qzero
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // 获取量化数据类型 x_qdtype
  auto tensors = constructTensors2(
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}},
      bufs_out_num);                              // 构建量化后的输入和输出张量数组
  const double scalar = ((double*)extra_args)[3];           // 获取标量值 scalar
  // NOLINTNEXTLINE
  auto r = quantized_mul_scalar(tensors[1], scalar);        // 对张量进行标量乘法运算
  buf_data[0] = r.data_ptr();                               // 设置输出缓冲区的数据指针
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());  // 增加输出张量的引用计数
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();  // 设置输出张量的指针
}

void nnc_aten_quantized_relu(
    int64_t bufs_num,                             // 参数 bufs_num：表示缓冲区数量
    void** buf_data,                              // 参数 buf_data：指向缓冲区数据的指针数组
    int64_t* buf_ranks,                           // 参数 buf_ranks：指向缓冲区秩的指针
    int64_t* buf_dims,                            // 参数 buf_dims：指向缓冲区维度的指针
    int64_t* buf_strides,                         // 参数 buf_strides：指向缓冲区步长的指针
    int8_t* buf_dtypes,                           // 参数 buf_dtypes：指向缓冲区数据类型的指针
    int64_t,                                      // 参数（未命名）：未使用的整数参数
    int64_t* extra_args) {                        // 参数 extra_args：指向额外参数数组的指针
  const double x_qscale = ((double*)extra_args)[0];    // 获取量化比例因子 x_qscale
  const int64_t x_qzero = extra_args[1];               // 获取量化零点 x_qzero
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // 获取量化数据类型 x_qdtype
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});   // 构建量化后的张量数组
  // NOLINTNEXTLINE
  auto r = at::relu(tensors[1]);                           // 对张量进行 ReLU 激活操作
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());  // 将结果复制回缓冲区
}

void nnc_aten_quantized_sigmoid(
    int64_t bufs_num,                             // 参数 bufs_num：表示缓冲区数量
    void** buf_data,                              // 参数 buf_data：指向缓冲区数据的指针数组
    int64_t* buf_ranks,                           // 参数 buf_ranks：指向缓冲区秩的指针
    int64_t* buf_dims,                            // 参数 buf_dims：指向缓冲区维度的指针
    int64_t* buf_strides,                         // 参数 buf_strides：指向缓冲区步长的指针
    int8_t* buf_dtypes,                           // 参数 buf_dtypes：指向缓冲区数据类型的指针
    int64_t,                                      // 参数（未命名）：未使用的整数参数
    int64_t* extra_args) {                        // 参数 extra_args：指向额外参数数组的指针
  const double x_qscale = ((double*)extra_args)[0];    // 获取量化比例因子 x_qscale
  const int64_t x_qzero = extra_args[1];               // 获取量化零点 x_qzero
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // 获取量化数据类型 x_qdtype
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});   // 构建量化后的张量数组
  // NOLINTNEXTLINE
  auto r = at::sigmoid(tensors[1]);                       // 对张量进行 sigmoid 激活操作
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());  // 将结果复制回缓冲区
}
    // 定义函数 `constructTensors`，根据传入的参数构造张量
    int64_t* extra_args) {
  // 从 `extra_args` 中获取量化参数 `x_qscale`，并转换为 double 类型
  const double x_qscale = ((double*)extra_args)[0];
  // 从 `extra_args` 中获取量化参数 `x_qzero`
  const int64_t x_qzero = extra_args[1];
  // 从 `extra_args` 中获取量化参数 `x_qdtype`，并转换为 PyTorch 的数据类型
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  // 调用 `constructTensors` 函数构造张量，其中包括量化参数信息
  auto tensors = constructTensors(
      bufs_num,                      // 张量的数量
      buf_data,                      // 数据缓冲区数组
      buf_ranks,                     // 张量秩数组
      buf_dims,                      // 张量维度数组
      buf_strides,                   // 张量步长数组
      buf_dtypes,                    // 张量数据类型数组
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}});  // 量化参数结构体数组

  // NOLINTNEXTLINE：忽略下一行的 lint 检查
  // 调用 PyTorch 的 sigmoid 函数，对 `tensors` 中的第二个张量进行 sigmoid 操作
  auto r = at::sigmoid(tensors[1]);
  // 将 sigmoid 操作的结果复制到 `buf_data` 数组的第一个缓冲区中
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_quantized_sigmoid_out(
    int64_t bufs_in_num,                          // 参数：输入缓冲区数量
    void** buf_data,                              // 参数：输入缓冲区数据
    int64_t* buf_ranks,                           // 参数：输入缓冲区秩（维度）
    int64_t* buf_dims,                            // 参数：输入缓冲区维度
    int64_t* buf_strides,                         // 参数：输入缓冲区步幅
    int8_t* buf_dtypes,                           // 参数：输入缓冲区数据类型
    int64_t,                                      // 参数：未使用，占位符
    int64_t* extra_args) {                        // 参数：额外参数数组

  // 从额外参数中获取量化参数
  const double x_qscale = ((double*)extra_args)[0];  // x的量化比例因子
  const int64_t x_qzero = extra_args[1];             // x的量化零点
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // x的量化数据类型
  const size_t bufs_out_num = 1u;

  // 构建输入和输出张量
  auto tensors = constructTensors2(
      bufs_in_num,                                // 输入缓冲区数量
      buf_data,                                   // 输入缓冲区数据指针
      buf_ranks,                                  // 输入缓冲区秩（维度）
      buf_dims,                                   // 输入缓冲区维度
      buf_strides,                                // 输入缓冲区步幅
      buf_dtypes,                                 // 输入缓冲区数据类型
      {{1u, {x_qscale, x_qzero, toQIntType(x_qdtype)}}},  // 量化参数元组
      bufs_out_num);                              // 输出缓冲区数量为1

  // 计算 sigmoid 函数
  auto r = at::sigmoid(tensors[1]);

  // 更新输出缓冲区数据指针
  buf_data[0] = r.data_ptr();

  // 增加引用计数，确保内存安全
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());

  // 将引用计数指针存入缓冲区
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

void nnc_aten_quantized_cat(
    int64_t bufs_num,                             // 参数：缓冲区数量
    void** buf_data,                              // 参数：缓冲区数据
    int64_t* buf_ranks,                           // 参数：缓冲区秩（维度）
    int64_t* buf_dims,                            // 参数：缓冲区维度
    int64_t* buf_strides,                         // 参数：缓冲区步幅
    int8_t* buf_dtypes,                           // 参数：缓冲区数据类型
    int64_t,                                      // 参数：未使用，占位符
    int64_t* extra_args) {                        // 参数：额外参数数组

  // 创建量化数据结构的向量
  std::vector<std::pair<size_t, QIData>> qdata;

  // 获取输出量化参数
  const auto in_bufs_num = bufs_num - 1;
  const double out_qscale = ((double*)extra_args)[3 * in_bufs_num + 1];  // 输出的量化比例因子
  const int64_t out_qzero = extra_args[3 * in_bufs_num + 2];             // 输出的量化零点
  qdata.emplace_back(
      0u,
      QIData{
          out_qscale, out_qzero, static_cast<c10::ScalarType>(extra_args[2])});  // 存储输出的量化数据信息

  // 循环处理输入的量化数据信息
  for (const size_t i : c10::irange(in_bufs_num)) {
    const double qscale = ((double*)extra_args)[3 * i + 0];        // 输入量化比例因子
    const int64_t qzero = extra_args[3 * i + 1];                   // 输入量化零点
    const c10::ScalarType qdtype =                                 // 输入量化数据类型
        static_cast<c10::ScalarType>(extra_args[3 * i + 2]);
    qdata.emplace_back(i + 1u, QIData{qscale, qzero, qdtype});      // 存储每个输入的量化数据信息
  }

  // 构建输入张量列表
  auto tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes, qdata);

  // 获取额外参数中的维度信息
  const int64_t dim = extra_args[3 * in_bufs_num + 0];

  // 构建输入张量列表
  auto qxs = c10::List<at::Tensor>(
      std::vector<at::Tensor>(tensors.begin() + 1, tensors.end()));

  // 执行量化的 concatenation 操作
  auto r = quantized_cat(qxs, dim, out_qscale, out_qzero);

  // 将结果复制到输出缓冲区
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_upsample_nearest2d(
    int64_t bufs_num,                              // 参数：缓冲区数量
    void** buf_data,                               // 参数：缓冲区数据
    int64_t* buf_ranks,                            // 参数：缓冲区秩（维度）
    int64_t* buf_dims,                             // 参数：缓冲区维度
    int64_t* buf_strides,                          // 参数：缓冲区步幅
    int8_t* buf_dtypes,                            // 参数：缓冲区数据类型
    int64_t,                                       // 参数：未使用，占位符
    int64_t* extra_args) {                         // 参数：额外参数数组

  // 获取输入量化参数
  const double x_qscale = ((double*)extra_args)[0];  // 输入的量化比例因子
  const int64_t x_qzero = extra_args[1];             // 输入的量化零点
  const int64_t x_qdtype = extra_args[2];            // 输入的量化数据类型
  const auto is_quantized = x_qdtype != -1;          // 检查是否量化

  // 声明可选的量化数据结构向量
  std::optional<std::vector<std::pair<size_t, QIData>>> qdata;

  // 如果输入是量化的，构建量化数据结构向量
  if (is_quantized) {
    qdata = {
        {1u,
         {x_qscale,
          x_qzero,
          at::toQIntType(static_cast<c10::ScalarType>(x_qdtype))}}};


    // 定义一个嵌套的初始化列表 qdata，包含一个无符号整数和一个嵌套的初始化列表
    // 嵌套的初始化列表包含 x_qscale、x_qzero 和将 x_qdtype 转换为对应量化整数类型的结果
  }



  auto tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes, qdata);


  // 调用 constructTensors 函数，传入多个参数用于构造张量
  // 这些参数包括 bufs_num（缓冲区数量）、buf_data（数据缓冲区数组）、buf_ranks（秩数组）、
  // buf_dims（维度数组）、buf_strides（步幅数组）、buf_dtypes（数据类型数组）和前面定义的 qdata
  auto x = tensors[1];


  // 从 tensors 数组中获取索引为 1 的张量，并赋值给变量 x



  int64_t output_size_h = extra_args[3];
  int64_t output_size_w = extra_args[4];
  double scale_factor_h = ((double*)extra_args)[5];
  double scale_factor_w = ((double*)extra_args)[6];


  // 从 extra_args 数组中读取输出高度和宽度的大小，并将它们分别赋值给 output_size_h 和 output_size_w
  // 将 extra_args 数组中索引为 5 和 6 的元素强制转换为 double 类型，并分别赋值给 scale_factor_h 和 scale_factor_w



  auto r = at::upsample_nearest2d(
      x,
      (output_size_h != -1)
          ? std::optional<at::IntArrayRef>({output_size_h, output_size_w})
          : c10::nullopt,
      (scale_factor_h != -1.f) ? std::optional<at::ArrayRef<double>>(
                                     {scale_factor_h, scale_factor_w})
                               : c10::nullopt);


  // 调用 at::upsample_nearest2d 函数进行最近邻插值上采样
  // 使用张量 x 作为输入
  // 根据 output_size_h 和 output_size_w 是否为 -1，决定是否传入包含输出大小的 optional<IntArrayRef>
  // 根据 scale_factor_h 是否为 -1.f，决定是否传入包含缩放因子的 optional<ArrayRef<double>>
  // 将结果赋值给变量 r



  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());


  // 使用 memcpy 函数将 r 的数据复制到 buf_data 数组的第一个元素所指向的内存位置
  // 复制的数据长度为 r 的元素大小乘以元素数量（r.numel()）
}

// 定义函数 nnc_aten_upsample_nearest2d_out，接受多个参数用于进行上采样操作
void nnc_aten_upsample_nearest2d_out(
    int64_t bufs_in_num,        // 输入缓冲区数量
    void** buf_data,            // 输入缓冲区数据的指针数组
    int64_t* buf_ranks,         // 输入缓冲区的秩数组
    int64_t* buf_dims,          // 输入缓冲区的维度数组
    int64_t* buf_strides,       // 输入缓冲区的步幅数组
    int8_t* buf_dtypes,         // 输入缓冲区的数据类型数组
    int64_t,                    // 未命名参数
    int64_t* extra_args) {      // 额外参数数组的指针
  const size_t bufs_out_num = 1u;  // 输出缓冲区数量为1
  // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
  const double x_qscale = ((double*)extra_args)[0];  // 提取量化比例因子
  const int64_t x_qzero = extra_args[1];             // 提取量化零点
  const int64_t x_qdtype = extra_args[2];            // 提取量化数据类型
  const auto is_quantized = x_qdtype != -1;          // 检查是否量化
  std::optional<std::vector<std::pair<size_t, QIData>>> qdata;
  if (is_quantized) {
    // 如果量化，创建量化数据结构
    qdata = {
        {1u,
         {x_qscale,
          x_qzero,
          at::toQIntType(static_cast<c10::ScalarType>(x_qdtype))}}};
  }
  // 构建输入和输出张量
  auto tensors = constructTensors2(
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      qdata,
      bufs_out_num);
  auto x = tensors[1];  // 获取第二个张量作为 x

  int64_t output_size_h = extra_args[3];     // 提取输出高度尺寸
  int64_t output_size_w = extra_args[4];     // 提取输出宽度尺寸
  double scale_factor_h = ((double*)extra_args)[5];  // 提取高度缩放因子
  double scale_factor_w = ((double*)extra_args)[6];  // 提取宽度缩放因子

  // 执行最近邻上采样操作
  auto r = at::upsample_nearest2d(
      x,
      (output_size_h != -1)
          ? std::optional<at::IntArrayRef>({output_size_h, output_size_w})  // 如果指定输出尺寸，使用之
          : c10::nullopt,
      (scale_factor_h != -1.f)
          ? std::optional<at::ArrayRef<double>>({scale_factor_h, scale_factor_w})  // 如果指定缩放因子，使用之
          : c10::nullopt);
  buf_data[0] = r.data_ptr();  // 将输出数据指针存入缓冲区
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());  // 增加引用计数
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();  // 存储输出数据指针
}

// 定义函数 nnc_aten_quantize_per_tensor，用于对张量进行量化操作
void nnc_aten_quantize_per_tensor(
    int64_t bufs_num,           // 缓冲区数量
    void** buf_data,            // 缓冲区数据的指针数组
    int64_t* buf_ranks,         // 缓冲区的秩数组
    int64_t* buf_dims,          // 缓冲区的维度数组
    int64_t* buf_strides,       // 缓冲区的步幅数组
    int8_t* buf_dtypes,         // 缓冲区的数据类型数组
    int64_t,                    // 未命名参数
    int64_t* extra_args) {      // 额外参数数组的指针
  auto tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);  // 构建输入张量数组
  // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
  at::Tensor x = tensors[1];   // 获取第二个张量作为 x
  const double qscale = ((double*)extra_args)[0];  // 提取量化比例因子
  const int64_t qzero = extra_args[1];             // 提取量化零点
  const c10::ScalarType qdtype = static_cast<c10::ScalarType>(extra_args[2]);  // 提取量化数据类型
  auto r = at::quantize_per_tensor(x, qscale, qzero, qdtype);  // 执行张量的量化操作
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());  // 将量化结果复制到输出缓冲区
}

// 定义函数 nnc_aten_quantize_per_tensor_out，用于对张量进行量化操作，并将结果输出到指定缓冲区
void nnc_aten_quantize_per_tensor_out(
    int64_t bufs_in_num,        // 输入缓冲区数量
    void** buf_data,            // 输入缓冲区数据的指针数组
    int64_t* buf_ranks,         // 输入缓冲区的秩数组
    int64_t* buf_dims,          // 输入缓冲区的维度数组
    int64_t* buf_strides,       // 输入缓冲区的步幅数组
    int8_t* buf_dtypes,         // 输入缓冲区的数据类型数组
    int64_t,                    // 未命名参数
    int64_t* extra_args) {      // 额外参数数组的指针
    // 构造输出张量数量常量为1
    const size_t bufs_out_num = 1u;
    // 使用给定参数构造张量数组
    auto tensors = constructTensors2(
        bufs_in_num,        // 输入缓冲区数量
        buf_data,           // 缓冲区数据数组
        buf_ranks,          // 缓冲区秩数组
        buf_dims,           // 缓冲区维度数组
        buf_strides,        // 缓冲区步长数组
        buf_dtypes,         // 缓冲区数据类型数组
        c10::nullopt,       // 不使用任何额外选项
        bufs_out_num        // 输出缓冲区数量
    );
    // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
    // 从张量数组中取出第二个张量作为 at::Tensor x
    at::Tensor x = tensors[1];
    // 从额外参数中获取量化参数 qscale 和 qzero
    const double qscale = ((double*)extra_args)[0];
    const int64_t qzero = extra_args[1];
    // 从额外参数中获取量化数据类型 qdtype
    const c10::ScalarType qdtype = static_cast<c10::ScalarType>(extra_args[2]);
    // 对张量 x 进行量化操作，使用给定的 qscale, qzero 和 qdtype
    auto r = at::quantize_per_tensor(x, qscale, qzero, qdtype);
    // 将量化后的数据指针存入 buf_data 的第一个位置
    buf_data[0] = r.data_ptr();
    // 增加量化结果的引用计数
    c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
    // 将量化结果的内部指针存入 buf_data 的第 bufs_in_num + bufs_out_num 个位置
    buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

void nnc_aten_dequantize(
    int64_t bufs_num,                            // 参数：缓冲区数量
    void** buf_data,                             // 参数：缓冲区数据数组
    int64_t* buf_ranks,                          // 参数：缓冲区秩数组
    int64_t* buf_dims,                           // 参数：缓冲区维度数组
    int64_t* buf_strides,                        // 参数：缓冲区步长数组
    int8_t* buf_dtypes,                          // 参数：缓冲区数据类型数组
    int64_t,                                     // 无用参数，未命名
    int64_t* extra_args) {                       // 参数：额外参数数组

  // 从额外参数中获取量化参数
  const double qscale = ((double*)extra_args)[0];  // 量化比例
  const int64_t qzero = extra_args[1];             // 量化零点
  const int64_t qdtype = extra_args[2];            // 量化数据类型
  // 构造张量对象数组，使用构造函数传递量化参数和类型
  auto tensors = constructTensors(
      bufs_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u,
        {qscale, qzero, toQIntType(static_cast<c10::ScalarType>(qdtype))}}});
  // 调用 ATen 库的 dequantize 函数，将张量对象的第二个元素进行反量化
  // NOLINTNEXTLINE: 禁止 lint 检查下一行的规则
  auto r = at::dequantize(tensors[1]);
  // 将结果张量的数据拷贝回缓冲区的第一个数据指针处
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_dequantize_out(
    int64_t bufs_in_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t,                                     // 无用参数，未命名
    int64_t* extra_args) {                       // 参数：额外参数数组

  const size_t bufs_out_num = 1u;                 // 输出缓冲区数量为1
  // 从额外参数中获取量化参数
  const double qscale = ((double*)extra_args)[0];  // 量化比例
  const int64_t qzero = extra_args[1];             // 量化零点
  const int64_t qdtype = extra_args[2];            // 量化数据类型
  // 构造张量对象数组，使用构造函数传递量化参数和类型，并指定输出缓冲区数量
  auto tensors = constructTensors2(
      bufs_in_num,
      buf_data,
      buf_ranks,
      buf_dims,
      buf_strides,
      buf_dtypes,
      {{1u, {qscale, qzero, toQIntType(static_cast<c10::ScalarType>(qdtype))}}},
      bufs_out_num);
  // 调用 ATen 库的 dequantize 函数，将张量对象的第二个元素进行反量化
  auto r = at::dequantize(tensors[1]);
  // 将结果张量的数据指针赋值给输出缓冲区的第一个元素
  buf_data[0] = r.data_ptr();
  // 增加结果张量的内部引用计数，避免其被提前释放
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
  // 将结果张量的内部引用指针赋值给输出缓冲区的第 bufs_in_num + bufs_out_num 个元素
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
}

void nnc_aten_conv1d(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  // 构造张量对象数组，使用构造函数传递参数
  auto tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);

  at::Tensor& r = tensors[0];                     // 结果张量的引用
  const at::Tensor& x = tensors[1];               // 输入张量的常量引用
  const at::Tensor& w = tensors[2];               // 权重张量的常量引用

  if (args_num > 0) {
    // 如果提供了额外参数，则检查是否也存在偏置张量
    TORCH_INTERNAL_ASSERT(args_num == 4 && bufs_num == 4);
    const at::Tensor& b = tensors[3];             // 偏置张量的常量引用

    int64_t stride = extra_args[0];               // 卷积步长
    int64_t padding = extra_args[1];              // 卷积填充
    int64_t dilation = extra_args[2];             // 卷积扩展
    int64_t groups = extra_args[3];               // 卷积分组

    try {
      // 调用 ATen 库的一维卷积函数，根据提供的参数计算卷积结果
      r = at::conv1d(x, w, b, {stride}, {padding}, {dilation}, groups);
    } catch (...) {
      // 异常处理，捕获所有异常
    }
  } else {
    try {
      // 调用 ATen 库的一维卷积函数，仅使用输入和权重张量进行计算
      r = at::conv1d(x, w);
    } catch (...) {
      // 异常处理，捕获所有异常
    }
  }

  // 将计算结果张量的数据拷贝回缓冲区的第一个数据指针处
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_conv1d_out(
    int64_t bufs_in_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
    // 定义一个函数，该函数用于执行一维卷积操作，并更新输入和输出缓冲区
    void forward_conv1d(
        const size_t bufs_in_num,
        void** buf_data,
        const int64_t* buf_ranks,
        const int64_t* buf_dims,
        const int64_t* buf_strides,
        const int64_t* buf_dtypes,
        const int64_t args_num,
        int64_t* extra_args) {
      
      // 输出张量数量为1
      const size_t bufs_out_num = 1u;
      
      // 使用给定参数构造张量
      auto tensors = constructTensors2(
          bufs_in_num,
          buf_data,
          buf_ranks,
          buf_dims,
          buf_strides,
          buf_dtypes,
          c10::nullopt,
          bufs_out_num);
    
      // 初始化结果张量
      at::Tensor r;
      const at::Tensor& x = tensors[1];  // 获取输入张量 x
      const at::Tensor& w = tensors[2];  // 获取权重张量 w
    
      // 如果提供了额外的参数
      if (args_num > 0) {
        // 检查额外的参数数量和输入缓冲区数量
        TORCH_INTERNAL_ASSERT(args_num == 4 && bufs_in_num == 3);
    
        // 获取偏置张量 b
        const at::Tensor& b = tensors[3];
    
        // 获取额外的卷积参数：步长、填充、扩展和分组
        int64_t stride = extra_args[0];
        int64_t padding = extra_args[1];
        int64_t dilation = extra_args[2];
        int64_t groups = extra_args[3];
    
        // 执行带有偏置的一维卷积操作，并捕获任何异常
        try {
          r = at::conv1d(x, w, b, {stride}, {padding}, {dilation}, groups);
        } catch (...) {
          // 捕获可能发生的异常，不做处理
        }
      } else {
        // 执行不带偏置的一维卷积操作，并捕获任何异常
        try {
          r = at::conv1d(x, w);
        } catch (...) {
          // 捕获可能发生的异常，不做处理
        }
      }
    
      // 更新输入缓冲区中的数据指针为结果张量的数据指针
      buf_data[0] = r.data_ptr();
    
      // 增加结果张量的引用计数，确保在缓冲区维持有效
      c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());
    
      // 将结果张量的指针存储在输出缓冲区中
      buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();
    }
void nnc_aten_adaptive_avg_pool2d(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区张量的秩（维度数量）数组
    int64_t* buf_dims,                    // 参数：缓冲区张量各维度大小的数组
    int64_t* buf_strides,                 // 参数：缓冲区张量各维度步长的数组
    int8_t* buf_dtypes,                   // 参数：缓冲区张量的数据类型数组
    int64_t args_num,                     // 参数：额外参数的数量
    int64_t* extra_args) {                // 参数：额外参数数组
  auto tensors = constructTensors(         // 构造张量数组对象，从缓冲区数据和描述中获取张量
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);

  at::Tensor& r = tensors[0];             // 获取结果张量的引用
  const at::Tensor& x = tensors[1];       // 获取输入张量的常量引用
  int64_t H = extra_args[0];              // 从额外参数中获取高度维度
  int64_t W = H;                          // 初始化宽度维度为高度的值
  if (args_num > 1) {                     // 如果有第二个额外参数
    W = extra_args[1];                    // 则将宽度维度设置为第二个额外参数的值
  }
  try {
    r = at::adaptive_avg_pool2d(x, {H, W});   // 执行自适应平均池化操作，并将结果存入 r 中
  } catch (...) {                         // 捕获可能发生的任何异常
  }
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());   // 将结果张量的数据复制回缓冲区
}

void nnc_aten_mean(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区张量的秩（维度数量）数组
    int64_t* buf_dims,                    // 参数：缓冲区张量各维度大小的数组
    int64_t* buf_strides,                 // 参数：缓冲区张量各维度步长的数组
    int8_t* buf_dtypes,                   // 参数：缓冲区张量的数据类型数组
    int64_t args_num,                     // 参数：额外参数的数量
    int64_t* extra_args) {                // 参数：额外参数数组
  auto tensors = constructTensors(         // 构造张量数组对象，从缓冲区数据和描述中获取张量
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);

  at::Tensor& r = tensors[0];             // 获取结果张量的引用
  const at::Tensor& x = tensors[1];       // 获取输入张量的常量引用
  std::vector<int64_t> mean_dims(args_num - 1);  // 创建维度大小的向量
  bool keepdim = (bool)extra_args[args_num - 1];  // 获取保持维度标志位
  if (args_num > 1) {                     // 如果有额外的维度参数
    memcpy(mean_dims.data(), extra_args, sizeof(int64_t) * (args_num - 1));  // 将额外的维度参数复制到 mean_dims
  }
  try {
    at::mean_out(r, x, mean_dims, keepdim);  // 执行均值池化操作，并将结果存入 r 中
  } catch (...) {                         // 捕获可能发生的任何异常
  }
}

void nnc_aten_max_red(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区张量的秩（维度数量）数组
    int64_t* buf_dims,                    // 参数：缓冲区张量各维度大小的数组
    int64_t* buf_strides,                 // 参数：缓冲区张量各维度步长的数组
    int8_t* buf_dtypes,                   // 参数：缓冲区张量的数据类型数组
    int64_t args_num,                     // 参数：额外参数的数量
    int64_t* extra_args) {                // 参数：额外参数数组
  auto tensors = constructTensors(         // 构造张量数组对象，从缓冲区数据和描述中获取张量
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);

  at::Tensor& r = tensors[0];             // 获取结果张量的引用
  const at::Tensor& x = tensors[1];       // 获取输入张量的常量引用
  int64_t max_dim = extra_args[0];        // 获取最大值约简的维度
  bool keep_dim = extra_args[1];          // 获取是否保持维度的标志位
  try {
    r = std::get<0>(at::max(x, max_dim, keep_dim));  // 执行最大值约简操作，并将结果存入 r 中
  } catch (...) {                         // 捕获可能发生的任何异常
  }
  memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());   // 将结果张量的数据复制回缓冲区
}

void nnc_aten_max_red_out(
    int64_t bufs_in_num,                  // 参数：输入缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓冲区张量的秩（维度数量）数组
    int64_t* buf_dims,                    // 参数：缓冲区张量各维度大小的数组
    int64_t* buf_strides,                 // 参数：缓冲区张量各维度步长的数组
    int8_t* buf_dtypes,                   // 参数：缓冲区张量的数据类型数组
    int64_t,                              // 参数：未使用的参数数量
    int64_t* extra_args) {                // 参数：额外参数数组
  size_t bufs_out_num = 1u;               // 输出缓冲区数量为 1
  auto tensors = constructTensors2(        // 构造双重张量数组对象，从输入缓冲区数据和描述中获取张量
      bufs_in_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);

  at::Tensor r;                           // 创建结果张量对象
  const at::Tensor& x = tensors[1];       // 获取输入张量的常量引用
  int64_t max_dim = extra_args[0];        // 获取最大值约简的维度
  bool keep_dim = extra_args[1];          // 获取是否保持维度的标志位
  try {
    r = std::get<0>(at::max(x, max_dim, keep_dim));  // 执行最大值约简操作，并将结果存入 r 中
  } catch (...) {                         // 捕获可能发生的任何异常
  }
  buf_data[0] = r.data_ptr();             // 将结果张量的数据指针存入输出缓冲区
  c10::raw::intrusive_ptr::incref(r.getIntrusivePtr().get());  // 增加结果张量的引用计数
  buf_data[bufs_in_num + bufs_out_num] = r.getIntrusivePtr().get();  // 将结果张量的指针存入输出缓冲区
}

void nnc_aten_addmm(
    int64_t bufs_num,                     // 参数：缓冲区数量
    void** buf_data,                      // 参数：缓冲区数据的指针数组
    int64_t* buf_ranks,                   // 参数：缓
    // 使用 int64_t 类型的指针参数 extra_args，作为额外参数传入函数
    int64_t* extra_args) {
      // 调用 constructTensors 函数构造张量对象，返回一个张量数组 tensors
      auto tensors = constructTensors(
          bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
    
      // 取得 tensors 数组中的各个张量对象的引用
      at::Tensor& r = tensors[0];      // 结果张量 r
      const at::Tensor& x = tensors[1];  // 输入张量 x
      const at::Tensor& y = tensors[2];  // 输入张量 y
      const at::Tensor& z = tensors[3];  // 输入张量 z
      
      // TODO: 处理其他 alpha 和 beta 的数据类型，例如 alpha=0.6，beta=0.2
      int64_t alpha = extra_args[0], beta = extra_args[1];  // 设置 alpha 和 beta 的值为 extra_args 数组的第一个和第二个元素
    
      // 尝试执行矩阵乘法和加法操作，并将结果保存在 r 中
      try {
        at::addmm_out(r, x, y, z, alpha, beta);  // 使用 alpha 和 beta 进行加权和
      } catch (...) {
        // 捕获任何可能的异常，但不做具体处理
      }
    }
// 结束函数定义
}

// 实现一个用于求解三角矩阵的函数，仅返回第一个输出，第二个输出是其中一个输入的复制
void nnc_aten_triangular_solve(
    int64_t bufs_num,         // 缓冲区数量
    void** buf_data,          // 缓冲区数据
    int64_t* buf_ranks,       // 缓冲区秩
    int64_t* buf_dims,        // 缓冲区维度
    int64_t* buf_strides,     // 缓冲区步长
    int8_t* buf_dtypes,       // 缓冲区数据类型
    int64_t args_num,         // 参数数量
    int64_t* extra_args) {    // 额外参数数组
  auto tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);  // 构造张量
  at::Tensor& r = tensors[0];     // 引用第一个张量
  at::Tensor r2 = tensors[2].clone();  // 复制第三个张量
  const at::Tensor& input = tensors[1];  // 引用第二个张量作为输入
  const at::Tensor& A = tensors[2];      // 引用第三个张量作为矩阵A
  try {
    at::triangular_solve_out(
        r, r2, input, A, extra_args[0], extra_args[2], extra_args[3]);  // 调用三角矩阵求解函数
  } catch (...) {
  }
}

#if AT_MKLDNN_ENABLED()

// 使用 MKL-DNN 加速的预打包卷积运行函数
void nnc_mkldnn_prepacked_conv_run(
    int64_t bufs_num,         // 缓冲区数量
    void** buf_data,          // 缓冲区数据
    int64_t* buf_ranks,       // 缓冲区秩
    int64_t* buf_dims,        // 缓冲区维度
    int64_t* buf_strides,     // 缓冲区步长
    int8_t* buf_dtypes,       // 缓冲区数据类型
    int64_t args_num,         // 参数数量
    int64_t* extra_args) {    // 额外参数数组
  using namespace at::native::mkldnn;  // 使用 MKL-DNN 命名空间

  auto tensors = constructTensors(
      bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);  // 构造张量，忽略第一个缓冲区

  const at::Tensor& x = tensors[1];  // 引用第二个张量
  auto context = reinterpret_cast<ConvOpContext*>(buf_data[2]);  // 解释第三个缓冲区数据为卷积操作上下文
  context->run(x, buf_data[0]);  // 运行卷积操作
}

#endif // AT_MKLDNN_ENABLED()

#ifdef USE_XNNPACK

// 使用 XNNPACK 加速的预打包线性clamp运行函数
void nnc_prepacked_linear_clamp_run(
    int64_t bufs_num,         // 缓冲区数量
    void** buf_data,          // 缓冲区数据
    int64_t* buf_ranks,       // 缓冲区秩
    int64_t* buf_dims,        // 缓冲区维度
    int64_t* buf_strides,     // 缓冲区步长
    int8_t* buf_dtypes,       // 缓冲区数据类型
    int64_t args_num,         // 参数数量
    int64_t* extra_args) {    // 额外参数数组
  using namespace at::native::xnnpack;  // 使用 XNNPACK 命名空间

  auto tensors = constructTensors(
      bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);  // 构造张量，忽略第一个缓冲区

  const at::Tensor& x = tensors[1];  // 引用第二个张量
  auto context = reinterpret_cast<LinearOpContext*>(buf_data[2]);  // 解释第三个缓冲区数据为线性操作上下文
  at::Tensor output = context->run(x);  // 运行线性操作
  memcpy(
      buf_data[0],  // 目标缓冲区
      output.const_data_ptr(),  // 输出张量数据指针
      output.element_size() * output.numel());  // 复制输出张量数据到目标缓冲区
}

// 使用 XNNPACK 加速的预打包二维卷积clamp运行函数
void nnc_prepacked_conv2d_clamp_run(
    int64_t bufs_num,         // 缓冲区数量
    void** buf_data,          // 缓冲区数据
    int64_t* buf_ranks,       // 缓冲区秩
    int64_t* buf_dims,        // 缓冲区维度
    int64_t* buf_strides,     // 缓冲区步长
    int8_t* buf_dtypes,       // 缓冲区数据类型
    int64_t args_num,         // 参数数量
    int64_t* extra_args) {    // 额外参数数组
  using namespace at::native::xnnpack;  // 使用 XNNPACK 命名空间

  auto tensors = constructTensors(
      bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);  // 构造张量，忽略第一个缓冲区

  const at::Tensor& x = tensors[1];  // 引用第二个张量
  auto context = reinterpret_cast<Conv2dOpContext*>(buf_data[2]);  // 解释第三个缓冲区数据为二维卷积操作上下文
  at::Tensor output = context->run(x);  // 运行二维卷积操作
  memcpy(
      buf_data[0],  // 目标缓冲区
      output.const_data_ptr(),  // 输出张量数据指针
      output.element_size() * output.numel());  // 复制输出张量数据到目标缓冲区
}

#endif // USE_XNNPACK

// ATen嵌入操作函数
void nnc_aten_embedding(
    int64_t bufs_num,         // 缓冲区数量
    void** buf_data,          // 缓冲区数据
    int64_t* buf_ranks,       // 缓冲区秩
    int64_t* buf_dims,        // 缓冲区维度
    int64_t* buf_strides,     // 缓冲区步长
    int8_t* buf_dtypes,       // 缓冲区数据类型
    int64_t args_num,         // 参数数量
    int64_t* extra_args) {    // 额外参数数组
    // 构造张量对象，使用传入的缓冲区数据、秩、维度、步幅和数据类型
    auto tensors = constructTensors(
        bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
    
    // 获取返回结果张量的引用
    at::Tensor& r = tensors[0];
    // 获取权重张量的常量引用
    const at::Tensor& weight = tensors[1];
    // 获取索引张量的常量引用
    const at::Tensor& indices = tensors[2];
    
    // 尝试进行嵌入操作，将结果存储在张量 r 中
    try {
      r = at::embedding(weight, indices);
    } catch (...) {
      // 捕获任何可能抛出的异常，但未进行具体处理
    }
    
    // TODO: 由于 at::embedding 不支持输出参数，需要复制输出数据
    // 因为 NNC 的外部调用不支持分配，所以使用 memcpy 将结果复制到 buf_data 的第一个缓冲区中
    memcpy(buf_data[0], r.const_data_ptr(), r.element_size() * r.numel());
#ifndef C10_MOBILE
// 如果不是在移动设备上编译，则注册以下静态变量为NNC外部函数
const static RegisterNNCExternalFunction nnc_conv2d(
    "nnc_aten_conv2d",
    nnc_aten_conv2d);

const static RegisterNNCExternalFunction nnc_quantized_conv1d(
    "nnc_aten_quantized_conv1d",
    nnc_aten_quantized_conv1d);
const static RegisterNNCExternalFunction nnc_quantized_conv1d_out(
    "nnc_aten_quantized_conv1d_out",
    nnc_aten_quantized_conv1d_out);
const static RegisterNNCExternalFunction nnc_quantized_conv2d(
    "nnc_aten_quantized_conv2d",
    nnc_aten_quantized_conv2d);
const static RegisterNNCExternalFunction nnc_quantized_conv2d_out(
    "nnc_aten_quantized_conv2d_out",
    nnc_aten_quantized_conv2d_out);
const static RegisterNNCExternalFunction nnc_quantized_conv2d_relu(
    "nnc_aten_quantized_conv2d_relu",
    nnc_aten_quantized_conv2d_relu);
const static RegisterNNCExternalFunction nnc_quantized_conv2d_relu_out(
    "nnc_aten_quantized_conv2d_relu_out",
    nnc_aten_quantized_conv2d_relu_out);
const static RegisterNNCExternalFunction nnc_quantized_linear(
    "nnc_aten_quantized_linear",
    nnc_aten_quantized_linear);
const static RegisterNNCExternalFunction nnc_quantized_linear_out(
    "nnc_aten_quantized_linear_out",
    nnc_aten_quantized_linear_out);
#ifndef _WIN32
// 如果不是在Windows平台上编译，则注册以下静态变量为NNC外部函数
const static RegisterNNCExternalFunction nnc_quantized_add(
    "nnc_aten_quantized_add",
    nnc_aten_quantized_add);
const static RegisterNNCExternalFunction nnc_quantized_mul(
    "nnc_aten_quantized_mul",
    nnc_aten_quantized_mul);
const static RegisterNNCExternalFunction nnc_quantized_mul_out(
    "nnc_aten_quantized_mul_out",
    nnc_aten_quantized_mul_out);
const static RegisterNNCExternalFunction nnc_quantized_mul_scalar(
    "nnc_aten_quantized_mul_scalar",
    nnc_aten_quantized_mul_scalar);
const static RegisterNNCExternalFunction nnc_quantized_mul_scalar_out(
    "nnc_aten_quantized_mul_scalar_out",
    nnc_aten_quantized_mul_scalar_out);
const static RegisterNNCExternalFunction nnc_quantized_sigmoid(
    "nnc_aten_quantized_sigmoid",
    nnc_aten_quantized_sigmoid);
const static RegisterNNCExternalFunction nnc_quantized_sigmoid_out(
    "nnc_aten_quantized_sigmoid_out",
    nnc_aten_quantized_sigmoid_out);
const static RegisterNNCExternalFunction nnc_quantized_cat(
    "nnc_aten_quantized_cat",
    nnc_aten_quantized_cat);
const static RegisterNNCExternalFunction nnc_quantized_relu(
    "nnc_aten_quantized_relu",
    nnc_aten_quantized_relu);
#endif // _WIN32
// 注册以下静态变量为NNC外部函数，用于量化张量操作
const static RegisterNNCExternalFunction nnc_quantize_per_tensor(
    "nnc_aten_quantize_per_tensor",
    nnc_aten_quantize_per_tensor);
const static RegisterNNCExternalFunction nnc_quantize_per_tensor_out(
    "nnc_aten_quantize_per_tensor_out",
    nnc_aten_quantize_per_tensor_out);
const static RegisterNNCExternalFunction nnc_dequantize(
    "nnc_aten_dequantize",
    nnc_aten_dequantize);
const static RegisterNNCExternalFunction nnc_dequantize_out(
    "nnc_aten_dequantize_out",
    nnc_aten_dequantize_out);
#endif
const static RegisterNNCExternalFunction nnc_upsample_nearest2d(
    "nnc_aten_upsample_nearest2d",
    nnc_aten_upsample_nearest2d);



// 注册一个外部函数 nnc_aten_upsample_nearest2d 到 nnc_upsample_nearest2d
const static RegisterNNCExternalFunction nnc_upsample_nearest2d(
    "nnc_aten_upsample_nearest2d",
    nnc_aten_upsample_nearest2d);



const static RegisterNNCExternalFunction nnc_upsample_nearest2d_out(
    "nnc_aten_upsample_nearest2d_out",
    nnc_aten_upsample_nearest2d_out);



// 注册一个外部函数 nnc_aten_upsample_nearest2d_out 到 nnc_upsample_nearest2d_out
const static RegisterNNCExternalFunction nnc_upsample_nearest2d_out(
    "nnc_aten_upsample_nearest2d_out",
    nnc_aten_upsample_nearest2d_out);



const static RegisterNNCExternalFunction nnc_conv1d(
    "nnc_aten_conv1d",
    nnc_aten_conv1d);



// 注册一个外部函数 nnc_aten_conv1d 到 nnc_conv1d
const static RegisterNNCExternalFunction nnc_conv1d(
    "nnc_aten_conv1d",
    nnc_aten_conv1d);



const static RegisterNNCExternalFunction nnc_conv1d_out(
    "nnc_aten_conv1d_out",
    nnc_aten_conv1d_out);



// 注册一个外部函数 nnc_aten_conv1d_out 到 nnc_conv1d_out
const static RegisterNNCExternalFunction nnc_conv1d_out(
    "nnc_aten_conv1d_out",
    nnc_aten_conv1d_out);



const static RegisterNNCExternalFunction nnc_adaptive_avg_pool2d(
    "nnc_aten_adaptive_avg_pool2d",
    nnc_aten_adaptive_avg_pool2d);



// 注册一个外部函数 nnc_aten_adaptive_avg_pool2d 到 nnc_adaptive_avg_pool2d
const static RegisterNNCExternalFunction nnc_adaptive_avg_pool2d(
    "nnc_aten_adaptive_avg_pool2d",
    nnc_aten_adaptive_avg_pool2d);



const static RegisterNNCExternalFunction nnc_mean(
    "nnc_aten_mean",
    nnc_aten_mean);



// 注册一个外部函数 nnc_aten_mean 到 nnc_mean
const static RegisterNNCExternalFunction nnc_mean(
    "nnc_aten_mean",
    nnc_aten_mean);



const static RegisterNNCExternalFunction nnc_max_red(
    "nnc_aten_max_red",
    nnc_aten_max_red);



// 注册一个外部函数 nnc_aten_max_red 到 nnc_max_red
const static RegisterNNCExternalFunction nnc_max_red(
    "nnc_aten_max_red",
    nnc_aten_max_red);



const static RegisterNNCExternalFunction nnc_max_red_out(
    "nnc_aten_max_red_out",
    nnc_aten_max_red_out);



// 注册一个外部函数 nnc_aten_max_red_out 到 nnc_max_red_out
const static RegisterNNCExternalFunction nnc_max_red_out(
    "nnc_aten_max_red_out",
    nnc_aten_max_red_out);



const static RegisterNNCExternalFunction nnc_addmm(
    "nnc_aten_addmm",
    nnc_aten_addmm);



// 注册一个外部函数 nnc_aten_addmm 到 nnc_addmm
const static RegisterNNCExternalFunction nnc_addmm(
    "nnc_aten_addmm",
    nnc_aten_addmm);



const static RegisterNNCExternalFunction nnc_triangular_solve(
    "nnc_aten_triangular_solve",
    nnc_aten_triangular_solve);



// 注册一个外部函数 nnc_aten_triangular_solve 到 nnc_triangular_solve
const static RegisterNNCExternalFunction nnc_triangular_solve(
    "nnc_aten_triangular_solve",
    nnc_aten_triangular_solve);



const static RegisterNNCExternalFunction nnc_embedding(
    "nnc_aten_embedding",
    nnc_aten_embedding);



// 注册一个外部函数 nnc_aten_embedding 到 nnc_embedding
const static RegisterNNCExternalFunction nnc_embedding(
    "nnc_aten_embedding",
    nnc_aten_embedding);



#if AT_MKLDNN_ENABLED()
const static RegisterNNCExternalFunction reg_nnc_mkldnn_prepacked_conv_run(
    "nnc_mkldnn_prepacked_conv_run",
    nnc_mkldnn_prepacked_conv_run);
#endif // AT_MKLDNN_ENABLED()



// 如果 AT_MKLDNN_ENABLED 宏被定义，则注册一个外部函数 nnc_mkldnn_prepacked_conv_run 到 reg_nnc_mkldnn_prepacked_conv_run
#if AT_MKLDNN_ENABLED()
const static RegisterNNCExternalFunction reg_nnc_mkldnn_prepacked_conv_run(
    "nnc_mkldnn_prepacked_conv_run",
    nnc_mkldnn_prepacked_conv_run);
#endif // AT_MKLDNN_ENABLED()



#ifdef USE_XNNPACK
const static RegisterNNCExternalFunction reg_nnc_prepacked_linear_clamp_run(
    "nnc_prepacked_linear_clamp_run",
    nnc_prepacked_linear_clamp_run);
const static RegisterNNCExternalFunction reg_nnc_prepacked_conv2d_clamp_run(
    "nnc_prepacked_conv2d_clamp_run",
    nnc_prepacked_conv2d_clamp_run);
#endif // USE_XNNPACK



// 如果 USE_XNNPACK 宏被定义，则分别注册外部函数 nnc_prepacked_linear_clamp_run 和 nnc_prepacked_conv2d_clamp_run
#ifdef USE_XNNPACK
const static RegisterNNCExternalFunction reg_nnc_prepacked_linear_clamp_run(
    "nnc_prepacked_linear_clamp_run",
    nnc_prepacked_linear_clamp_run);
const static RegisterNNCExternalFunction reg_nnc_prepacked_conv2d_clamp_run(
    "nnc_prepacked_conv2d_clamp_run",
    nnc_prepacked_conv2d_clamp_run);
#endif // USE_XNNPACK



#endif // C10_MOBILE

#ifdef C10_MOBILE
} // extern "C"
#endif



// 如果 C10_MOBILE 宏被定义，则结束 extern "C" 块
#ifdef C10_MOBILE
} // extern "C"
#endif



} // namespace torch::jit::tensorexpr



// 结束命名空间 torch::jit::tensorexpr
} // namespace torch::jit::tensorexpr
```