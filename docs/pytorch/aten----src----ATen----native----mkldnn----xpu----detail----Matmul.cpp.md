# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Matmul.cpp`

```py
// 引入C10的XPU函数头文件
#include <c10/xpu/XPUFunctions.h>

// 引入ATen库的头文件
#include <ATen/ATen.h>
// 引入ATen库的记录函数功能头文件
#include <ATen/record_function.h>

// 引入自定义属性和工具函数的头文件
#include <Attr.h>
#include <Utils.h>

// 引入oneDNN库的头文件
#include <oneapi/dnnl/dnnl.hpp>

// 定义命名空间at::native::onednn
namespace at::native::onednn {

// 实现矩阵乘法函数matmul
sycl::event matmul(
    at::Tensor& result,                  // 输出结果张量
    const at::Tensor& mat1,              // 输入矩阵1
    const at::Tensor& mat2,              // 输入矩阵2
    const at::Tensor& b_raw,             // 偏置张量
    bool m2_trans,                       // 是否转置输入矩阵2
    Attr attr,                           // 自定义属性
    const std::vector<sycl::event>& deps // 依赖事件向量
) {
  // 获取结果张量的维度
  int64_t dims = result.dim();
  
  // 检查结果张量的维度是否为2或3
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  
  // 检查输入矩阵的维度与结果张量的维度是否相同
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  
  // 检查结果张量是否已定义
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  // 获取当前设备并创建GPU引擎和流
  at::Device cur_device = at::Device(at::kXPU, c10::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(cur_device);
  auto stream = GpuStreamManager::Instance().get_stream();

  // 根据条件是否需要重新排列输入矩阵和结果张量
  at::Tensor m1 = is_onednn_matmul_strides(mat1) ? mat1 : mat1.contiguous();
  at::Tensor m2 = is_onednn_matmul_strides(mat2) ? mat2 : mat2.contiguous();
  at::Tensor dst = is_onednn_matmul_strides(result, true) ? result : result.contiguous();

  // 获取矩阵的维度信息
  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  // 如果结果张量为3维，则检查批处理大小是否一致
  if (dims == 3) {
    mb = dst.size(0);
    TORCH_CHECK(
        mb == m1.size(0) && mb == m2.size(0),
        "batch size mismatch, dst mb: ",
        mb,
        "m1 mb",
        m1.size(0),
        " m2 mb: ",
        m2.size(0));
  }

  // 验证偏置张量，并使其与oneDNN实现兼容
  bool with_bias = false;
  at::Tensor b = b_raw;
  if (b.defined()) {
    with_bias = true;
    if (b.dim() == 1) {
      TORCH_CHECK(
          b.size(0) == n || b.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1 ...");
      if (b.size(0) == 0) {
        with_bias = false;
      } else if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else if (m1.dim() == 2) {
        b = b.expand({1, n}).contiguous();
      }
    } else if (b.dim() == 2) {
      TORCH_CHECK(
          (b.size(0) == m && b.size(1) == n) ||
              (b.size(0) == 1 && b.size(1) == n) ||
              (b.size(0) == m && b.size(1) == 1) ||
              (b.size(0) == 1 && b.size(1) == 1),
          "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
      if (b.size(0) == 1 && b.size(1) == 1)
        b = b.expand({1, n}).contiguous();
    } else if (b.dim() == 3) {
      TORCH_CHECK(
          at::are_expandable({mb, m, n}, b.sizes()),
          "matmul bias must be expandable to:",
          dst.sizes(),
          " but got:",
          b.sizes());
      b = b.expand({mb, m, n}).contiguous();
    }
  }
    } else if (b.dim() == 0) {
      // 如果偏置张量 b 的维度为 0，则执行以下逻辑
      TORCH_CHECK(
          b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      // 检查偏置张量的元素数量是否为 1，用于 matmul 函数支持当偏置维度为 [] 时的情况
      if (m1.dim() == 3) {
        // 如果输入张量 m1 的维度为 3
        b = b.expand({mb, m, n}).contiguous();
        // 将偏置张量 b 扩展为指定的形状 {mb, m, n} 并确保其在内存中是连续的
      } else {
        // 如果输入张量 m1 的维度不为 3
        b = b.expand({1, n}).contiguous();
        // 将偏置张量 b 扩展为指定的形状 {1, n} 并确保其在内存中是连续的
      }
    } else {
      // 如果偏置张量 b 的维度不为 0，则执行以下逻辑
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
      // 抛出错误，表示在 matmul 函数中不支持当前偏置张量的维度
    }
  }

  b = b.contiguous(); // 避免重复两次重新排列

  // xpu matmul 支持 m2 张量的 ab/ba 形状，不再进行进一步检查
  auto m1_usr_dt = get_onednn_dtype(m1);
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto dst_usr_dt = get_onednn_dtype(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;
  dnnl::memory::data_type bias_dt;

  dnnl::memory::desc m1_md, m1_usr_md, m1_any_md;
  dnnl::memory::desc m2_md, m2_usr_md, m2_any_md;
  dnnl::memory::desc dst_md, dst_usr_md, dst_any_md;
  dnnl::memory::desc bias_md;

  // Naive Master weight
  // 根据 m1_dt 和 m2_dt 的数据类型，调整为对应的数据类型以支持混合精度计算
  if (m1_dt == dnnl::memory::data_type::bf16 && m2_dt == dnnl::memory::data_type::f32) {
    m2_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  } else if (
      m1_dt == dnnl::memory::data_type::f32 && m2_dt == dnnl::memory::data_type::bf16) {
    m1_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  }

  dnnl::memory::dims m1_dims, m2_dims, dst_dims, bias_dims;
  dnnl::memory::dims m1_strides, m2_strides, dst_strides, bias_strides;
  if (dims == 2) {
    // 如果张量维度为 2
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1)};
    } else {
      m2_strides = {m2.stride(1), m2.stride(0)};
    }
    dst_strides = {dst.stride(0), dst.stride(1)};
  } else {
    // 如果张量维度不为 2
    m1_dims = {mb, m, k};
    m2_dims = {mb, k, n};
    dst_dims = {mb, m, n};

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1), m2.stride(2)};
    } else {
      m2_strides = {m2.stride(0), m2.stride(2), m2.stride(1)};
    }
    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  if (with_bias) {
    // 如果存在偏置项
    bias_dims = get_onednn_dims(b);
    bias_dt = get_onednn_dtype(b);
    bias_strides = get_onednn_strides(b);
  }

  dnnl::post_ops po = attr.extract_post_ops(dst);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  // 步骤1：创建内存描述
  m1_md = dnnl::memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = dnnl::memory::desc(m2_dims, m2_dt, m2_strides);
  dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);

  // 步骤2：创建属性
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

  #if ONEDNN_SUPPORT_DETERMINISTIC
    // 如果全局上下文要求确定性算法或者确定性 MKL-DNN，则设置推演属性为确定性
    if(at::globalContext().deterministicAlgorithms() || at::globalContext().deterministicMkldnn())
        pattr.set_deterministic(true);
  #endif

  // 设置矩阵乘法原语的划分模式为用户定义的划分模式
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // 如果输入数据类型是单精度浮点数（f32），则设置浮点运算模式为严格模式
  if (m1_dt == dnnl::memory::data_type::f32) {
    pattr.set_fpmath_mode(dnnl::fpmath_mode::strict);
  }

  // STEP3: 创建原语描述
  if (with_bias) {
    // 如果有偏置项，则创建矩阵乘法的原语描述，包括偏置
    bias_md = dnnl::memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, bias_md, dst_md, pattr);
  } else {
    // 如果没有偏置项，则创建矩阵乘法的原语描述，不包括偏置
    matmul_pd = dnnl::matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }

  // 创建矩阵乘法的原语
  matmul_p = dnnl::matmul(matmul_pd);

  // 根据用户定义的描述创建输入矩阵和输出矩阵的内存对象
  m1_usr_md = dnnl::memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = dnnl::memory::desc(m2_dims, m2_usr_dt, m2_strides);
  dst_usr_md = dnnl::memory::desc(dst_dims, dst_usr_dt, dst_strides);

  // STEP4: 创建内存对象
  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());

  // 获取期望的输入、权重和输出描述
  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  // 创建 ONEDNN 内存对象并绑定输入、权重和输出张量
  dnnl::memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  at::Tensor m1_, m2_, dst_;

  // 如果属性要求使用二进制后操作，则构造二进制后操作
  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  // 计算需要的临时空间大小，并创建对应的张量作为 scratchpad
  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)}, m1.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  // 将输入、权重和输出内存对象插入参数中
  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (with_bias) {
    // 如果有偏置项，则创建偏置内存对象并插入参数中
    auto bias_m = make_onednn_memory(bias_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, bias_m});
  }

  // 使用 SYCL 交互操作执行矩阵乘法，并返回事件对象
  sycl::event matmul_event = dnnl::sycl_interop::execute(matmul_p, stream, args, deps);

  // 如果输出张量不是预期的结果张量，则将计算结果复制到预期的结果张量中
  if (!dst.is_same(result))
    result.copy_(dst);

  // 返回矩阵乘法的执行事件对象
  return matmul_event;
}

} // namespace at::native::onednn
```