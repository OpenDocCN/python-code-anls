# `.\pytorch\test\cpp\tensorexpr\test_cuda.cpp`

```py
#ifdef USE_CUDA
// 如果定义了 USE_CUDA 宏，则包含以下头文件和命名空间
#include <cmath>  // 数学函数库，包含 expf 函数
#include <sstream>  // 字符串流，用于字符串处理
#include <stdexcept>  // 标准异常类的定义

#include <gtest/gtest.h>  // Google 测试框架头文件

#include <test/cpp/tensorexpr/test_base.h>  // 测试基础头文件

#include <test/cpp/tensorexpr/padded_buffer.h>  // 带填充的缓冲区类
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>  // CUDA 代码生成器
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>  // IR 简化器
#include <torch/csrc/jit/tensorexpr/loopnest.h>  // 循环嵌套类
#include <torch/csrc/jit/tensorexpr/tensor.h>  // 张量类定义
#include <torch/csrc/jit/testing/file_check.h>  // 文件检查类

#include <c10/cuda/CUDACachingAllocator.h>  // CUDA 缓存分配器
#include <c10/util/Half.h>  // 半精度浮点数支持
#include <c10/util/irange.h>  // 整数范围迭代器支持

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;  // 使用张量表达命名空间
using namespace torch::jit::tensorexpr;  // 同样使用张量表达命名空间

// 模板函数，用于测试 CUDA 下的向量加法实现
template <typename ctype>
static void testCudaTestVectorAdd01_impl() {
  const int num_iter = 3;  // 迭代次数
  const int block_count = 16;  // 块数量
  const int block_size = 128;  // 块大小
  Dtype dtype = ToDtype<ctype>();  // 根据 ctype 获取数据类型
  BufHandle a_buf("a", {num_iter, block_count, block_size}, dtype);  // 创建缓冲区 a
  BufHandle b_buf("b", {num_iter, block_count, block_size}, dtype);  // 创建缓冲区 b
  // 创建张量 c，对应于 a 和 b 的加法操作
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return a_buf.load(n, b_id, t_id) + b_buf.load(n, b_id, t_id);  // 加载 a 和 b 中的数据并相加
      });
  LoopNest l({c});  // 创建循环嵌套对象，包含张量 c
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);  // 获取张量 c 的循环语句
  loops[1]->set_gpu_block_index(0);  // 设置第 1 个循环为 GPU 块索引
  loops[2]->set_gpu_thread_index(0);  // 设置第 2 个循环为 GPU 线程索引
  l.prepareForCodegen();  // 准备进行代码生成
  StmtPtr stmt = l.root_stmt();  // 获取根语句
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);  // 创建 CUDA 代码生成器对象
  const int N = block_count * block_size * num_iter;  // 计算总元素数量
  PaddedBuffer<ctype> a_v(N);  // 创建填充缓冲区 a_v
  PaddedBuffer<ctype> b_v(N);  // 创建填充缓冲区 b_v
  PaddedBuffer<ctype> c_v(N);  // 创建填充缓冲区 c_v
  PaddedBuffer<ctype> c_ref(N);  // 创建填充缓冲区 c_ref

  // 填充输入数据和参考结果数据
  for (const auto i : c10::irange(N)) {
    a_v(i) = ctype(i);  // 填充 a_v
    b_v(i) = ctype(i * 3 + 7);  // 填充 b_v
    c_ref(i) = a_v(i) + b_v(i);  // 计算参考结果 c_ref
  }

  // TODO: move gpu support into PaddedBuffer
  ctype* a_dev = nullptr;  // 定义设备指针 a_dev
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(ctype)));  // 在 CUDA 设备上分配 a_dev 内存
  ctype* b_dev = nullptr;  // 定义设备指针 b_dev
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(ctype)));  // 在 CUDA 设备上分配 b_dev 内存
  ctype* c_dev = nullptr;  // 定义设备指针 c_dev
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(ctype)));  // 在 CUDA 设备上分配 c_dev 内存
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));  // 将 a_v 复制到 CUDA 设备
  C10_CUDA_CHECK(
      cudaMemcpy(b_dev, b_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));  // 将 b_v 复制到 CUDA 设备
  C10_CUDA_CHECK(
      cudaMemcpy(c_dev, c_v.data(), N * sizeof(ctype), cudaMemcpyHostToDevice));  // 将 c_v 复制到 CUDA 设备
  C10_CUDA_CHECK(cudaDeviceSynchronize());  // 同步 CUDA 设备

  cuda_cg(c_dev, a_dev, b_dev);  // 在 CUDA 设备上执行 CUDA 代码生成器

  C10_CUDA_CHECK(cudaDeviceSynchronize());  // 同步 CUDA 设备
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(ctype), cudaMemcpyDeviceToHost));  // 将计算结果复制回主机
  C10_CUDA_CHECK(cudaDeviceSynchronize());  // 同步 CUDA 设备

  ExpectAllNear(c_v, c_ref, 1e-5);  // 检查计算结果和参考结果的近似性

  C10_CUDA_CHECK(cudaFree(a_dev));  // 释放 CUDA 设备上的内存 a_dev
  C10_CUDA_CHECK(cudaFree(b_dev));  // 释放 CUDA 设备上的内存 b_dev
  C10_CUDA_CHECK(cudaFree(c_dev));  // 释放 CUDA 设备上的内存 c_dev
}
#endif  // 结束 USE_CUDA 宏的条件编译部分

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-0.0f - x));  // 计算 sigmoid 函数
}
TEST(Cuda, Sigmoid_CUDA) {
  // 定义常量和变量
  const int num_iter = 3;  // 迭代次数
  const int block_count = 16;  // 块数量
  const int block_size = 128;  // 块大小
  Dtype dtype = ToDtype<float>();  // 数据类型为 float
  // 创建缓冲区对象 a_buf，表示一个三维数组
  BufHandle a_buf("a", {num_iter, block_count, block_size}, dtype);
  // 定义计算表达式 c，应用 sigmoid 函数
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return sigmoid(sigmoid(a_buf.load(n, b_id, t_id)));
      });
  // 创建 LoopNest 对象 l，并获取循环语句列表
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  // 设置 GPU 块和线程索引
  loops[1]->set_gpu_block_index(0);
  loops[2]->set_gpu_thread_index(0);
  // 为代码生成做准备
  l.prepareForCodegen();
  // 获取根语句并创建 CudaCodeGen 对象 cuda_cg
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf);
  // 计算元素总数 N
  const int N = block_count * block_size * num_iter;
  // 创建浮点数类型的 PaddedBuffer 对象
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  // 填充输入和参考输出数据
  for (const auto i : c10::irange(N)) {
    a_v(i) = float(i);
    c_ref(i) = sigmoid(sigmoid(a_v(i)));
  }

  // 分配和复制数据到 GPU
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(
      cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 在 GPU 上执行计算
  cuda_cg(c_dev, a_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 验证计算结果
  ExpectAllNear(c_v, c_ref, 1e-5);

  // 释放 GPU 上的内存
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
}

TEST(Cuda, TestVectorAdd01_CUDA) {
  // 浮点类型的测试
  testCudaTestVectorAdd01_impl<float>();
  testCudaTestVectorAdd01_impl<at::Half>();
  testCudaTestVectorAdd01_impl<double>();

  // 整数类型的测试
  testCudaTestVectorAdd01_impl<int8_t>();
  testCudaTestVectorAdd01_impl<uint8_t>();
  testCudaTestVectorAdd01_impl<int16_t>();
  testCudaTestVectorAdd01_impl<int32_t>();
  testCudaTestVectorAdd01_impl<int64_t>();
}

static void testCudaTestVectorAdd02_impl(int64_t N, int64_t block_size) {
  // 创建缓冲区对象 a_buf 和 b_buf，表示两个一维数组
  BufHandle a_buf("a", {N}, kFloat);
  BufHandle b_buf("b", {N}, kFloat);
  // 创建计算表达式 c，表示对应元素相加
  Tensor c = Compute("c", {N}, [&](const VarHandle& n) {
    return a_buf.load(n) + b_buf.load(n);
  });
  // 创建 LoopNest 对象 l，并获取循环语句列表
  LoopNest l({c});
  ForPtr n_inner;
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  // 按照给定的块大小进行循环分割
  l.splitWithMask(loops[0], block_size, &n_inner);
  loops[0]->set_gpu_block_index(0);
  n_inner->set_gpu_thread_index(0);
  // 为代码生成做准备
  l.prepareForCodegen();
  // 获取根语句并创建 CudaCodeGen 对象 cuda_cg
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  // 创建浮点数类型的 PaddedBuffer 对象
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);

  // 填充输入和参考输出数据
  for (const auto i : c10::irange(N)) {
    a_v(i) = i;
    b_v(i) = i * 3 + 7;


这些代码片段涵盖了对 CUDA 加速计算的测试和实现，包括定义常量、创建缓冲区对象、定义计算表达式、设置 GPU 索引、数据准备、在 GPU 上执行计算、验证结果以及释放 GPU 内存等步骤。
  // 对每个索引 i，计算 c_ref(i) = a_v(i) + b_v(i);
  // 这里假设 a_v 和 b_v 是主机上的 float 数组，表示输入向量

  // TODO: 将 GPU 支持移入 PaddedBuffer 类

  // 在设备上分配存储空间以存放 a_dev 数组，并检查 CUDA 函数的返回状态
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));

  // 在设备上分配存储空间以存放 b_dev 数组，并检查 CUDA 函数的返回状态
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(float)));

  // 在设备上分配存储空间以存放 c_dev 数组，并检查 CUDA 函数的返回状态
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));

  // 将主机上的 a_v 数组内容复制到设备上的 a_dev 数组中，并检查 CUDA 函数的返回状态
  C10_CUDA_CHECK(cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // 将主机上的 b_v 数组内容复制到设备上的 b_dev 数组中，并检查 CUDA 函数的返回状态
  C10_CUDA_CHECK(cudaMemcpy(b_dev, b_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // 将主机上的 c_v 数组内容复制到设备上的 c_dev 数组中，并检查 CUDA 函数的返回状态
  C10_CUDA_CHECK(cudaMemcpy(c_dev, c_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // 同步设备，等待所有设备上的操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 函数 cuda_cg，执行计算 c_dev = a_dev + b_dev
  cuda_cg(c_dev, a_dev, b_dev);

  // 再次同步设备，确保所有设备上的操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将设备上的计算结果 c_dev 复制回主机上的 c_v 数组中，并检查 CUDA 函数的返回状态
  C10_CUDA_CHECK(cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 再次同步设备，确保复制操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 使用 ExpectAllNear 函数检查 c_v 和 c_ref 数组是否在指定精度范围内接近
  ExpectAllNear(c_v, c_ref, 1e-5);

  // 释放设备上分配的存储空间 a_dev、b_dev、c_dev，并检查 CUDA 函数的返回状态
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
}

// 测试 CUDA 下的向量加法，使用不同的参数进行测试
TEST(Cuda, TestVectorAdd02_CUDA) {
  // 调用实现函数，分别测试两组参数
  testCudaTestVectorAdd02_impl(1024, 128);
  testCudaTestVectorAdd02_impl(1030, 128);
}

// 测试 CUDA 下的半精度转换
TEST(Cuda, HalfCast_CUDA) {
  // 定义半精度数据类型
  auto half = ToDtype<at::Half>();
  // 创建缓冲区 a，包含 4 个元素，类型为半精度
  BufHandle a("a", {4}, half);
  // 定义张量 b，从缓冲区 a 转换为单精度
  Tensor b = Compute("b", {4}, [&](const VarHandle& i) {
    return Cast::make(kFloat, a.load(i));
  });

  // 创建循环嵌套对象 l，用于代码生成前的准备
  LoopNest l({b});
  l.prepareForCodegen();
  // 获取根语句对象 s
  StmtPtr s = l.root_stmt();
  // 使用 CUDA 代码生成器创建对象 cg，并传入相关参数
  CudaCodeGen cg(s, {a, b});

  // 初始化输入数据和设备内存指针
  std::vector<at::Half> aData(4, 2.0f);
  std::vector<float> bData(4, 0.0f);
  at::Half* aDev = nullptr;
  float* bDev = nullptr;
  auto aSize = aData.size() * sizeof(aData[0]);
  auto bSize = bData.size() * sizeof(bData[0]);

  // 在 CUDA 设备上分配内存，并将数据从主机复制到设备
  C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bSize));
  C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(bDev, bData.data(), bSize, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 代码生成器的方法，执行 GPU 计算
  cg.call({aDev, bDev});
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将结果从设备复制回主机，并同步 CUDA 设备
  C10_CUDA_CHECK(cudaMemcpy(aData.data(), aDev, aSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(bData.data(), bDev, bSize, cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 断言所有 bData 元素均为 2.0f
  assertAllEqual(bData, 2.0f);

  // 释放 CUDA 设备上的内存
  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
}

// 测试 CUDA 下的动态二维形状
TEST(Cuda, DynamicShape2D_CUDA) {
  // 定义 Lambda 函数，接受两个参数 M 和 N，用于测试不同尺寸
  auto testWithSize = [](int32_t M, int32_t N) {
    // 定义两个变量 m 和 n，类型为整数
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    // 创建两个缓冲区 a 和 b，形状为 M x N，元素类型为单精度浮点数
    BufHandle a("a", {m, n}, kFloat);
    BufHandle b("b", {m, n}, kFloat);
    // 创建张量 c，表示矩阵相加操作
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    // 创建循环嵌套对象 l，用于代码生成前的准备
    LoopNest l({c});
    l.prepareForCodegen();
    // 获取根语句对象 s
    StmtPtr s = l.root_stmt();
    // 使用 CUDA 代码生成器创建对象 cg，并传入相关参数
    CudaCodeGen cg(s, {a, b, c, m, n});

    // 初始化输入数据和设备内存指针
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    float* aDev = nullptr;
    float* bDev = nullptr;
    float* cDev = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&aDev, aData.size() * sizeof(aData[0])));
    C10_CUDA_CHECK(cudaMalloc(&bDev, bData.size() * sizeof(bData[0])));
    C10_CUDA_CHECK(cudaMalloc(&cDev, cData.size() * sizeof(cData[0])));
    C10_CUDA_CHECK(cudaMemcpy(
        aDev,
        aData.data(),
        aData.size() * sizeof(aData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(
        bDev,
        bData.data(),
        bData.size() * sizeof(bData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(
        cDev,
        cData.data(),
        cData.size() * sizeof(cData[0]),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    // 调用 CUDA 代码生成器的方法，执行 GPU 计算
    cg.call({aDev, bDev, cDev, M, N});
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果从设备复制回主机，并同步 CUDA 设备
    C10_CUDA_CHECK(cudaMemcpy(
        cData.data(),
        cDev,
        cData.size() * sizeof(cData[0]),
        cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    # 使用 ExpectAllNear 函数检查 cData 是否接近于由 M * N 个元素组成的每个值为 3.0 的标准向量
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
    
    # 释放 GPU 上分配的内存空间，分别释放 aDev、bDev 和 cDev 指向的内存
    C10_CUDA_CHECK(cudaFree(aDev));
    C10_CUDA_CHECK(cudaFree(bDev));
    C10_CUDA_CHECK(cudaFree(cDev));
}

# 定义一个名为 "TestRand01_CUDA" 的测试用例，用于测试 CUDA 相关功能
TEST(Cuda, TestRand01_CUDA) {
  # 定义常量，迭代次数为3，CUDA 块数为16，每个块的线程数为128
  const int num_iter = 3;
  const int block_count = 16;
  const int block_size = 128;
  # 创建一个名为 c 的 Tensor，形状为 [num_iter, block_count, block_size]
  Tensor c = Compute(
      "c",
      {
          num_iter,
          block_count,
          block_size,
      },
      # 使用 Lambda 表达式生成随机数的 Compute 表达式
      [&](const VarHandle& n, const VarHandle& b_id, const VarHandle& t_id) {
        return Intrinsics::make(IntrinsicsOp::kRand, kFloat);
      });
  # 创建一个 LoopNest 对象，处理 Tensor c
  LoopNest l({c});
  # 获取 c 的循环语句列表
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  # 设置第二个循环为 GPU 块索引
  loops[1]->set_gpu_block_index(0);
  # 设置第三个循环为 GPU 线程索引
  loops[2]->set_gpu_thread_index(0);
  # 准备进行代码生成前的准备工作
  l.prepareForCodegen();
  # 获取整个计算的根语句
  StmtPtr stmt = l.root_stmt();
  # 使用 CudaCodeGen 对象进行 CUDA 代码生成，针对 Tensor c
  CudaCodeGen cuda_cg(stmt, c);
  # 计算数据总数 N
  const int N = block_count * block_size * num_iter;
  # 创建一个 PaddedBuffer<float> 对象，大小为 N
  PaddedBuffer<float> c_v(N);

  # TODO: 将 GPU 支持移入 PaddedBuffer 类

  # 在 GPU 上分配内存，将 c_dev 指向分配的设备内存
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, N * sizeof(float)));
  # 同步 CUDA 设备，等待所有设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  # 将计算传递给 CUDA 代码生成器，执行在 CUDA 设备上
  cuda_cg(c_dev);

  # 同步 CUDA 设备，等待所有设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  # 将计算结果从 CUDA 设备复制回主机端，存储在 c_v 中
  C10_CUDA_CHECK(
      cudaMemcpy(c_v.data(), c_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  # 再次同步 CUDA 设备，确保所有设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  # 计算结果的均值
  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;
  # 遍历 c_v 中的每个元素，计算均值和平方均值
  for (const auto i : c10::irange(N)) {
    float v = c_v.data()[i];
    sum1 += v;
    sum2 += v * v;
    sum3 += v * v * v;
    # 使用 ASSERT_TRUE 验证 v 的值在 [0, 1) 区间内
    ASSERT_TRUE(v >= 0 && v < 1);
  }
  # 计算均值
  sum1 /= N;
  sum2 /= N;
  sum3 /= N;
  # 预期的均值
  float sum1_mean = 1.f / 2;
  float sum2_mean = 1.f / 3;
  float sum3_mean = 1.f / 4;

  # 使用 ASSERT_NEAR 验证计算的均值与预期值之间的接近程度
  ASSERT_NEAR(sum1, sum1_mean, 2e-2);
  ASSERT_NEAR(sum2, sum2_mean, 2e-2);
  ASSERT_NEAR(sum3, sum3_mean, 2e-2);
  
  # 释放在 CUDA 设备上分配的内存
  C10_CUDA_CHECK(cudaFree(c_dev));
}
// 定义一个名为 `DynamicShapeSplit_CUDA` 的测试用例，用于测试 CUDA 动态形状分割功能
TEST(Cuda, DynamicShapeSplit_CUDA) {
  // 定义常量 N 为 4096
  constexpr int64_t N = 4096;
  // 声明一个名为 `n` 的变量句柄，类型为长整型
  VarHandle n("n", kLong);
  // 声明一个名为 `a` 的缓冲句柄，包含 `n` 个元素，元素类型为浮点型
  BufHandle a("a", {n}, kFloat);
  // 定义张量 `b`，形状为 {n}，通过 lambda 表达式计算每个元素为对应 `a` 的元素乘以 2.0f
  Tensor b =
      Compute("b", {n}, [&](const VarHandle& i) { return a.load(i) * 2.0f; });
  // 创建循环嵌套 `l`，包含张量 `b`
  LoopNest l({b});
  // 声明内部循环指针 `inner`
  ForPtr inner;
  // 获取张量 `b` 的循环语句列表
  std::vector<ForPtr> loops = l.getLoopStmtsFor(b);
  // 在 `loops[0]` 上使用大小为 1024 的掩码进行分割，并将内部循环指针存储在 `inner` 中
  l.splitWithMask(loops[0], 1024, &inner);
  // 设置 `loops[0]` 在 GPU 上的块索引为 0
  loops[0]->set_gpu_block_index(0);
  // 设置 `inner` 在 GPU 上的线程索引为 0
  inner->set_gpu_thread_index(0);
  // 获取根语句 `s`，作为 CUDA 代码生成的输入
  StmtPtr s = l.root_stmt();
  // 创建 CUDA 代码生成器 `cg`，传入参数为 {a, b, n}
  CudaCodeGen cg(s, {a, b, n});

  // 准备输入数据
  std::vector<float> aData(N, 1.0f);
  std::vector<float> bData(N, 1.0f);
  float* aDev = nullptr;
  float* bDev = nullptr;
  // 在 GPU 上分配内存并将数据从主机内存拷贝到设备内存
  C10_CUDA_CHECK(cudaMalloc(&aDev, aData.size() * sizeof(aData[0])));
  C10_CUDA_CHECK(cudaMalloc(&bDev, bData.size() * sizeof(bData[0])));
  C10_CUDA_CHECK(cudaMemcpy(
      aDev,
      aData.data(),
      aData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      bDev,
      bData.data(),
      bData.size() * sizeof(aData[0]),
      cudaMemcpyHostToDevice));
  // 同步设备，等待操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 代码生成器 `cg`，执行 CUDA 函数，传入设备指针和数据大小 `N`
  cg.call({aDev, bDev, N});
  // 同步设备，等待操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将计算结果从设备拷贝回主机内存
  C10_CUDA_CHECK(cudaMemcpy(
      bData.data(),
      bDev,
      bData.size() * sizeof(aData[0]),
      cudaMemcpyDeviceToHost));
  // 同步设备，等待操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 断言 `bData` 的每个元素接近于 2.0f，允许的误差为 1e-7
  ExpectAllNear(bData, std::vector<float>(N, 2.0f), 1e-7);

  // 释放在设备上分配的内存
  C10_CUDA_CHECK(cudaFree(aDev));
  C10_CUDA_CHECK(cudaFree(bDev));
}
// 定义一个名为 "TEST" 的测试用例，测试 CUDA 下的单块单线程全局归约功能
TEST(Cuda, OneBlockOneThreadGlobalReduce1_CUDA) {
  const static int N = 1024;  // 定义常量 N 为 1024
  BufHandle data_buf("data", {N}, kFloat);  // 创建一个名为 "data" 的缓冲区，包含 N 个元素，元素类型为 float
  BufHandle output_buf("output", {1}, kFloat);  // 创建一个名为 "output" 的缓冲区，包含 1 个元素，元素类型为 float

  // 测试添加了下列代码以进行简单的归约：
  // for (const auto bidx : c10::irange(1)) { // blockIdx.x
  //   for (const auto tidx : c10::irange(1)) { // threadIdx.x
  //     output[0] = 0.f;
  //     for (const auto i1 : c10::irange(1024)) {
  //       output[0] = output[0] + data[i1];
  //     }
  //   }
  // }

  // 初始化 output_buf 的存储为 0.0
  StorePtr init_store = output_buf.store({0}, 0.f);
  VarHandle i1("i1", kInt);  // 创建一个名为 "i1" 的整型变量
  // 加载 data_buf 中的数据元素 i1
  ExprHandle load_data = Load::make(data_buf, {i1});
  // 加载 output_buf 中的数据元素，索引为 {0}
  ExprHandle load_output = Load::make(output_buf, {0});
  // 计算加法表达式，将 load_output 和 load_data 相加
  ExprHandle add_value = load_output + load_data;
  // 将计算后的结果存储回 output_buf 的索引为 {0} 的位置
  StorePtr store_output = output_buf.store({0}, add_value);
  // 创建一个循环，对 i1 从 0 到 N 进行迭代，每次迭代执行 store_output
  ForPtr for_output = For::make(i1, 0, N, store_output);
  // 创建一个代码块，包含初始化存储和循环
  StmtPtr reduce_block = Block::make({init_store, for_output});
  VarHandle thread_idx("tidx", kInt);  // 创建一个名为 "tidx" 的整型变量
  LoopOptions thread_idx_options;  // 创建一个循环选项对象
  thread_idx_options.set_gpu_thread_index(0);  // 设置 GPU 线程索引为 0
  // 创建一个循环，对 thread_idx 从 0 到 1 进行迭代，每次迭代执行 reduce_block，应用线程索引选项
  ForPtr thread_idx_loop =
      For::make(thread_idx, 0, 1, reduce_block, thread_idx_options);
  VarHandle block_idx("bidx", kInt);  // 创建一个名为 "bidx" 的整型变量
  LoopOptions block_idx_options;  // 创建一个循环选项对象
  block_idx_options.set_gpu_block_index(0);  // 设置 GPU 块索引为 0
  // 创建一个循环，对 block_idx 从 0 到 1 进行迭代，每次迭代执行 thread_idx_loop，应用块索引选项
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, thread_idx_loop, block_idx_options);

  // 创建 CUDA 代码生成器，使用 block_idx_loop、data_buf 和 output_buf 作为参数
  CudaCodeGen cuda_cg(block_idx_loop, data_buf, output_buf);
  // 创建大小为 N 的填充缓冲区 data_v，用于存储 float 类型的数据
  PaddedBuffer<float> data_v(N);
  // 创建大小为 1 的填充缓冲区 output_v 和 output_ref，用于存储 float 类型的数据
  PaddedBuffer<float> output_v(1, "output_v");
  PaddedBuffer<float> output_ref(1, "output_ref");

  output_ref(0) = 0;  // 将 output_ref 的第一个元素设置为 0
  // 使用 c10::irange(N) 迭代，初始化 data_v 和计算 output_ref(0) 的值
  for (const auto i : c10::irange(N)) {
    data_v(i) = i;
    output_ref(0) += data_v(i);
  }

  float* data_dev = nullptr;
  // 在 CUDA 设备上分配大小为 N * sizeof(float) 的内存，将 data_v 的数据复制到 data_dev
  C10_CUDA_CHECK(cudaMalloc(&data_dev, N * sizeof(float)));
  // 将 data_v 的数据从主机内存复制到设备内存中的 data_dev
  C10_CUDA_CHECK(cudaMemcpy(
      data_dev, data_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  float* output_dev = nullptr;
  // 在 CUDA 设备上分配大小为 1 * sizeof(float) 的内存，用于存储输出数据
  C10_CUDA_CHECK(cudaMalloc(&output_dev, 1 * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 使用 CUDA 代码生成器执行 GPU 计算，将 data_dev 的数据作为输入，输出存储到 output_dev
  cuda_cg(data_dev, output_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  // 将 output_dev 的数据从设备内存复制到主机内存中的 output_v.data()
  C10_CUDA_CHECK(cudaMemcpy(
      output_v.data(), output_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 检查 output_v 和 output_ref 的所有元素是否在给定精度范围内接近
  ExpectAllNear(output_v, output_ref, 1e-5);

  // 释放 CUDA 设备上分配的内存
  C10_CUDA_CHECK(cudaFree(data_dev));
  C10_CUDA_CHECK(cudaFree(output_dev));
}
TEST(Cuda, OneBlockMultiThreadGlobalReduce1_CUDA) {
  const static int N = 1024;

  // This test does the following reduction:
  // clang-format off
  //   for b in 0..1 // block-idx
  //    for t in 0..1024: // thread-idx
  //      if t < 1:
  //        b[0] = 0
  //    // implied sync_threads
  //    for t in 0..1024: // thread-idx
  //      b[0] = b[0] + a[t] // implied atomic
  // clang-format on

  // 定义缓冲区 a_buf，包含 N 个元素的浮点数
  BufHandle a_buf("a", {N}, kFloat);
  // 定义缓冲区 b_buf，包含 1 个元素的浮点数
  BufHandle b_buf("b", {1}, kFloat);

  // 在 b_buf 中的位置 {0} 上存储初始值 0.0
  StorePtr init_store = b_buf.store({0}, 0.f);
  // 定义变量 t，类型为整数
  VarHandle t("t", kInt);
  // 定义变量 b，类型为整数
  VarHandle b("b", kInt);

  // 创建条件表达式 cond_t_lt_1，判断 t 是否小于 1
  ExprHandle cond_t_lt_1 =
      CompareSelect::make(t, 1, CompareSelectOperation::kLT);
  // 创建条件语句 masked_init_b，如果 cond_t_lt_1 成立则执行 init_store，否则为 nullptr
  CondPtr masked_init_b = Cond::make(cond_t_lt_1, init_store, nullptr);
  // 定义线程索引选项 thread_idx_options，设置 GPU 线程索引为 0
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  // 创建循环语句 for_init，从 t=0 到 t=N 执行 masked_init_b，带有线程索引选项
  ForPtr for_init = For::make(t, 0, N, masked_init_b, thread_idx_options);

  // 加载 a_buf 中位置为 {t} 的元素到 load_a
  ExprHandle load_a = Load::make(a_buf, {t});
  // 加载 b_buf 中位置为 {0} 的元素到 load_b
  ExprHandle load_b = Load::make(b_buf, {0});
  // 计算 add_value 为 load_b + load_a
  ExprHandle add_value = load_b + load_a;
  // 在 b_buf 中的位置 {0} 上存储 add_value
  StorePtr store_b = b_buf.store({0}, add_value);
  // 创建循环语句 for_b，从 t=0 到 t=N 执行 store_b，带有线程索引选项
  ForPtr for_b = For::make(t, 0, N, store_b, thread_idx_options);

  // 创建 reduce_block 块语句，包含 for_init 和 for_b
  StmtPtr reduce_block = Block::make({for_init, for_b});

  // 定义变量 block_idx，类型为整数
  VarHandle block_idx("bidx", kInt);
  // 定义块索引选项 block_idx_options，设置 GPU 块索引为 0
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  // 创建循环语句 block_idx_loop，从 block_idx=0 到 block_idx=1 执行 reduce_block，带有块索引选项
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  // 创建 CudaCodeGen 对象 cuda_cg，用 block_idx_loop、a_buf 和 b_buf 初始化
  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  // 创建大小为 N 的浮点数缓冲区 a_v
  PaddedBuffer<float> a_v(N);
  // 创建大小为 1 的浮点数缓冲区 b_v，命名为 "b_v"
  PaddedBuffer<float> b_v(1, "b_v");
  // 创建大小为 1 的浮点数缓冲区 b_ref，命名为 "b_ref"
  PaddedBuffer<float> b_ref(1, "b_ref");

  // 初始化 b_ref(0) 为 0
  b_ref(0) = 0;
  // 循环遍历范围为 N，将 a_v(i) 初始化为 i，并累加到 b_ref(0)
  for (const auto i : c10::irange(N)) {
    a_v(i) = i;
    b_ref(0) += a_v(i);
  }

  // 在设备上分配大小为 N 的浮点数内存 a_dev
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, N * sizeof(float)));
  // 将 a_v 数据复制到设备端的 a_dev
  C10_CUDA_CHECK(
      cudaMemcpy(a_dev, a_v.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  // 在设备上分配大小为 1 的浮点数内存 b_dev
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, 1 * sizeof(float)));
  // 同步设备，等待之前的操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 执行 cuda_cg，将计算结果写入 b_dev
  cuda_cg(a_dev, b_dev);

  // 再次同步设备，确保计算完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  // 将 b_dev 数据从设备端复制到 b_v
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  // 再次同步设备，确保复制完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 验证 b_v 和 b_ref 在给定精度下是否近似相等
  ExpectAllNear(b_v, b_ref, 1e-5);

  // 释放设备上的内存 a_dev 和 b_dev
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}
// 定义 CUDA 测试，测试在无线程索引写入的情况下的情况1
TEST(Cuda, NoThreadIdxWrite_1_CUDA) {
  // 此测试执行以下归约操作:
  //
  // for k in 0..1: // 块索引
  //   a[0] = 0
  //   for n in 0..2:
  //     a[0] = a[0] + n
  //   for m in 0..1024: // 线程索引
  //     b[m] = m
  //   a[1] = 1
  //   for l in 0..2:
  //     a[1] = a[1] + n
  //
  // 注意，未被线程索引覆盖的语句应该由其自己的线程索引覆盖

  const static int N = 1024;
  // 创建用于存储 a 和 b 的缓冲区
  BufHandle a_buf("a", {2}, kFloat);
  BufHandle b_buf("b", {N}, kFloat);

  VarHandle k("k", kInt);
  VarHandle l("l", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  // a[0] = 0
  // 初始化 a[0] 为 0
  StorePtr store_a0_0 = a_buf.store({0}, 0.f);
  // 加载 a[0] 的值
  ExprHandle load_a0 = Load::make(a_buf, {0});
  // 计算新的 a[0] 值并存储
  ExprHandle v1 = load_a0 + n;
  StorePtr store_a0_v1 = a_buf.store({0}, v1);
  // 对 n 进行循环，更新 a[0]
  ForPtr loop_a_0 = For::make(n, 0, 2, store_a0_v1);

  // for m in 0..1024: // 线程索引
  // 设置 b[m] = m
  StorePtr store_bm_m = b_buf.store({m}, m + 0.f);
  LoopOptions thread_idx_options;
  thread_idx_options.set_gpu_thread_index(0);
  // 对 m 进行循环，更新 b[m]
  ForPtr loop_b_1 = For::make(m, 0, N, store_bm_m, thread_idx_options);

  // a[1] = 1
  // 初始化 a[1] 为 1
  StorePtr store_a1_1 = a_buf.store({1}, 1.f);
  // 加载 a[1] 的值
  ExprHandle load_a1 = a_buf.load(1);
  // 计算新的 a[1] 值并存储
  ExprHandle v2 = load_a1 + l;
  StorePtr store_a1_v2 = a_buf.store({1}, v2);
  // 对 l 进行循环，更新 a[1]
  ForPtr loop_a_1 = For::make(l, 0, 2, store_a1_v2);

  // 构建整个归约块
  StmtPtr reduce_block =
      Block::make({store_a0_0, loop_a_0, loop_b_1, store_a1_1, loop_a_1});

  VarHandle block_idx("bidx", kInt);
  LoopOptions block_idx_options;
  block_idx_options.set_gpu_block_index(0);
  // 对 block_idx 进行循环，执行整个归约块
  ForPtr block_idx_loop =
      For::make(block_idx, 0, 1, reduce_block, block_idx_options);

  // 生成 CUDA 代码生成器，使用 a_buf 和 b_buf 作为输入
  CudaCodeGen cuda_cg(block_idx_loop, a_buf, b_buf);
  // 分配 CPU 上的缓冲区用于比较
  PaddedBuffer<float> a_v(2);
  PaddedBuffer<float> b_v(N, "b_v");
  PaddedBuffer<float> a_ref(2, "a_ref");
  PaddedBuffer<float> b_ref(N, "b_ref");

  // 设置参考值 a_ref 和 b_ref
  a_ref(0) = 0;
  for (const auto i : c10::irange(2)) {
    a_ref(0) += i;
  }
  a_ref(1) = a_ref(0) + 1;
  for (const auto i : c10::irange(N)) {
    b_ref(i) = i;
  }

  // TODO: 添加生成代码的检查
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, 2 * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, N * sizeof(float)));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 在 CUDA 设备上运行生成的代码
  cuda_cg(a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  // 将结果从设备复制回主机
  C10_CUDA_CHECK(
      cudaMemcpy(a_v.data(), a_dev, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(
      cudaMemcpy(b_v.data(), b_dev, N * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 使用 ExpectAllNear 函数检查结果是否接近预期
  ExpectAllNear(a_v, a_ref, 1e-5);
  ExpectAllNear(b_v, b_ref, 1e-5);

  // 释放 CUDA 设备上的内存
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
}
TEST(Cuda, SharedMemReduce_1_CUDA) {
  // FIXME: this test is flaky in CI.
  // This test does the following:
  //  for k in 0..1:  // block-idx
  //    alloc(c, 64)
  //    for n in 0..64:  // thread-idx
  //      c(n) = 0
  //    for m in 0..128:
  //      for n in 0..64:  // thread_idx
  //        c(n) = c(n) + a(k, m, n)
  //    b(k) = 0
  //    for n in 0..64:  // thread_idx
  //      b(k) = b(k) + c(n)
  //    free(c)

  const int M = 128;
  const int N = 64;
  const int kTotalSize = M * N;
  LoopOptions thread_idx_opt;
  thread_idx_opt.set_gpu_thread_index(0);
  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  BufHandle a("a", {1, M, N}, kFloat);
  BufHandle b("b", {1}, kFloat);
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  std::vector<StmtPtr> block;
  std::vector<ExprPtr> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{alloc<Buf>("c", dims, kFloat)};
  {
    // alloc(c, 64);
    // 分配数组 c 的内存空间，大小为 64
    AllocatePtr alloc = Allocate::make(c);
    block.push_back(alloc);
  }

  {
    //    for n in 0..64:  // thread-idx
    //      c(n) = 0
    // 初始化数组 c 中每个元素为 0
    StorePtr store_cn_0 = Store::make(c, {n}, 0.f);
    ForPtr loop_n1 = For::make(n, 0, N, store_cn_0, thread_idx_opt);
    block.push_back(loop_n1);
  }

  {
    //  for m in 0..128:
    //    for n in 0..64:  // thread_idx
    //      c(n) = c(n) + a(k, m, n)
    // 计算数组 c 的每个元素，加上数组 a 的对应元素的值
    ExprHandle load_cn = Load::make(kFloat, c, {n});
    ExprHandle a_kmn = Load::make(a, {k * (M * N) + m * N + n});
    ExprHandle v_add = load_cn + a_kmn;
    StorePtr store_cn_v = Store::make(c, {n}, v_add);
    ForPtr loop_n2 = For::make(n, 0, N, store_cn_v, thread_idx_opt);
    ForPtr loop_m1 = For::make(m, 0, M, loop_n2);
    block.push_back(loop_m1);
  }

  {
    //    b(k) = 0
    //    for n in 0..64:  // thread_idx
    //      b(k) = b(k) + c(n)
    // 初始化数组 b(k) 为 0，然后计算数组 c 的每个元素的和赋给 b(k)
    StorePtr store_bk_0 = b.store({k}, 0.f);
    block.push_back(store_bk_0);
    ExprHandle load_bk = b.load(k);
    ExprHandle load_cn = Load::make(kFloat, c, {n});
    ExprHandle v_add = load_bk + load_cn;
    StorePtr store_bk = b.store({k}, v_add);
    ForPtr loop_n3 = For::make(n, 0, N, store_bk, thread_idx_opt);
    block.push_back(loop_n3);
  }

  {
    //    free(c)
    // 释放数组 c 的内存空间
    FreePtr free_stmt = Free::make(c);
    block.push_back(free_stmt);
  }

  // 组装所有语句块成为一个大的块
  BlockPtr reduce_body = Block::make(block);
  // 用块创建一个循环，用于 GPU 加速
  ForPtr loop_k1 = For::make(k, 0, 1, reduce_body, block_idx_opt);

  // TODO: check the generated code for correctness.
  // 使用 CUDA 代码生成器生成 CUDA 代码
  CudaCodeGen cuda_cg(loop_k1, a, b);

  std::ostringstream oss;
  // 将生成的 CUDA 代码输出到字符串流中
  oss << *cuda_cg.stmt();

  // Check the c write is not masked, but the d write is.
  // 检查生成的代码中是否正确写入了 c，但未写入 d

  const std::string& verification_pattern =
      R"IR(
# CHECK: c_1 = 0
# CHECK: for (int m = 0; m < 128
# CHECK:   c_1 = c_1 +
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<1
# CHECK:   b[blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: atomicAdd(&b[blockIdx.x], c_1)
// 定义测试函数，用于验证 CUDA 设备上的局部内存减少操作
TEST(Cuda, LocalMemReduce_1_CUDA) {
  // 设定常量 M 和 N 的值，分别表示第一和第二维的大小
  const int M = 128;
  const int N = 64;
  // 计算总元素数量，用于分配内存
  const int kTotalSize = M * N;

  // 设置 GPU 线程索引的选项
  LoopOptions thread_idx_opt;
  thread_idx_opt.set_gpu_thread_index(0);
  // 设置 GPU 块索引的选项
  LoopOptions block_idx_opt;
  block_idx_opt.set_gpu_block_index(0);

  // 创建带有名称和大小的缓冲区 a、b，并指定数据类型为 float
  BufHandle a("a", {1, M, N}, kFloat);
  BufHandle b("b", {1}, kFloat);
  // 定义用于索引的变量 k、m、n
  VarHandle k("k", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);

  // 创建缓冲区 c，其大小为 1，并分配 float 类型的内存
  BufHandle c{
      alloc<Buf>("c", std::vector<ExprPtr>({alloc<IntImm>(1)}), kFloat)};
  
  // 创建存储语句列表 block_k，用于存储 b(k) = 0 的操作
  std::vector<StmtPtr> block_k;
  {
    // b(k) = 0
    StorePtr store_bk_0 = b.store({k}, 0.f);
    block_k.push_back(store_bk_0);
  }

  // 创建存储语句列表 block_n，用于存储 n 的循环体操作
  std::vector<StmtPtr> block_n;
  {
    // 分配 c 的内存空间
    AllocatePtr alloc = Allocate::make(c);
    block_n.push_back(alloc);
  }
  {
    // c(0) = 0
    StorePtr store_c0_0 = Store::make(c, {0}, 0.f);
    block_n.push_back(store_c0_0);
  }
  {
    // 对 m 的循环，计算 c(0) += a(k, m, n)
    ExprHandle load_c0 = Load::make(kFloat, c, {0});
    ExprHandle a_kmn = a.load(k * (M * N) + m * N + n);
    ExprHandle v_add = load_c0 + a_kmn;
    StorePtr store_c0_v = Store::make(c, {0}, v_add);
    ForPtr loop_m = For::make(m, 0, M, store_c0_v);
    block_n.push_back(loop_m);
  }
  {
    // 计算 b(k) += c(0)
    ExprHandle load_bk = b.load(k);
    ExprHandle load_c0 = Load::make(kFloat, c, {0});
    ExprHandle v_add = load_bk + load_c0;
    StorePtr store_bk = b.store({k}, v_add);
    block_n.push_back(store_bk);
  }
  {
    // 释放 c 的内存空间
    FreePtr free_stmt = Free::make(c);
    block_n.push_back(free_stmt);
  }
  {
    // 创建整体的语句块 block_n_stmt，包含所有针对 n 的操作
    BlockPtr block_n_stmt = Block::make(block_n);
    // 创建一个 ForPtr 对象，表示一个循环，循环变量为 n，范围从 0 到 N，循环体为 block_n_stmt，线程索引为 block_idx_opt
    ForPtr for_n = For::make(n, 0, N, block_n_stmt, thread_idx_opt);
    // 将上述创建的循环对象添加到 block_k 中
    block_k.push_back(for_n);
    
    // 创建一个 BlockPtr 对象，表示一个代码块，其中包含了 block_k 中的所有循环和语句
    BlockPtr block_k_stmt = Block::make(block_k);
    
    // 创建一个 ForPtr 对象，表示一个循环，循环变量为 k，范围从 0 到 1，循环体为 block_k_stmt，线程索引为 block_idx_opt
    ForPtr loop_k = For::make(k, 0, 1, block_k_stmt, block_idx_opt);
    
    // 创建一个 CudaCodeGen 对象，用于生成 CUDA 代码，以 loop_k 为主循环，a 和 b 为输入参数
    CudaCodeGen cuda_cg(loop_k, a, b);
    
    // 创建 PaddedBuffer 对象 a_v，用于存储大小为 1xMxN 的 float 类型数据，命名为 "a_v"
    PaddedBuffer<float> a_v(1, M, N, "a_v");
    // 创建 PaddedBuffer 对象 b_v，用于存储大小为 1 的 float 类型数据，命名为 "b_v"
    PaddedBuffer<float> b_v(1, "b_v");
    // 创建 PaddedBuffer 对象 b_ref，用于存储大小为 1 的 float 类型数据，命名为 "b_ref"
    PaddedBuffer<float> b_ref(1, "b_ref");
    
    // 初始化 b_ref 中的数据为 0
    b_ref(0) = 0;
    // 遍历 M 次，循环变量为 i
    for (const auto i : c10::irange(M)) {
        // 遍历 N 次，循环变量为 j
        for (const auto j : c10::irange(N)) {
            // 计算当前位置的值 v
            int v = i + j;
            // 将计算得到的 v 存入 a_v 中对应的位置
            a_v(0, i, j) = v;
            // 将 v 累加到 b_ref(0) 中
            b_ref(0) += v;
        }
    }
    
    // 声明并初始化一个指向设备上内存的指针 a_dev，用于存储大小为 kTotalSize 的 float 类型数据
    float* a_dev = nullptr;
    // 在 CUDA 设备上为 a_dev 分配内存空间
    C10_CUDA_CHECK(cudaMalloc(&a_dev, kTotalSize * sizeof(float)));
    // 将 a_v 中的数据拷贝到 a_dev 指向的设备内存中
    C10_CUDA_CHECK(cudaMemcpy(
        a_dev, a_v.data(), kTotalSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // 声明并初始化一个指向设备上内存的指针 b_dev，用于存储大小为 1 的 float 类型数据
    float* b_dev = nullptr;
    // 在 CUDA 设备上为 b_dev 分配内存空间
    C10_CUDA_CHECK(cudaMalloc(&b_dev, 1 * sizeof(float)));
    // 等待 CUDA 设备上的所有操作完成
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    // 使用 cuda_cg 对象执行 CUDA 计算，将结果存储在 b_dev 指向的设备内存中
    cuda_cg(a_dev, b_dev);
    
    // 等待 CUDA 设备上的所有操作完成
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    // 将 b_dev 指向的设备内存中的数据拷贝到 b_v 中的主机内存中
    C10_CUDA_CHECK(
        cudaMemcpy(b_v.data(), b_dev, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 等待 CUDA 设备上的所有操作完成
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    // 检查 b_v 中的数据与 b_ref 中的数据是否在给定精度下相近
    ExpectAllNear(b_v, b_ref, 1e-5);
    
    // 释放之前分配的 CUDA 设备上的内存空间
    C10_CUDA_CHECK(cudaFree(a_dev));
    C10_CUDA_CHECK(cudaFree(b_dev));
TEST(Cuda, HalfPropagation_CUDA) {
  auto half = ToDtype<at::Half>();  // 定义半精度数据类型
  BufHandle a("a", {4}, half);  // 创建名为 "a" 的缓冲区，包含4个元素，每个元素为半精度
  Tensor relu = Compute("relu", {4}, [&](const VarHandle& i) {
    return Max::make(a.load(i), ExprHandle(alloc<HalfImm>(0)), true);  // 计算每个元素的ReLU，使用半精度的最大值函数
  });

  LoopNest l({relu});  // 创建循环嵌套对象，包含计算结果张量relu
  l.prepareForCodegen();  // 准备进行代码生成
  StmtPtr s = l.root_stmt();  // 获取循环嵌套的根语句
  CudaCodeGen cg(s, {a, relu});  // 创建CUDA代码生成器，处理缓冲区a和张量relu

  std::ostringstream oss;
  oss << *cg.stmt();  // 将生成的CUDA代码流输出到字符串流oss中

  // Check the types used by the Max are Float.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (
# CHECK:  float v = float(a[i]);
# CHECK:  relu[i] = half(Max(v, 0.f
// 创建一个 FileCheck 对象，用于验证生成的代码是否符合指定的模式
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

// 初始化包含四个元素，每个元素值为2.0f的 at::Half 类型向量 aData
std::vector<at::Half> aData(4, 2.0f);
// 初始化包含四个元素，每个元素值为0.0f的 at::Half 类型向量 reluData
std::vector<at::Half> reluData(4, 0.0f);
// 初始化指针 aDev 和 reluDev 为 nullptr
at::Half* aDev = nullptr;
at::Half* reluDev = nullptr;
// 计算向量 aData 和 reluData 所占用的总字节数
auto aSize = aData.size() * sizeof(aData[0]);
auto reluSize = reluData.size() * sizeof(reluData[0]);

// 分配设备内存，将 aData 数据复制到设备内存中，并同步设备与主机的数据
C10_CUDA_CHECK(cudaMalloc(&aDev, aSize));
C10_CUDA_CHECK(cudaMalloc(&reluDev, reluSize));
C10_CUDA_CHECK(cudaMemcpy(aDev, aData.data(), aSize, cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaMemcpy(reluDev, reluData.data(), reluSize, cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaDeviceSynchronize());

// 调用 cg 对象的 call 方法，执行计算图计算
cg.call({aDev, reluDev});
// 将设备上的 reluDev 数据复制回主机端的 reluData，并同步设备与主机的数据
C10_CUDA_CHECK(cudaMemcpy(reluData.data(), reluDev, reluSize, cudaMemcpyDeviceToHost));
C10_CUDA_CHECK(cudaDeviceSynchronize());

// 使用断言检查 aData 和 reluData 是否完全相等
assertAllEqual(aData, reluData);

// 释放设备内存 aDev 和 reluDev
C10_CUDA_CHECK(cudaFree(aDev));
C10_CUDA_CHECK(cudaFree(reluDev));
TEST(Cuda, MaskBlockDim_CUDA) {
  // 定义两个数组的大小
  int A_SIZE = 100;
  int B_SIZE = 50;
  // 创建缓冲区对象，用于存储大小分别为 A_SIZE 和 B_SIZE 的浮点数数组
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  // 定义张量 c，表示 a_buf 中每个元素加上常数 10
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  // 定义张量 d，表示 a_buf 和 b_buf 中对应元素的和
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    return a_buf.load(i) + b_buf.load(i);
  });

  // 创建 LoopNest 对象用于管理循环嵌套
  LoopNest l({c, d});
  // 获取张量 c 的循环语句列表，并将其第一个循环指定为 GPU 块索引
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  // 获取张量 d 的循环语句列表，并将其第一个循环指定为 GPU 块索引
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);

  // 准备进行代码生成的前期工作
  l.prepareForCodegen();
  // 获取整个循环嵌套的根语句
  StmtPtr stmt = l.root_stmt();
  // 使用 CudaCodeGen 对象生成 CUDA 代码，传入参数包括 c, d, a_buf, b_buf
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建一个字符串流对象 oss，用于存储生成的 CUDA 代码
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 检查生成的 CUDA 代码中，c 的写入不被屏蔽，但 d 的写入会被屏蔽
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (blockIdx
# CHECK: c[blockIdx.x] =
// CHECK: if (blockIdx.x<50
// CHECK:   d[blockIdx.x] =)IR";

  // 运行 FileCheck 对象，验证字符串是否匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 获取 CUDA compute graph 的块维度和线程维度
  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();

  // 断言检查块维度和线程维度是否符合预期
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(1)));

  // 对于数组 a_v 和 c_ref 进行初始化
  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> c_ref(A_SIZE);

  // 对数组 a_v 和 c_ref 进行填充
  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  // 对于数组 b_v 和 d_ref 进行初始化
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  // 对数组 b_v 和 d_ref 进行填充
  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  // 分配并复制设备端内存
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 运行 CUDA compute graph，计算结果
  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将计算结果从设备端复制回主机端
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 检查计算结果是否在允许的误差范围内接近预期值
  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放设备端内存
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// Tests the case with two loops, which have different extents that are bound
/// to the same thread dimension. This is the same as the above - the smaller
/// rank write should be masked. But this time we also need to syncthreads.
TEST(Cuda, MaskThreadDim_CUDA) {
  // 定义数组大小
  int A_SIZE = 50;
  int B_SIZE = 100;

  // 创建缓冲区和张量对象
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  BufHandle b_buf("b", {B_SIZE}, kFloat);

  // 定义张量 c 和 d，分别依赖于缓冲区 a_buf 和 b_buf
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    // 返回 a_buf 中第 i/2 位置的加载值加上 b_buf 中第 i 位置的加载值
    return a_buf.load(i / 2) + b_buf.load(i);
  });

  // 创建一个循环嵌套对象 l，包含 c 和 d 两个循环变量
  LoopNest l({c, d});

  // 获取循环嵌套对象 l 中关于 c 的循环语句，并将其第一个循环设置为 GPU 线程索引 0
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_thread_index(0);

  // 获取循环嵌套对象 l 中关于 d 的循环语句，并将其第一个循环设置为 GPU 线程索引 0
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_thread_index(0);

  // 准备循环嵌套对象 l 进行代码生成
  l.prepareForCodegen();

  // 获取循环嵌套对象 l 的根语句
  StmtPtr stmt = l.root_stmt();

  // 使用 CudaCodeGen 类对 stmt 进行 CUDA 代码生成，使用 c, d, a_buf, b_buf 作为参数
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建一个字符串流 oss，并将 cuda_cg.stmt() 的内容写入其中
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 检查代码中的 c 写操作是否被屏蔽，但 d 写操作未被屏蔽
  const std::string& verification_pattern =
      R"IR(
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

// 使用 `FileCheck` 对象执行字符串模式匹配验证，确保 `oss` 字符串符合 `verification_pattern` 的要求。


  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(1)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(B_SIZE)));

// 调用 `cuda_cg` 对象的方法获取 GPU 块和线程的尺寸信息，并使用 `exprEquals` 函数断言其值符合预期。


  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

// 定义用于存储数据的缓冲区对象 `a_v`, `b_v`, `c_v`, `d_v`, `c_ref`, `d_ref`，它们分别用于存储不同大小的浮点数数组。


  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

// 初始化数组 `a_v` 和 `c_ref` 的值，分别为 `i` 的浮点数形式和 `i+10` 的浮点数形式。


  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i / 2) + b_v(i);
  }

// 初始化数组 `b_v` 和 `d_ref` 的值，分别为 `B_SIZE-i` 的浮点数形式和 `a_v(i/2) + b_v(i)` 的浮点数形式。


  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));

// 在 GPU 上分配存储空间 `a_dev`, `b_dev`, `c_dev`, `d_dev`，分别用于存储大小为 `A_SIZE` 和 `B_SIZE` 的浮点数数组。


  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));

// 将主机上的数据 `a_v`, `b_v`, `c_v`, `d_v` 复制到 GPU 的 `a_dev`, `b_dev`, `c_dev`, `d_dev` 中。


  C10_CUDA_CHECK(cudaDeviceSynchronize());

// 同步等待 CUDA 设备操作完成。


  cuda_cg(c_dev, d_dev, a_dev, b_dev);

// 调用 `cuda_cg` 对象执行 CUDA 内核函数，将 `c_dev`, `d_dev`, `a_dev`, `b_dev` 作为参数传递给该函数。


  C10_CUDA_CHECK(cudaDeviceSynchronize());

// 同步等待 CUDA 设备操作完成。


  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

// 将 GPU 上的数据 `c_dev` 和 `d_dev` 复制到主机的 `c_v` 和 `d_v` 中。


  C10_CUDA_CHECK(cudaDeviceSynchronize());

// 同步等待 CUDA 设备操作完成。


  ExpectAllNear(c_v, c_ref, 1e-5);
  ExpectAllNear(d_v, d_ref, 1e-5);

// 使用 `ExpectAllNear` 函数验证 `c_v` 和 `d_v` 的数据与 `c_ref` 和 `d_ref` 在给定的误差范围内接近。


  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// 释放在 GPU 上分配的存储空间 `a_dev`, `b_dev`, `c_dev`, `d_dev`。
    // 返回 a_buf 和 b_buf 在索引 i 处加载的值的总和
    return a_buf.load(i) + b_buf.load(i);
  });

  // 创建一个循环嵌套对象，包含变量 c 和 d
  LoopNest l({c, d});

  // 获取循环嵌套对象中关于变量 c 的循环语句，并将其设为 GPU 的第 0 个块索引
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);

  // 获取循环嵌套对象中关于变量 d 的循环语句，并将其设为 GPU 的第 1 个块索引
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(1);

  // 准备循环嵌套对象以进行代码生成
  l.prepareForCodegen();

  // 获取循环嵌套对象的根语句
  StmtPtr stmt = l.root_stmt();

  // 使用 CudaCodeGen 对象生成 CUDA 代码，传入变量 c、d、a_buf、b_buf
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建一个字符串流对象，将 CUDA 代码输出到 oss 流中
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 对变量 c 写入的操作应该对 y 进行掩码处理，对变量 d 写入的操作应该对 x 进行掩码处理
  const std::string& verification_pattern =
      R"IR(
// 检查条件：如果 blockIdx.y < 1，则执行以下代码块
// 检查条件：如果 blockIdx.x < 1，则执行以下代码块

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 获取 CUDA 网格块的尺寸
  auto blockExtents = cuda_cg.gpu_block_extents();
  // 获取 CUDA 线程块的尺寸
  auto threadExtents = cuda_cg.gpu_thread_extents();
  // 断言 CUDA 网格块的第一个维度大小与 A_SIZE 相等
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
  // 断言 CUDA 网格块的第二个维度大小与 B_SIZE 相等
  ASSERT_TRUE(exprEquals(blockExtents[1], alloc<IntImm>(B_SIZE)));

  // 分配并初始化 CPU 端的数组和引用
  PaddedBuffer<float> a_v(A_SIZE);
  PaddedBuffer<float> b_v(B_SIZE);
  PaddedBuffer<float> c_v(A_SIZE);
  PaddedBuffer<float> d_v(B_SIZE);

  PaddedBuffer<float> c_ref(A_SIZE);
  PaddedBuffer<float> d_ref(B_SIZE);

  // 初始化数组 a_v 和 c_ref
  for (const auto i : c10::irange(A_SIZE)) {
    a_v(i) = (float)i;
    c_ref(i) = (float)(i + 10);
  }

  // 初始化数组 b_v 和 d_ref
  for (const auto i : c10::irange(B_SIZE)) {
    b_v(i) = (float)(B_SIZE - i);
    d_ref(i) = a_v(i) + b_v(i);
  }

  // 分配并初始化 CUDA 设备上的内存
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));

  // 将数据从主机端复制到设备端
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  
  // 同步 CUDA 设备
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 内核函数 cuda_cg
  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  // 再次同步 CUDA 设备
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将计算结果从设备端复制到主机端
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

  // 再次同步 CUDA 设备
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 断言 c_v 和 c_ref 数组的元素在一定精度内相近
  ExpectAllNear(c_v, c_ref, 1e-5);
  // 断言 d_v 和 d_ref 数组的元素在一定精度内相近
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放 CUDA 设备上分配的内存
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

/// 测试情况：blockDim 和 threadDim 分别绑定到不同的循环中。
/// 在此情况下，由于它们是不同的，应当屏蔽两个存储操作。
// 注意：这是一个极其愚蠢的模式，我们实际不应该看到这种情况，但这是一个有用的边界案例，用来确保我们已经覆盖了所有情况。
TEST(Cuda, MaskBlockAndThreadDim_CUDA) {
  int A_SIZE = 100;
  int B_SIZE = 50;
  // 创建名称为 "a" 的缓冲区 a_buf，大小为 {A_SIZE}，数据类型为 kFloat
  BufHandle a_buf("a", {A_SIZE}, kFloat);
  // 创建名称为 "b" 的缓冲区 b_buf，大小为 {B_SIZE}，数据类型为 kFloat
  BufHandle b_buf("b", {B_SIZE}, kFloat);
  // 创建张量 c，依赖于缓冲区 a_buf，计算方法是每个元素加上 10
  Tensor c = Compute(
      "c", {A_SIZE}, [&](const VarHandle& i) { return a_buf.load(i) + 10; });
  // 创建张量 d，依赖于缓冲区 b_buf，计算方法是每个元素为 a_buf 和 b_buf 对应元素之和
  Tensor d = Compute("d", {B_SIZE}, [&](const VarHandle& i) {
    // 返回 a_buf 和 b_buf 在索引 i 处加载的数据之和
    return a_buf.load(i) + b_buf.load(i);
  });

  // 创建一个循环嵌套对象，其中包含循环变量 c 和 d
  LoopNest l({c, d});

  // 获取循环变量 c 的循环语句并设置其在 GPU 上的块索引为 0
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);

  // 获取循环变量 d 的循环语句并设置其在 GPU 上的线程索引为 0
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_thread_index(0);

  // 准备循环嵌套对象以进行代码生成
  l.prepareForCodegen();

  // 获取根语句用于 CUDA 代码生成，并传入相关参数
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建一个字符串流对象 oss，将 CUDA 代码输出到其中
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 设置验证模式字符串，使用原始字符串字面量(R"IR(...)")方式
  const std::string& verification_pattern =
      R"IR(
// 分析测试用例中的GPU计算是否正确，运行FileCheck以验证指定的模式
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

// 获取CUDA代码生成器中的块和线程维度信息
auto blockExtents = cuda_cg.gpu_block_extents();
auto threadExtents = cuda_cg.gpu_thread_extents();

// 断言块和线程维度是否符合预期
ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(A_SIZE)));
ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(B_SIZE)));

// 初始化输入和参考数据的缓冲区
PaddedBuffer<float> a_v(A_SIZE);
PaddedBuffer<float> b_v(B_SIZE);
PaddedBuffer<float> c_v(A_SIZE);
PaddedBuffer<float> d_v(B_SIZE);

PaddedBuffer<float> c_ref(A_SIZE);
PaddedBuffer<float> d_ref(B_SIZE);

// 填充输入数据和参考数据
for (const auto i : c10::irange(A_SIZE)) {
  a_v(i) = (float)i;
  c_ref(i) = (float)(i + 10);
}

for (const auto i : c10::irange(B_SIZE)) {
  b_v(i) = (float)(B_SIZE - i);
  d_ref(i) = a_v(i) + b_v(i);
}

// 在设备上分配内存并将数据从主机复制到设备
float* a_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&a_dev, A_SIZE * sizeof(float)));
float* b_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&b_dev, B_SIZE * sizeof(float)));
float* c_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&c_dev, A_SIZE * sizeof(float)));
float* d_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&d_dev, B_SIZE * sizeof(float)));
C10_CUDA_CHECK(cudaMemcpy(
    a_dev, a_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaMemcpy(
    b_dev, b_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaMemcpy(
    c_dev, c_v.data(), A_SIZE * sizeof(float), cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaMemcpy(
    d_dev, d_v.data(), B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
C10_CUDA_CHECK(cudaDeviceSynchronize());

// 调用CUDA代码生成器执行GPU计算
cuda_cg(c_dev, d_dev, a_dev, b_dev);

// 等待设备执行完成
C10_CUDA_CHECK(cudaDeviceSynchronize());

// 将计算结果从设备复制回主机
C10_CUDA_CHECK(cudaMemcpy(
    c_v.data(), c_dev, A_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
C10_CUDA_CHECK(cudaMemcpy(
    d_v.data(), d_dev, B_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
C10_CUDA_CHECK(cudaDeviceSynchronize());

// 验证计算结果是否接近预期值
ExpectAllNear(c_v, c_ref, 1e-5);
ExpectAllNear(d_v, d_ref, 1e-5);

// 释放在设备上分配的内存
C10_CUDA_CHECK(cudaFree(a_dev));
C10_CUDA_CHECK(cudaFree(b_dev));
C10_CUDA_CHECK(cudaFree(c_dev));
C10_CUDA_CHECK(cudaFree(d_dev));
}
TEST(Cuda, MaskMultiDim_CUDA) {
  int OUTER_SIZE = 10;  // 定义外部尺寸
  int A_SIZE = 100;  // 定义A的尺寸
  int B_SIZE = 50;  // 定义B的尺寸
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);  // 创建名为'a'的缓冲区a_buf，尺寸为{OUTER_SIZE, A_SIZE}，数据类型为kFloat
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);  // 创建名为'b'的缓冲区b_buf，尺寸为{OUTER_SIZE, B_SIZE}，数据类型为kFloat
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);  // 定义张量C，计算每个元素为2乘以a_buf的对应元素
      });
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);  // 定义张量D，计算每个元素为c的对应元素加上b_buf的对应元素
      });

  LoopNest l({c, d});  // 创建循环嵌套对象l，包含张量c和d
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);  // 获取张量c的循环语句
  loops[0]->set_gpu_block_index(0);  // 设置第一个循环的GPU块索引为0
  loops[1]->set_gpu_thread_index(0);  // 设置第二个循环的GPU线程索引为0
  loops = l.getLoopStmtsFor(d);  // 获取张量d的循环语句
  loops[0]->set_gpu_block_index(0);  // 设置第一个循环的GPU块索引为0
  loops[1]->set_gpu_thread_index(0);  // 设置第二个循环的GPU线程索引为0

  l.prepareForCodegen();  // 准备进行代码生成
  StmtPtr stmt = l.root_stmt();  // 获取循环嵌套的根语句
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);  // 创建CUDA代码生成器对象cuda_cg，用于生成c、d、a_buf和b_buf的CUDA代码

  std::ostringstream oss;  // 创建字符串输出流oss
  oss << *cuda_cg.stmt();  // 将CUDA代码生成器生成的语句写入oss中

  // The write to D should be masked, but not the write to C.
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[threadIdx.x + 100 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   D[threadIdx.x + 50 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());  // 使用FileCheck验证oss中的CUDA代码与verification_pattern是否匹配

  auto blockExtents = cuda_cg.gpu_block_extents();  // 获取GPU块的尺寸
  auto threadExtents = cuda_cg.gpu_thread_extents();  // 获取GPU线程的尺寸
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));  // 断言GPU块的第一个维度尺寸等于OUTER_SIZE
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));  // 断言GPU线程的第一个维度尺寸等于A_SIZE

  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);  // 创建填充缓冲区a_v，尺寸为{OUTER_SIZE, A_SIZE}
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);  // 创建填充缓冲区b_v，尺寸为{OUTER_SIZE, B_SIZE}
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);  // 创建填充缓冲区c_v，尺寸为{OUTER_SIZE, A_SIZE}
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);  // 创建填充缓冲区d_v，尺寸为{OUTER_SIZE, B_SIZE}

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);  // 创建填充缓冲区c_ref，尺寸为{OUTER_SIZE, A_SIZE}
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);  // 创建填充缓冲区d_ref，尺寸为{OUTER_SIZE, B_SIZE}

  for (const auto o : c10::irange(OUTER_SIZE)) {  // 外层循环，遍历OUTER_SIZE
    for (const auto i : c10::irange(A_SIZE)) {  // 内层循环，遍历A_SIZE
      a_v(o, i) = (float)i;  // 设置a_v的(o, i)处的值为i的浮点数形式
      c_ref(o, i) = (float)(i * 2);  // 设置c_ref的(o, i)处的值为i乘以2的浮点数形式
    }
  }

  for (const auto o : c10::irange(OUTER_SIZE)) {  // 外层循环，遍历OUTER_SIZE
    for (const auto i : c10::irange(B_SIZE)) {  // 内层循环，遍历B_SIZE
      b_v(o, i) = (float)(B_SIZE - i);  // 设置b_v的(o, i)处的值为B_SIZE减去i的浮点数形式
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);  // 设置d_ref的(o, i)处的值为c_ref的(o, i*2)处的值加上b_v的(o, i)处的值
  }
  // 分配 GPU 内存空间给数组 a_dev，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  // 分配 GPU 内存空间给数组 b_dev，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  // 分配 GPU 内存空间给数组 c_dev，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  // 分配 GPU 内存空间给数组 d_dev，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  // 将主机上的数组 a_v 的数据复制到 GPU 上的 a_dev
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机上的数组 b_v 的数据复制到 GPU 上的 b_dev
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机上的数组 c_v 的数据复制到 GPU 上的 c_dev
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机上的数组 d_v 的数据复制到 GPU 上的 d_dev
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 同步 CUDA 设备，确保所有前面的内存复制操作都已完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 函数 cuda_cg 处理 GPU 上的数据
  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  // 再次同步 CUDA 设备，确保 cuda_cg 函数执行完毕
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  // 将 GPU 上的计算结果复制回主机上的数组 c_v
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  // 将 GPU 上的计算结果复制回主机上的数组 d_v
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  // 再次同步 CUDA 设备，确保所有的结果数据都已传输回主机
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 检查主机上的数组 c_v 是否接近参考数组 c_ref，误差限为 1e-5
  ExpectAllNear(c_v, c_ref, 1e-5);
  // 检查主机上的数组 d_v 是否接近参考数组 d_ref，误差限为 1e-5
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放 GPU 上分配的内存空间
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
// Tests the case where loop extents are symbolic and not known at compile time.
// This test verifies handling of symbolic loop extents by ensuring that stores are masked
// against the extent of the other loop when it is larger.
TEST(Cuda, MaskMultiDimSymbolic_CUDA) {
  // 定义符号变量表示外层大小、A大小和B大小
  VarHandle OUTER_SIZE("OUTER_SIZE", kLong);
  VarHandle A_SIZE("A_SIZE", kLong);
  VarHandle B_SIZE("B_SIZE", kLong);

  // 定义缓冲区a和b，其形状由外层大小、A大小和B大小决定，元素类型为float
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);

  // 定义张量c，形状为{OUTER_SIZE, A_SIZE}，计算方法为2 * a_buf.load(i, j)
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });

  // 定义张量d，形状为{OUTER_SIZE, B_SIZE}，计算方法为c.load(i, j * 2) + b_buf.load(i, j)
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  // 创建循环嵌套对象，包括张量c和d
  LoopNest l({c, d});

  // 获取张量c的循环语句，并设置第一个循环的GPU块索引为0，第二个循环的GPU线程索引为0
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  // 获取张量d的循环语句，并设置第一个循环的GPU块索引为0，第二个循环的GPU线程索引为0
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  // 准备用于代码生成的环境
  l.prepareForCodegen();
  StmtPtr stmt = l.root_stmt();

  // 创建CUDA代码生成器对象，使用张量c和d以及相关的符号变量和缓冲区
  CudaCodeGen cuda_cg(stmt, c, d, OUTER_SIZE, A_SIZE, B_SIZE, a_buf, b_buf);

  // 将生成的CUDA代码转换为字符串流
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 检查生成的CUDA代码，确保正确处理A_SIZE和B_SIZE中较大的值
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.x<A_SIZE
# CHECK:   C[A_SIZE * int64_t(blockIdx.x) + int64_t(threadIdx.x)] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<B_SIZE
# CHECK:   D[B_SIZE * int64_t(blockIdx.x) + int64_t(threadIdx.x)] =)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 获取CUDA代码生成器中的GPU块和线程的尺寸信息，验证其与预期的符号变量匹配
  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], OUTER_SIZE.node()));
  ASSERT_TRUE(exprEquals(
      threadExtents[0], alloc<Max>(A_SIZE.node(), B_SIZE.node(), true)));

  // 定义外层、A大小和B大小的具体值，用于填充测试用的缓冲区和参考结果
  int64_t OUTER_EXTENT = 10;
  int64_t A_EXTENT = 100;
  int64_t B_EXTENT = 50;

  PaddedBuffer<float> a_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> b_v(OUTER_EXTENT, B_EXTENT);
  PaddedBuffer<float> c_v(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_v(OUTER_EXTENT, B_EXTENT);

  PaddedBuffer<float> c_ref(OUTER_EXTENT, A_EXTENT);
  PaddedBuffer<float> d_ref(OUTER_EXTENT, B_EXTENT);

  // 填充缓冲区a_v和c_ref，模拟张量c的计算
  for (const auto o : c10::irange(OUTER_EXTENT)) {
    for (const auto i : c10::irange(A_EXTENT)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  // 填充缓冲区b_v和d_ref，模拟张量d的计算
  for (const auto o : c10::irange(OUTER_EXTENT)) {
    for (const auto i : c10::irange(B_EXTENT)) {
      b_v(o, i) = (float)(B_EXTENT - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }
  }
}

# 分配 GPU 内存空间用于存储数组 a 的数据，大小为 OUTER_EXTENT * A_EXTENT * sizeof(float)
float* a_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_EXTENT * A_EXTENT * sizeof(float)));

# 分配 GPU 内存空间用于存储数组 b 的数据，大小为 OUTER_EXTENT * B_EXTENT * sizeof(float)
float* b_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_EXTENT * B_EXTENT * sizeof(float)));

# 分配 GPU 内存空间用于存储数组 c 的数据，大小为 OUTER_EXTENT * A_EXTENT * sizeof(float)
float* c_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_EXTENT * A_EXTENT * sizeof(float)));

# 分配 GPU 内存空间用于存储数组 d 的数据，大小为 OUTER_EXTENT * B_EXTENT * sizeof(float)
float* d_dev = nullptr;
C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_EXTENT * B_EXTENT * sizeof(float)));

# 将数组 a_v 的数据从主机内存复制到 GPU 的 a_dev 内存中
C10_CUDA_CHECK(cudaMemcpy(
    a_dev,
    a_v.data(),
    OUTER_EXTENT * A_EXTENT * sizeof(float),
    cudaMemcpyHostToDevice));

# 将数组 b_v 的数据从主机内存复制到 GPU 的 b_dev 内存中
C10_CUDA_CHECK(cudaMemcpy(
    b_dev,
    b_v.data(),
    OUTER_EXTENT * B_EXTENT * sizeof(float),
    cudaMemcpyHostToDevice));

# 将数组 c_v 的数据从主机内存复制到 GPU 的 c_dev 内存中
C10_CUDA_CHECK(cudaMemcpy(
    c_dev,
    c_v.data(),
    OUTER_EXTENT * A_EXTENT * sizeof(float),
    cudaMemcpyHostToDevice));

# 将数组 d_v 的数据从主机内存复制到 GPU 的 d_dev 内存中
C10_CUDA_CHECK(cudaMemcpy(
    d_dev,
    d_v.data(),
    OUTER_EXTENT * B_EXTENT * sizeof(float),
    cudaMemcpyHostToDevice));

# 同步 CUDA 设备，等待所有前面的 CUDA 操作完成
C10_CUDA_CHECK(cudaDeviceSynchronize());

# 调用 CUDA 函数 cuda_cg 运行在 GPU 上，处理 c_dev 和 d_dev 的数据
cuda_cg(c_dev, d_dev, OUTER_EXTENT, A_EXTENT, B_EXTENT, a_dev, b_dev);

# 再次同步 CUDA 设备，确保所有 CUDA 操作完成
C10_CUDA_CHECK(cudaDeviceSynchronize());

# 将 GPU 的 c_dev 数据复制回主机内存的 c_v 数组中
C10_CUDA_CHECK(cudaMemcpy(
    c_v.data(),
    c_dev,
    OUTER_EXTENT * A_EXTENT * sizeof(float),
    cudaMemcpyDeviceToHost));

# 将 GPU 的 d_dev 数据复制回主机内存的 d_v 数组中
C10_CUDA_CHECK(cudaMemcpy(
    d_v.data(),
    d_dev,
    OUTER_EXTENT * B_EXTENT * sizeof(float),
    cudaMemcpyDeviceToHost));

# 再次同步 CUDA 设备，确保所有 CUDA 操作完成
C10_CUDA_CHECK(cudaDeviceSynchronize());

# 检查 c_v 数组和 c_ref 数组中的所有元素是否在给定的精度范围内接近
ExpectAllNear(c_v, c_ref, 1e-5);

# 检查 d_v 数组和 d_ref 数组中的所有元素是否在给定的精度范围内接近
ExpectAllNear(d_v, d_ref, 1e-5);

# 释放 GPU 上分配的内存空间
C10_CUDA_CHECK(cudaFree(a_dev));
C10_CUDA_CHECK(cudaFree(b_dev));
C10_CUDA_CHECK(cudaFree(c_dev));
C10_CUDA_CHECK(cudaFree(d_dev));
// 定义一个名为 MaskCompoundInnerLoop_CUDA 的测试用例，用于验证两个循环在共同的父循环下融合的情况，
// 其中父循环绑定到块维度。内部循环在内部有不同的范围，但绑定到相同的线程维度。较小的循环应该被屏蔽。
TEST(Cuda, MaskCompoundInnerLoop_CUDA) {
  // 定义外部循环的大小
  int OUTER_SIZE = 10;
  // 定义数组 A 的大小
  int A_SIZE = 100;
  // 定义数组 B 的大小
  int B_SIZE = 50;
  
  // 定义四个缓冲区对象，分别为数组 a、b、c、d
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  BufHandle c_buf("c", {OUTER_SIZE, A_SIZE}, kFloat);
  BufHandle d_buf("d", {OUTER_SIZE, B_SIZE}, kFloat);

  // 设置块绑定的循环选项
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  // 设置线程绑定的循环选项
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);

  // 定义三个变量句柄 i、j、k，分别为整型
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);

  // 构建嵌套的 For 循环语句，内部包含两个 For 循环，用于填充数组 c 和 d 的值
  StmtPtr stmt = For::make(
      i,
      0,
      OUTER_SIZE,
      Block::make(
          {For::make(
               j,
               0,
               A_SIZE,
               c_buf.store({i, j}, ExprHandle(2) * a_buf.load(i, j)),
               threadBound),
           For::make(
               k,
               0,
               B_SIZE,
               d_buf.store({i, k}, c_buf.load(i, k * 2) + b_buf.load(i, k)),
               threadBound)}),
      blockBound);

  // 对生成的语句进行索引展开
  stmt = FlattenIndexes(stmt);
  // 简化生成的中间表示语句
  stmt = IRSimplifier::simplify(stmt);

  // 生成 CUDA 代码生成器对象，用于生成 CUDA 代码
  CudaCodeGen cuda_cg(stmt, a_buf, b_buf, c_buf, d_buf);

  // 创建一个字符串流对象 oss，用于保存生成的 CUDA 代码
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 验证生成的 CUDA 代码中的特定模式，确保写入数组 D 的操作被屏蔽，但写入数组 C 的操作没有被屏蔽
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: c[threadIdx.x + 100 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[threadIdx.x + 50 * blockIdx.x] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 检查生成的块维度和线程维度是否与预期相符
  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  // 定义并初始化用于存储数据的 PaddedBuffer 对象，包括数组 a、b、c、d 的值和参考值 c_ref、d_ref
  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  // 填充数组 a_v 和 c_ref 的值
  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    // 填充数组 b_v 和 d_ref 的值
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }
  }
  }



// 分隔不同的代码段，可能是在示例中的不同实验或任务之间的界限

  float* a_dev = nullptr;
  // 在设备上为数组 a_dev 分配内存空间，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));

  float* b_dev = nullptr;
  // 在设备上为数组 b_dev 分配内存空间，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));

  float* c_dev = nullptr;
  // 在设备上为数组 c_dev 分配内存空间，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));

  float* d_dev = nullptr;
  // 在设备上为数组 d_dev 分配内存空间，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));

  // 将主机上的数组 a_v 的数据拷贝到设备上的数组 a_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));

  // 将主机上的数组 b_v 的数据拷贝到设备上的数组 b_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));

  // 将主机上的数组 c_v 的数据拷贝到设备上的数组 c_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));

  // 将主机上的数组 d_v 的数据拷贝到设备上的数组 d_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));

  // 同步设备，等待所有的设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 函数 cuda_cg 处理数据在设备上的计算
  cuda_cg(a_dev, b_dev, c_dev, d_dev);

  // 再次同步设备，确保所有的设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将设备上的数组 c_dev 的数据拷贝回主机上的数组 c_v 中
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));

  // 将设备上的数组 d_dev 的数据拷贝回主机上的数组 d_v 中
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));

  // 再次同步设备，确保所有的设备任务完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 检查主机上的数组 c_v 是否和 c_ref 在给定的误差范围内相近
  ExpectAllNear(c_v, c_ref, 1e-5);

  // 检查主机上的数组 d_v 是否和 d_ref 在给定的误差范围内相近
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放设备上的内存空间，分别释放 a_dev, b_dev, c_dev, d_dev
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));



// 结束分段代码，所有与 CUDA 相关的任务已经完成
// Tests the case with two loops fused into a common parent, which is not bound
// to any block or thread dimension - however it's two inner loops are bound to
// the first thread dimensions. This should work just like the MaskThreadDim
// test where the bigger loop is unmasked but the smaller is masked.
TEST(Cuda, MaskInnerLoopOneBlock_CUDA) {
  // 定义外部循环大小
  int OUTER_SIZE = 10;
  // 定义数组A的大小
  int A_SIZE = 100;
  // 定义数组B的大小
  int B_SIZE = 50;
  // 创建数组a_buf，形状为{OUTER_SIZE, A_SIZE}，数据类型为float
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  // 创建数组b_buf，形状为{OUTER_SIZE, B_SIZE}，数据类型为float
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  // 创建数组c_buf，形状为{OUTER_SIZE, A_SIZE}，数据类型为float
  BufHandle c_buf("c", {OUTER_SIZE, A_SIZE}, kFloat);
  // 创建数组d_buf，形状为{OUTER_SIZE, B_SIZE}，数据类型为float
  BufHandle d_buf("d", {OUTER_SIZE, B_SIZE}, kFloat);

  // 创建一个循环选项，用于设置GPU块维度索引
  LoopOptions blockBound;
  blockBound.set_gpu_block_index(0);
  // 创建一个循环选项，用于设置GPU线程维度索引
  LoopOptions threadBound;
  threadBound.set_gpu_thread_index(0);
  // 定义整数变量i
  VarHandle i("i", kInt);
  // 定义整数变量j
  VarHandle j("j", kInt);
  // 定义整数变量k
  VarHandle k("k", kInt);

  // 创建一个循环语句，外部循环变量为i，范围从0到OUTER_SIZE-1
  // 内部包含一个块，其中包含两个内部循环：
  //   第一个内部循环变量为j，范围从0到A_SIZE-1，对c_buf进行存储操作
  //   第二个内部循环变量为k，范围从0到B_SIZE-1，对d_buf进行存储操作
  StmtPtr stmt = For::make(
      i,
      0,
      OUTER_SIZE,
      Block::make(
          {For::make(
               j,
               0,
               A_SIZE,
               c_buf.store({i, j}, ExprHandle(2) * a_buf.load(i, j)),
               threadBound),
           For::make(
               k,
               0,
               B_SIZE,
               d_buf.store({i, k}, c_buf.load(i, k * 2) + b_buf.load(i, k)),
               threadBound)}));

  // 对生成的语句进行索引扁平化处理
  stmt = FlattenIndexes(stmt);
  // 对生成的IR进行简化处理
  stmt = IRSimplifier::simplify(stmt);

  // 创建CUDA代码生成器对象，用于生成CUDA代码
  CudaCodeGen cuda_cg(stmt, a_buf, b_buf, c_buf, d_buf);

  // 创建一个字符串流对象oss，用于存储生成的CUDA代码
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 验证生成的CUDA代码符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i = 0; i < 10
# CHECK-NOT: if (
# CHECK: c[threadIdx.x + 100 * i] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<50
# CHECK:   d[threadIdx.x + 50 * i] =)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 获取生成的GPU块维度和线程维度的表达式
  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  // 断言块维度的第一个元素等于整数常量1
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(1)));
  // 断言线程维度的第一个元素等于整数常量A_SIZE
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  // 创建PaddedBuffer对象，用于存储和操作float类型的数据
  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  // 嵌套循环，对a_v、b_v、c_ref进行初始化赋值操作
  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
    // 嵌套循环，对b_v、d_ref进行初始化赋值操作
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
  }
}

float* a_dev = nullptr;
// 在 GPU 上分配内存用于存储 a_dev 数组，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
float* b_dev = nullptr;
// 在 GPU 上分配内存用于存储 b_dev 数组，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
float* c_dev = nullptr;
// 在 GPU 上分配内存用于存储 c_dev 数组，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
float* d_dev = nullptr;
// 在 GPU 上分配内存用于存储 d_dev 数组，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));

C10_CUDA_CHECK(cudaMemcpy(
    a_dev,
    a_v.data(),
    OUTER_SIZE * A_SIZE * sizeof(float),
    cudaMemcpyHostToDevice));
// 将主机上的 a_v 数组数据拷贝到 GPU 上的 a_dev 数组中

C10_CUDA_CHECK(cudaMemcpy(
    b_dev,
    b_v.data(),
    OUTER_SIZE * B_SIZE * sizeof(float),
    cudaMemcpyHostToDevice));
// 将主机上的 b_v 数组数据拷贝到 GPU 上的 b_dev 数组中

C10_CUDA_CHECK(cudaMemcpy(
    c_dev,
    c_v.data(),
    OUTER_SIZE * A_SIZE * sizeof(float),
    cudaMemcpyHostToDevice));
// 将主机上的 c_v 数组数据拷贝到 GPU 上的 c_dev 数组中

C10_CUDA_CHECK(cudaMemcpy(
    d_dev,
    d_v.data(),
    OUTER_SIZE * B_SIZE * sizeof(float),
    cudaMemcpyHostToDevice));
// 将主机上的 d_v 数组数据拷贝到 GPU 上的 d_dev 数组中

C10_CUDA_CHECK(cudaDeviceSynchronize());
// 等待 GPU 上的所有操作完成

cuda_cg(a_dev, b_dev, c_dev, d_dev);
// 调用 CUDA 核函数 cuda_cg 在 GPU 上执行计算，传入四个数组作为参数

C10_CUDA_CHECK(cudaDeviceSynchronize());
// 等待 GPU 上的所有操作完成

C10_CUDA_CHECK(cudaMemcpy(
    c_v.data(),
    c_dev,
    OUTER_SIZE * A_SIZE * sizeof(float),
    cudaMemcpyDeviceToHost));
// 将 GPU 上的 c_dev 数组数据拷贝回主机的 c_v 数组中

C10_CUDA_CHECK(cudaMemcpy(
    d_v.data(),
    d_dev,
    OUTER_SIZE * B_SIZE * sizeof(float),
    cudaMemcpyDeviceToHost));
// 将 GPU 上的 d_dev 数组数据拷贝回主机的 d_v 数组中

C10_CUDA_CHECK(cudaDeviceSynchronize());
// 等待 GPU 上的所有操作完成

ExpectAllNear(c_v, c_ref, 1e-5);
// 使用 ExpectAllNear 函数比较 c_v 数组和 c_ref 数组，期望它们在 1e-5 的误差范围内相等
ExpectAllNear(d_v, d_ref, 1e-5);
// 使用 ExpectAllNear 函数比较 d_v 数组和 d_ref 数组，期望它们在 1e-5 的误差范围内相等

C10_CUDA_CHECK(cudaFree(a_dev));
// 释放 GPU 上分配的 a_dev 数组的内存
C10_CUDA_CHECK(cudaFree(b_dev));
// 释放 GPU 上分配的 b_dev 数组的内存
C10_CUDA_CHECK(cudaFree(c_dev));
// 释放 GPU 上分配的 c_dev 数组的内存
C10_CUDA_CHECK(cudaFree(d_dev));
// 释放 GPU 上分配的 d_dev 数组的内存
// Tests the case with two loop nests, each of which bound to the same block
// size, but with internal loops bound to different thread rank (ie x and y). In
// this case both bodies must be masked against the other dimension being > 0.
// Note: this is a bit degenerate no one would actually write this for perf.
TEST(Cuda, MaskMultiDimMultiAxis_CUDA) {
  // 定义外部循环大小
  int OUTER_SIZE = 10;
  // 定义数组 A 的大小
  int A_SIZE = 30;
  // 定义数组 B 的大小
  int B_SIZE = 15;
  // 创建数组 A 的缓冲区
  BufHandle a_buf("a", {OUTER_SIZE, A_SIZE}, kFloat);
  // 创建数组 B 的缓冲区
  BufHandle b_buf("b", {OUTER_SIZE, B_SIZE}, kFloat);
  // 定义张量 C，使用 lambda 表达式生成计算函数
  Tensor c = Compute(
      "C", {OUTER_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  // 定义张量 D，使用 lambda 表达式生成计算函数
  Tensor d = Compute(
      "D", {OUTER_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  // 创建 LoopNest 对象，包含张量 C 和 D
  LoopNest l({c, d});
  // 获取张量 C 的循环语句
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  // 设置张量 C 的 GPU 块索引
  loops[0]->set_gpu_block_index(0);
  // 设置张量 C 的 GPU 线程索引
  loops[1]->set_gpu_thread_index(0);
  // 获取张量 D 的循环语句
  loops = l.getLoopStmtsFor(d);
  // 设置张量 D 的 GPU 块索引
  loops[0]->set_gpu_block_index(0);
  // 设置张量 D 的 GPU 线程索引
  loops[1]->set_gpu_thread_index(1);

  // 准备用于代码生成的 LoopNest 对象
  l.prepareForCodegen();
  // 获取根语句的指针
  StmtPtr stmt = l.root_stmt();
  // 创建 CudaCodeGen 对象，用于生成 CUDA 代码
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建输出字符串流对象
  std::ostringstream oss;
  // 将 CUDA 代码写入输出字符串流
  oss << *cuda_cg.stmt();

  // 验证生成的 CUDA 代码是否符合预期模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: if (threadIdx.y<1
# CHECK:   C[threadIdx.x + 30 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (threadIdx.x<1
# CHECK:   D[threadIdx.y + 15 * blockIdx.x] =)IR";
  // 使用 FileCheck 进行模式匹配验证
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 获取 GPU 块的扩展范围
  auto blockExtents = cuda_cg.gpu_block_extents();
  // 获取 GPU 线程的扩展范围
  auto threadExtents = cuda_cg.gpu_thread_extents();
  // 断言 GPU 块的范围是否与预期相等
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_SIZE)));
  // 断言 GPU 线程的范围是否与预期相等
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  // 创建数组 A 的填充缓冲区
  PaddedBuffer<float> a_v(OUTER_SIZE, A_SIZE);
  // 创建数组 B 的填充缓冲区
  PaddedBuffer<float> b_v(OUTER_SIZE, B_SIZE);
  // 创建数组 C 的填充缓冲区
  PaddedBuffer<float> c_v(OUTER_SIZE, A_SIZE);
  // 创建数组 D 的填充缓冲区
  PaddedBuffer<float> d_v(OUTER_SIZE, B_SIZE);

  // 创建数组 C 的参考填充缓冲区
  PaddedBuffer<float> c_ref(OUTER_SIZE, A_SIZE);
  // 创建数组 D 的参考填充缓冲区
  PaddedBuffer<float> d_ref(OUTER_SIZE, B_SIZE);

  // 填充数组 A 和数组 C 的填充缓冲区
  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  // 填充数组 B 和数组 D 的填充缓冲区
  for (const auto o : c10::irange(OUTER_SIZE)) {
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
    }
  }
  }
  // 分配设备内存并检查错误，用于存储数组 a_dev，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  float* a_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  // 分配设备内存并检查错误，用于存储数组 b_dev，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  float* b_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_SIZE * B_SIZE * sizeof(float)));
  // 分配设备内存并检查错误，用于存储数组 c_dev，大小为 OUTER_SIZE * A_SIZE * sizeof(float)
  float* c_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_SIZE * A_SIZE * sizeof(float)));
  // 分配设备内存并检查错误，用于存储数组 d_dev，大小为 OUTER_SIZE * B_SIZE * sizeof(float)
  float* d_dev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_SIZE * B_SIZE * sizeof(float)));

  // 将主机内存中的数组 a_v 的数据复制到设备内存 a_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机内存中的数组 b_v 的数据复制到设备内存 b_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机内存中的数组 c_v 的数据复制到设备内存 c_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将主机内存中的数组 d_v 的数据复制到设备内存 d_dev 中
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));

  // 同步所有设备上的 CUDA 线程
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 核函数 cuda_cg 运行在设备上，处理 c_dev, d_dev, a_dev, b_dev 四个数组
  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  // 再次同步所有设备上的 CUDA 线程
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 将设备内存 c_dev 的数据复制回主机内存数组 c_v 中
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  // 将设备内存 d_dev 的数据复制回主机内存数组 d_v 中
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));

  // 再次同步所有设备上的 CUDA 线程
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 使用 ExpectAllNear 函数检查 c_v 和 c_ref 数组的每个元素是否在指定精度内接近
  ExpectAllNear(c_v, c_ref, 1e-5);
  // 使用 ExpectAllNear 函数检查 d_v 和 d_ref 数组的每个元素是否在指定精度内接近
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放设备内存 a_dev
  C10_CUDA_CHECK(cudaFree(a_dev));
  // 释放设备内存 b_dev
  C10_CUDA_CHECK(cudaFree(b_dev));
  // 释放设备内存 c_dev
  C10_CUDA_CHECK(cudaFree(c_dev));
  // 释放设备内存 d_dev
  C10_CUDA_CHECK(cudaFree(d_dev));
}

// Tests the case with two loop nests, each bound to both Block and Thread but
// the second loop is smaller in both cases - the second store must be masked
// for both the block and thread dimension.
TEST(Cuda, MaskMultiDimMultiLevel_CUDA) {
  // 定义外层循环的大小
  int OUTER_A_SIZE = 10;
  // 定义外层循环中内部循环的大小
  int OUTER_B_SIZE = 5;
  // 定义内部循环的大小
  int A_SIZE = 30;
  // 定义内部循环中内部循环的大小
  int B_SIZE = 15;
  // 创建名为 a 的缓冲区，形状为 {OUTER_A_SIZE, A_SIZE}，数据类型为 kFloat
  BufHandle a_buf("a", {OUTER_A_SIZE, A_SIZE}, kFloat);
  // 创建名为 b 的缓冲区，形状为 {OUTER_B_SIZE, B_SIZE}，数据类型为 kFloat
  BufHandle b_buf("b", {OUTER_B_SIZE, B_SIZE}, kFloat);
  // 创建张量 C，形状为 {OUTER_A_SIZE, A_SIZE}，使用 lambda 表达式定义计算逻辑
  Tensor c = Compute(
      "C", {OUTER_A_SIZE, A_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return ExprHandle(2) * a_buf.load(i, j);
      });
  // 创建张量 D，形状为 {OUTER_B_SIZE, B_SIZE}，使用 lambda 表达式定义计算逻辑
  Tensor d = Compute(
      "D", {OUTER_B_SIZE, B_SIZE}, [&](const VarHandle& i, const VarHandle& j) {
        return c.load(i, j * 2) + b_buf.load(i, j);
      });

  // 创建循环嵌套对象，并传入张量 C 和 D
  LoopNest l({c, d});
  // 获取张量 C 的循环语句，并设置其在 GPU 中的块索引和线程索引
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);
  // 获取张量 D 的循环语句，并设置其在 GPU 中的块索引和线程索引
  loops = l.getLoopStmtsFor(d);
  loops[0]->set_gpu_block_index(0);
  loops[1]->set_gpu_thread_index(0);

  // 准备进行代码生成前的准备工作
  l.prepareForCodegen();
  // 获取根语句并传递给 CUDA 代码生成器，同时传递相关的缓冲区和张量
  StmtPtr stmt = l.root_stmt();
  CudaCodeGen cuda_cg(stmt, c, d, a_buf, b_buf);

  // 创建一个字符串流 oss，并将 CUDA 代码生成器的语句写入其中
  std::ostringstream oss;
  oss << *cuda_cg.stmt();

  // 验证生成的 CUDA 代码，确保对张量 D 的写操作被正确地掩码
  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: if (
# CHECK: C[threadIdx.x + 30 * blockIdx.x] =
# CHECK: __syncthreads();
# CHECK: if (blockIdx.x<5
# CHECK:   if (threadIdx.x<15
# CHECK:     D[threadIdx.x + 15 * blockIdx.x] =)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 检查块和线程维度的表达式是否正确
  auto blockExtents = cuda_cg.gpu_block_extents();
  auto threadExtents = cuda_cg.gpu_thread_extents();
  ASSERT_TRUE(exprEquals(blockExtents[0], alloc<IntImm>(OUTER_A_SIZE)));
  ASSERT_TRUE(exprEquals(threadExtents[0], alloc<IntImm>(A_SIZE)));

  // 创建填充缓冲区用于测试
  PaddedBuffer<float> a_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> b_v(OUTER_B_SIZE, B_SIZE);
  PaddedBuffer<float> c_v(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_v(OUTER_B_SIZE, B_SIZE);

  // 创建参考的填充缓冲区 c_ref 和 d_ref，并填充数据
  PaddedBuffer<float> c_ref(OUTER_A_SIZE, A_SIZE);
  PaddedBuffer<float> d_ref(OUTER_B_SIZE, B_SIZE);

  // 填充缓冲区 a_v 和 c_ref 的数据
  for (const auto o : c10::irange(OUTER_A_SIZE)) {
    for (const auto i : c10::irange(A_SIZE)) {
      a_v(o, i) = (float)i;
      c_ref(o, i) = (float)(i * 2);
    }
  }

  // 填充缓冲区 b_v 和 d_ref 的数据
  for (const auto o : c10::irange(OUTER_B_SIZE)) {
    for (const auto i : c10::irange(B_SIZE)) {
      b_v(o, i) = (float)(B_SIZE - i);
      d_ref(o, i) = c_ref(o, i * 2) + b_v(o, i);
  }
  // 分配设备端的内存空间，用于存储矩阵 a 的数据
  float* a_dev = nullptr;
  // 调用 CUDA 函数分配内存，并检查是否成功
  C10_CUDA_CHECK(cudaMalloc(&a_dev, OUTER_A_SIZE * A_SIZE * sizeof(float)));
  // 分配设备端的内存空间，用于存储矩阵 b 的数据
  float* b_dev = nullptr;
  // 调用 CUDA 函数分配内存，并检查是否成功
  C10_CUDA_CHECK(cudaMalloc(&b_dev, OUTER_B_SIZE * B_SIZE * sizeof(float)));
  // 分配设备端的内存空间，用于存储矩阵 c 的数据
  float* c_dev = nullptr;
  // 调用 CUDA 函数分配内存，并检查是否成功
  C10_CUDA_CHECK(cudaMalloc(&c_dev, OUTER_A_SIZE * A_SIZE * sizeof(float)));
  // 分配设备端的内存空间，用于存储矩阵 d 的数据
  float* d_dev = nullptr;
  // 调用 CUDA 函数分配内存，并检查是否成功
  C10_CUDA_CHECK(cudaMalloc(&d_dev, OUTER_B_SIZE * B_SIZE * sizeof(float)));
  // 将矩阵 a 的数据从主机端复制到设备端
  C10_CUDA_CHECK(cudaMemcpy(
      a_dev,
      a_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将矩阵 b 的数据从主机端复制到设备端
  C10_CUDA_CHECK(cudaMemcpy(
      b_dev,
      b_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将矩阵 c 的数据从主机端复制到设备端
  C10_CUDA_CHECK(cudaMemcpy(
      c_dev,
      c_v.data(),
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 将矩阵 d 的数据从主机端复制到设备端
  C10_CUDA_CHECK(cudaMemcpy(
      d_dev,
      d_v.data(),
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyHostToDevice));
  // 等待设备端所有操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 调用 CUDA 函数计算 c_dev 和 d_dev 的结果
  cuda_cg(c_dev, d_dev, a_dev, b_dev);

  // 等待设备端所有操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  // 将计算结果从设备端复制回主机端的 c_v
  C10_CUDA_CHECK(cudaMemcpy(
      c_v.data(),
      c_dev,
      OUTER_A_SIZE * A_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  // 将计算结果从设备端复制回主机端的 d_v
  C10_CUDA_CHECK(cudaMemcpy(
      d_v.data(),
      d_dev,
      OUTER_B_SIZE * B_SIZE * sizeof(float),
      cudaMemcpyDeviceToHost));
  // 等待设备端所有操作完成
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 检查计算结果 c_v 是否接近于预期结果 c_ref，允许误差为 1e-5
  ExpectAllNear(c_v, c_ref, 1e-5);
  // 检查计算结果 d_v 是否接近于预期结果 d_ref，允许误差为 1e-5
  ExpectAllNear(d_v, d_ref, 1e-5);

  // 释放设备端内存空间
  C10_CUDA_CHECK(cudaFree(a_dev));
  C10_CUDA_CHECK(cudaFree(b_dev));
  C10_CUDA_CHECK(cudaFree(c_dev));
  C10_CUDA_CHECK(cudaFree(d_dev));
}

} // namespace jit
} // namespace torch

#endif


注释：


}
// 结束 jit 命名空间
} // namespace jit
// 结束 torch 命名空间
} // namespace torch
// 结束条件编译指令
#endif
```