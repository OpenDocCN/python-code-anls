# `.\pytorch\aten\src\ATen\test\quantized_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架头文件

#include <ATen/ATen.h>  // 引入 PyTorch ATen 库
#include <ATen/test/test_assert.h>
#include <cmath>  // 数学函数库，例如 round 函数
#include <iostream>  // 标准输入输出流库
#include <limits>  // 数值极限库
#include <memory>  // 内存管理库
#include <sstream>  // 字符串流库
#include <type_traits>  // 类型特性库
// For quantize_val
#include <ATen/native/quantized/AffineQuantizer.h>  // 引入量化相关头文件
#include <c10/core/ScalarType.h>  // 引入标量类型定义
#include <c10/util/irange.h>  // 引入迭代范围工具
#include <ATen/quantized/Quantizer.h>  // 引入量化器

using namespace at;  // 使用 ATen 命名空间

#ifndef ATEN_CPU_STATIC_DISPATCH

TEST(TestQTensor, QuantDequantAPIs) {  // 定义测试用例 TestQTensor.QuantDequantAPIs
  auto num_elements = 10;  // 元素数量设定为 10
  Tensor r = at::ones({num_elements});  // 创建所有元素为 1 的张量 r

  const double scale = 1.0;  // 设置量化比例为 1.0
  const int64_t zero_point = 2;  // 设置量化零点为 2
  const Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);  // 对张量 r 进行量化

  ASSERT_EQ(qr.q_scale(), scale);  // 断言量化后的张量 qr 的量化比例为 scale
  ASSERT_EQ(qr.q_zero_point(), zero_point);  // 断言量化后的张量 qr 的量化零点为 zero_point
  ASSERT_TRUE(qr.is_quantized());  // 断言张量 qr 已被量化
  ASSERT_FALSE(r.is_quantized());  // 断言原始张量 r 未被量化

  // int_repr
  Tensor int_repr = qr.int_repr();  // 获取量化后的整数表示
  auto* int_repr_data = int_repr.data_ptr<uint8_t>();  // 获取整数表示的数据指针
  for (const auto i : c10::irange(num_elements)) {  // 遍历元素数量范围
    ASSERT_EQ(int_repr_data[i], 3);  // 断言整数表示的每个元素值为 3
  }

  // Check for correct quantization
  auto r_data = r.data_ptr<float>();  // 获取原始张量 r 的数据指针
  auto qr_data = qr.data_ptr<quint8>();  // 获取量化后的张量 qr 的数据指针
  for (const auto i : c10::irange(num_elements)) {  // 遍历元素数量范围
    ASSERT_EQ(
        native::quantize_val<quint8>(scale, zero_point, r_data[i]).val_,
        qr_data[i].val_);  // 断言量化值的正确性
  }

  // Check for correct dequantization
  Tensor rqr = qr.dequantize();  // 对量化后的张量 qr 进行反量化操作
  auto rqr_data = rqr.data_ptr<float>();  // 获取反量化后张量 rqr 的数据指针
  for (const auto i : c10::irange(num_elements)) {  // 遍历元素数量范围
    ASSERT_EQ(r_data[i], rqr_data[i]);  // 断言反量化后的值与原始值相等
  }
  for (const auto i : c10::irange(num_elements)) {  // 遍历元素数量范围
    ASSERT_EQ(
        r_data[i],
        native::dequantize_val(qr.q_scale(), qr.q_zero_point(), qr_data[i]));  // 断言反量化值的正确性
  }

  // Check for correct requantization
  double new_scale = 2.0;  // 设置新的量化比例
  int64_t new_zero_point = 1;  // 设置新的量化零点
  Tensor reqr = at::quantize_per_tensor(r, new_scale, new_zero_point, kQInt8);  // 对张量 r 进行新的量化
  auto reqr_data = reqr.data_ptr<qint8>();  // 获取新量化后的张量 reqr 的数据指针
  for (const auto i : c10::irange(num_elements)) {  // 遍历元素数量范围
    reqr_data[i].val_ =
        native::requantize_val<quint8, qint8>(
            scale, zero_point, new_scale, new_zero_point, qr_data[i])
            .val_;  // 执行重新量化操作并更新值
    const qint8 expected =
        native::quantize_val<qint8>(new_scale, new_zero_point, rqr_data[i]);  // 获取期望的量化值
    ASSERT_EQ(expected.val_, reqr_data[i].val_);  // 断言重新量化后的值的正确性
  }
}

TEST(TestQTensor, RoundingMode) {
  // We assume that quantization is defined as:
  //   qx = clamp(zero_point + round(x / scale))
  // If the zero_point is added before rounding, the result will be wrong.
  int32_t zero_point = 5;  // 设置量化零点为 5
  std::vector<float> x_values{
      -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5};  // 设置浮点数向量 x_values
  std::vector<uint8_t> qx_expect{
      0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11};  // 期望的量化结果 qx_expect，scale = 1.0

  Tensor x = from_blob(x_values.data(), x_values.size());  // 创建张量 x，并使用 x_values 初始化
  Tensor qx = at::quantize_per_tensor(x, /*scale=*/1.0, zero_point, kQUInt8);  // 对张量 x 进行量化

  auto qx_data = qx.data_ptr<quint8>();  // 获取量化后的张量 qx 的数据指针
  for (const auto idx : c10::irange(x_values.size())) {  // 遍历 x_values 的大小范围
    // 使用 ASSERT_EQ 宏断言 qx_expect[idx] 与 qx_data[idx].val_ 相等，
    // 否则输出错误信息表明在舍入过程中对索引 idx 的元素进行了失败的打破平局处理。
    ASSERT_EQ(qx_expect[idx], qx_data[idx].val_)
        << "Tie breaking during rounding element " << idx << " failed!";
  }
}

// 定义测试用例 TestQTensor.Item，测试量化张量的 item 方法
TEST(TestQTensor, Item) {
  // 创建一个值为 1 的张量 r
  Tensor r = at::ones({1});
  // 定义量化的比例和零点
  const float scale = 1;
  const int32_t zero_point = 2;
  // 对张量 r 进行量化
  Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);
  // 断言量化后的值与原始值相等
  ASSERT_EQ(r.item().to<float>(), qr.item().to<float>());
}

// 定义测试用例 TestQTensor.EmptyQuantized，测试创建空的仿射量化张量
TEST(TestQTensor, EmptyQuantized) {
  // 定义量化的比例、零点、初始值、张量元素数量
  float scale = 0.5;
  int zero_point = 10;
  int val = 100;
  int numel = 10;
  // 创建空的仿射量化张量 q
  Tensor q = at::_empty_affine_quantized(
      {numel}, at::device(at::kCPU).dtype(kQUInt8), scale, zero_point);
  // 将初始值赋给量化张量 q
  auto* q_data = q.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    q_data[i].val_ = val;
  }

  // 反量化
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  // 断言反量化后的值与期望值相等
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ(r_data[i], (val - zero_point) * scale);
  }
}

// 定义测试用例 TestQTensor.EmptyPerchannelQuantized，测试创建空的通道仿射量化张量
TEST(TestQTensor, EmptyPerchannelQuantized) {
  // 定义张量元素数量、量化的比例和零点、初始值、通道轴
  int numel = 10;
  auto scales = rand({numel}).toType(kDouble);
  auto zero_points = randint(10, {10}).toType(kLong);
  int val = 100;
  int ch_axis = 0;
  // 创建空的通道仿射量化张量 q
  Tensor q = at::_empty_per_channel_affine_quantized(
      {numel},
      scales,
      zero_points,
      ch_axis,
      at::device(at::kCPU).dtype(kQUInt8));
  // 将初始值赋给量化张量 q
  auto* q_data = q.data_ptr<quint8>();
  for (const auto i : c10::irange(numel)) {
    q_data[i].val_ = val;
  }

  // 反量化
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  // 断言反量化后的值与期望值相等
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ(
        r_data[i],
        (val - zero_points[i].item().to<int>()) * scales[i].item().to<float>());
  }
}

// 定义测试用例 TestQTensor.QuantizePerChannel4d，测试创建 4 维张量的通道量化
TEST(TestQTensor, QuantizePerChannel4d) {
  // 定义通道数 C，高度 H，宽度 W
  int C = 64, H = 10, W = 10;
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  int ch_axis = 1;
  // 创建 4 维张量 tensor，每个 H x W 图像是一个范围为 0 到 H*W 的序列
  Tensor tensor = at::empty({1, C, H, W}, at::device(at::kCPU).dtype(kFloat));
  auto* tensor_data = tensor.mutable_data_ptr<float>();
  // 初始化张量 tensor 的数据
  for (int c = 0, i = 0; c < C; ++c) {
    for (int e = 0; e < H * W; ++e, ++i) {
      tensor_data[i] = e;
    }
  }
  // 对张量进行通道量化，并检查值
  Tensor q = at::native::quantize_per_channel(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  for (int c = 0, i = 0; c < C; ++c) {
    float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
    int64_t zero_point = zero_points[c].item<int64_t>();
    for (int e = 0; e < H * W; ++e, ++i) {
      // 如果值大于最大的 uint8_t 值，则将 qval 缩小到 255
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      // 断言量化后的值与期望值相等
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}
// 定义名为 TEST 的测试用例，名称为 TestQTensor，测试量化四维张量在通道尾部的情况
TEST(TestQTensor, QuantizePerChannel4dChannelsLast) {
  // 设置张量维度参数：通道数 C = 64，高度 H = 10，宽度 W = 10
  int C = 64, H = 10, W = 10;
  // 随机生成长度为 C 的双精度浮点数数组作为缩放因子
  auto scales = rand({C}).toType(kDouble);
  // 随机生成长度为 C 的长整型数组作为零点
  auto zero_points = randint(10, {C}).toType(kLong);
  // 通道轴的索引，此处为 1
  int ch_axis = 1;
  
  // 创建一个四维张量 tensor，形状为 {1, C, H, W}，数据类型为 float，存储在 CPU 上，内存格式为通道尾部
  Tensor tensor = at::empty(
      {1, C, H, W},
      at::device(at::kCPU).dtype(kFloat).memory_format(
          at::MemoryFormat::ChannelsLast));
  // 获取 tensor 数据指针
  auto* tensor_data = tensor.data_ptr<float>();
  
  // 填充 tensor 数据，使每个 H x W 图像的值范围为 0 到 H*W
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      tensor_data[i] = e;
    }
  }

  // 对 tensor 进行量化，并检查值
  Tensor q = at::native::quantize_per_channel(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  // 获取量化后数据的指针
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  
  // 遍历量化后的数据，计算量化后的值，并与期望值进行比较
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      // 计算缩放因子的倒数
      float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
      // 获取零点值
      int64_t zero_point = zero_points[c].item<int64_t>();
      // 将 e * inv_scale 后的值四舍五入，并确保不超过 uint8_t 的最大值 255
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      // 断言量化后的值与期望的 qval 相等
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}

// 定义名为 TEST 的测试用例，名称为 TestQTensor，测试从 Blob 数据创建量化张量并进行仿射变换的情况
TEST(TestQTensor, FromBlobQuantizedPerTensor) {
  // 设置量化的比例和零点
  const float scale = 0.1;
  const int64_t zero_point = 10;
  // 定义张量的形状
  std::vector<int64_t> shape = {5, 10};
  // 计算张量的元素总数
  auto numel = c10::multiply_integers(shape);

  // 设置张量选项为 kQUInt8 类型
  TensorOptions options(at::kQUInt8);

  // 创建一个自定义的 uint8_t 类型的向量，并初始化大小为 numel
  auto custom_vec = std::make_unique<std::vector<uint8_t>>();
  custom_vec->resize(numel);

  // 获取自定义数据的指针
  uint8_t* custom_data = custom_vec->data();
  // 填充自定义数据，值为索引 i
  for (const auto i : c10::irange(numel)) {
    custom_data[i] = i;
  }
  
  // 定义一个布尔变量，用于跟踪自定义数据是否已被删除
  bool customDataDeleted{false};
  // 释放 custom_vec 的所有权，并定义删除器，确保在释放 custom_vec 时检查指针的一致性，并将 customDataDeleted 设置为 true
  auto deleteWhenDone = custom_vec.release();
  auto deleter = [deleteWhenDone, custom_data, &customDataDeleted](void* inp) {
    ASSERT_EQ((void*)inp, (void*)custom_data);
    delete deleteWhenDone;
    customDataDeleted = true;
  };

  {
  // 使用 from_blob_quantized_per_tensor_affine 创建量化张量 qtensor
  Tensor qtensor = at::from_blob_quantized_per_tensor_affine(custom_data, shape, deleter, scale, zero_point, options);

  // 获取量化张量数据的指针
  uint8_t* q_data = (uint8_t*)qtensor.data_ptr<quint8>();
  // 断言量化后的数据与自定义数据的每个值相等
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ((int)custom_data[i], (int)q_data[i]);
  }
  
  // 遍历量化张量的形状，检查量化后的值是否满足仿射变换的要求
  for (int h = 0, i = 0; h < shape[0]; ++h) {
    for (int w = 0; w < shape[1]; ++w, ++i) {
      // 断言量化张量的每个元素值是否等于 (custom_data[i] - zero_point) * scale
      ASSERT_EQ(
          qtensor[h][w].item<float>(),
          (custom_data[i] - zero_point) * scale);
    }
  }
  // 断言量化张量的比例因子是否等于预期的 scale
  ASSERT_EQ((float)qtensor.q_scale(), (float)scale);
  // 断言量化张量的零点是否等于预期的 zero_point
  ASSERT_EQ(qtensor.q_zero_point(), zero_point);
  }
  // 断言 customDataDeleted 已被设置为 true，即自定义数据已被正确删除
  TORCH_CHECK(customDataDeleted);
}
TEST(TestQTensor, FromBlobQuantizedPerChannel) {
  // 定义测试所需的通道数、高度和宽度
  int C = 64, H = 10, W = 5;
  // 创建张量的形状
  std::vector<int64_t> shape = {1, C, H, W};
  // 随机生成缩放因子并转换为双精度类型
  auto scales = rand({C}).toType(kDouble);
  // 随机生成零点并转换为长整型
  auto zero_points = randint(10, {C}).toType(kLong);
  // 计算张量中元素的总数
  auto numel = c10::multiply_integers(shape);
  // 通道轴的索引
  int ch_axis = 1;
  // 张量选项，使用kQUInt8类型
  TensorOptions options(at::kQUInt8);

  // 创建自定义的无符号字节向量并分配内存
  auto custom_vec = std::make_unique<std::vector<uint8_t>>();
  custom_vec->resize(numel);

  // 获取自定义数据的指针
  uint8_t* custom_data = custom_vec->data();
  // 填充自定义数据
  for (const auto i : c10::irange(numel)) {
    custom_data[i] = i;
  }

  // 初始化自定义数据删除标志
  bool customDataDeleted{false};
  // 准备在完成时释放自定义向量
  auto deleteWhenDone = custom_vec.release();
  // 定义自定义释放器，并检查是否正确释放数据
  auto deleter = [deleteWhenDone, custom_data, &customDataDeleted](void* inp) {
    ASSERT_EQ((void*)inp, (void*)custom_data);
    delete deleteWhenDone;
    customDataDeleted = true;
  };

  {
    // 使用自定义数据创建量化的张量
    Tensor qtensor = at::from_blob_quantized_per_channel_affine(custom_data, shape, deleter, scales, zero_points, ch_axis, options);
    // 获取量化数据的指针
    uint8_t* q_data = (uint8_t*)qtensor.data_ptr<quint8>();
    // 检查量化数据是否与自定义数据相等
    for (const auto i : c10::irange(numel)) {
      ASSERT_EQ((int)custom_data[i], (int)q_data[i]);
    }
    // 检查量化张量的通道缩放因子是否正确
    ASSERT_TRUE(at::allclose(qtensor.q_per_channel_scales(), scales));
    // 检查量化张量的通道零点是否正确
    ASSERT_TRUE(at::allclose(qtensor.q_per_channel_zero_points(), zero_points));
    // 检查张量是否已经量化
    ASSERT_TRUE(qtensor.is_quantized());
  }
  // 检查自定义数据是否在释放后被删除
  TORCH_CHECK(customDataDeleted);
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
TEST(TestQTensor, TestArmVectorizedQuantizeDequantize) {
  // 设置量化的缩放因子
  const float scale = 7;
  // 张量中的元素数量
  const int numel = 132;

  // 创建浮点数向量，并填充数据
  std::vector<float> x_values;
  for (const auto i : c10::irange(numel)) {
    x_values.push_back(9 * i);
  }

  // 从浮点数向量创建张量
  const Tensor x = from_blob(x_values.data(), x_values.size());

  // 定义测试函数，用于不同的数据类型进行量化和反量化测试
  auto test_for_datatype = [&](
      const ScalarType scalar_type,
      const auto get_data_ptr,
      const auto quantize_val_with_datatype,
      const int zero_point_min,
      const int zero_point_max) {
    for (int zero_point : {zero_point_min, 10, zero_point_max}) {
      // 对张量进行量化
      const Tensor q = at::quantize_per_tensor(x, scale, zero_point, scalar_type);
      auto* q_data = get_data_ptr(q);
      // 检查量化后的值是否正确
      for (const auto i : c10::irange(numel)) {
        ASSERT_EQ(
          q_data[i].val_,
          quantize_val_with_datatype(scale, zero_point, x_values[i]).val_);
      }
      // 对量化后的张量进行反量化
      const Tensor r = q.dequantize();
      const float* r_data = r.const_data_ptr<float>();
      // 检查反量化后的值是否正确
      for (const auto i : c10::irange(numel)) {
        ASSERT_FLOAT_EQ(
          r_data[i],
          native::dequantize_val(scale, zero_point, q_data[i]));
      }
    }
  };

  // 无符号整数8位类型的量化测试
  test_for_datatype(
    kQUInt8,
    [](Tensor q) { return q.data_ptr<quint8>(); },
    native::quantize_val<quint8>,
    std::numeric_limits<uint8_t>::min(),
    std::numeric_limits<uint8_t>::max());

  // 有符号整数8位类型的量化测试
  test_for_datatype(
    kQInt8,
    [](Tensor q) { return q.data_ptr<qint8>(); },
    native::quantize_val<qint8>,
    std::numeric_limits<int8_t>::min(),
    std::numeric_limits<int8_t>::max());
}
#endif
  // 使用 std::numeric_limits<int8_t>::max() 来获取 int8_t 类型的最大值
  std::numeric_limits<int8_t>::max());

  // Signed Int 32 (not optimized with vectorization)
  // 对于类型 kQInt32 进行数据类型测试，使用 lambda 表达式捕获 Tensor q 并返回 qint32 类型的数据指针
  test_for_datatype(
    kQInt32,
    [](Tensor q) { return q.data_ptr<qint32>(); },
    // 使用 native::quantize_val<qint32> 进行 qint32 类型的量化
    native::quantize_val<qint32>,
    // 获取 int32_t 类型的最小值作为量化的下界
    std::numeric_limits<int32_t>::min(),
    // 获取 int32_t 类型的最大值作为量化的上界
    std::numeric_limits<int32_t>::max());
}
#endif // (__ARM_NEON__) || defined(__aarch64__)
```