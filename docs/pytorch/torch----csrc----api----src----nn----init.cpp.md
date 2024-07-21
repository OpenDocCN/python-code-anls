# `.\pytorch\torch\csrc\api\src\nn\init.cpp`

```
// 引入 Torch 库中的初始化头文件
#include <torch/nn/init.h>

// 引入 Torch 的线性代数、类型和实用函数头文件
#include <torch/linalg.h>
#include <torch/types.h>
#include <torch/utils.h>

// 引入 ATen 库中的核心头文件
#include <ATen/ATen.h>
// 引入 C10 库中的异常处理和范围迭代头文件
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 引入标准库头文件
#include <algorithm>  // 引入算法标准库，用于 STL 算法
#include <cmath>      // 引入数学标准库，用于数学计算
#include <cstddef>    // 引入 cstddef 标准库，用于 nullptr_t 和 size_t 等
#include <tuple>      // 引入元组标准库，用于元组操作

// Torch 命名空间
namespace torch {
// Torch 的神经网络命名空间
namespace nn {
// 初始化模块的命名空间
namespace init {
// 匿名命名空间，用于限定符号的作用域
namespace {

// Fan 结构体，用于计算输入和输出的尺寸
struct Fan {
  // 构造函数，计算输入和输出的尺寸
  explicit Fan(Tensor& tensor) {
    const auto dimensions = tensor.ndimension();
    // 检查张量维度是否大于等于 2
    TORCH_CHECK(
        dimensions >= 2,
        "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");

    if (dimensions == 2) {
      // 对于二维张量，输入大小为第二维，输出大小为第一维
      in = tensor.size(1);
      out = tensor.size(0);
    } else {
      // 对于高维张量，计算输入和输出的尺寸
      in = tensor.size(1) * tensor[0][0].numel();
      out = tensor.size(0) * tensor[0][0].numel();
    }
  }

  int64_t in;   // 输入大小
  int64_t out;  // 输出大小
};

// 计算 Kaiming 初始化的标准差
double calculate_kaiming_std(
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  // 进入无梯度上下文
  NoGradGuard guard;
  // 创建 Fan 对象，计算输入和输出大小
  Fan fan(tensor);
  // 计算激活函数的增益
  const auto gain = calculate_gain(nonlinearity, a);
  double std = 0.0;

  // 根据初始化模式选择标准差计算方式
  if (std::holds_alternative<enumtype::kFanIn>(mode)) {
    std = gain / std::sqrt(fan.in);  // Fan-in 初始化
  } else {
    std = gain / std::sqrt(fan.out);  // Fan-out 初始化
  }
  return std;  // 返回计算得到的标准差
}

// 根据非线性类型和参数计算增益
double calculate_gain(NonlinearityType nonlinearity, double param) {
  // 根据非线性类型返回相应的增益值
  if (std::holds_alternative<enumtype::kTanh>(nonlinearity)) {
    return 5.0 / 3.0;  // Tanh 激活函数的默认增益
  } else if (std::holds_alternative<enumtype::kReLU>(nonlinearity)) {
    return std::sqrt(2.0);  // ReLU 激活函数的默认增益
  } else if (std::holds_alternative<enumtype::kLeakyReLU>(nonlinearity)) {
    return std::sqrt(2.0 / (1 + pow(param, 2)));  // LeakyReLU 激活函数的增益
  }

  return 1.0;  // 默认增益为 1.0
}

// 将张量的所有元素填充为指定的标量值
Tensor constant_(Tensor tensor, Scalar value) {
  // 进入无梯度上下文
  NoGradGuard guard;
  return tensor.fill_(value);  // 使用指定的标量值填充张量并返回
}

// 生成一个单位矩阵
Tensor eye_(Tensor matrix) {
  // 进入无梯度上下文
  NoGradGuard guard;
  // 检查输入张量维度是否为 2
  TORCH_CHECK(
      matrix.ndimension() == 2, "Only tensors with 2 dimensions are supported");
  return torch::eye_out(matrix, matrix.size(0), matrix.size(1));  // 生成并返回单位矩阵
}

// 生成一个均值为 mean，标准差为 std 的正态分布张量
Tensor normal_(Tensor tensor, double mean, double std) {
  // 进入无梯度上下文
  NoGradGuard guard;
  return tensor.normal_(mean, std);  // 生成正态分布张量并返回
}

// 将张量的所有元素填充为 1
Tensor ones_(Tensor tensor) {
  // 进入无梯度上下文
  NoGradGuard guard;
  return tensor.fill_(1);  // 使用值 1 填充张量并返回
}

// 创建一个 Dirac δ 函数张量
Tensor dirac_(Tensor tensor) {
  // 进入无梯度上下文
  NoGradGuard guard;

  // 检查张量维度是否在支持的范围内
  TORCH_CHECK(
      tensor.ndimension() >= 3 && tensor.ndimension() <= 5,
      "Only tensors with 3, 4, or 5 dimensions are supported");

  const auto sizes = tensor.sizes();
  const auto min_dim = std::min(sizes[0], sizes[1]);

  tensor.zero_();  // 将张量所有元素置零

  // 根据张量维度不同，设置 Dirac δ 函数值
  for (const auto d : c10::irange(min_dim)) {
    switch (tensor.ndimension()) {
      case 3:  // 三维张量，用于时间卷积
        tensor[d][d][sizes[2] / 2] = 1;
        break;
      case 4:  // 四维张量，用于空间卷积
        tensor[d][d][sizes[2] / 2][sizes[3] / 2] = 1;
        break;
      case 5:  // 五维张量，用于体积卷积
        tensor[d][d][sizes[2] / 2][sizes[3] / 2][sizes[4] / 2] = 1;
        break;
    }
  }

  return tensor;  // 返回设置好的 Dirac δ 函数张量
}

}  // namespace
}  // namespace init
}  // namespace nn
}  // namespace torch
// 计算受控环境下的正交化
Tensor orthogonal_(Tensor tensor, double gain) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 检查张量维度是否至少为2
  TORCH_CHECK(
      tensor.ndimension() >= 2,
      "Only tensors with 2 or more dimensions are supported");

  // 获取张量的行数和列数
  const auto rows = tensor.size(0);
  const auto columns = tensor.numel() / rows;

  // 创建一个形状为[rows, columns]的随机张量
  auto flattened = torch::randn({rows, columns});

  // 如果行数小于列数，则转置flattened张量
  if (rows < columns) {
    flattened.t_();
  }

  // 计算qr分解
  auto [q, r] = torch::linalg::qr(flattened);

  // 根据文献调整Q使其更均匀分布
  auto d = torch::diag(r, 0);
  auto ph = d.sign();
  q *= ph;

  // 如果行数小于列数，则再次转置q
  if (rows < columns) {
    q.t_();
  }

  // 将q视为与输入张量相同形状，并复制到输入张量中
  tensor.view_as(q).copy_(q);

  // 将张量乘以增益
  tensor.mul_(gain);

  // 返回处理后的张量
  return tensor;
}

// 在受控环境中创建稀疏张量
Tensor sparse_(Tensor tensor, double sparsity, double std) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 检查张量维度是否为2
  TORCH_CHECK(
      tensor.ndimension() == 2, "Only tensors with 2 dimensions are supported");

  // 获取张量的行数和列数
  const auto rows = tensor.size(0);
  const auto columns = tensor.size(1);

  // 计算需要置零的元素数量
  const int64_t num_zeros = std::ceil(sparsity * rows);

  // 使用正态分布填充张量
  tensor.normal_(0, std);

  // 对每一列执行稀疏化操作
  for (const auto column : c10::irange(columns)) {
    // 创建一个随机排列的行索引
    auto row_indices = torch::randperm(rows, tensor.options().dtype(kLong));

    // 选择需要置零的索引范围
    auto zero_indices =
        row_indices.slice(/*dim=*/0, /*start=*/0, /*end=*/num_zeros);

    // 将选定的索引位置置零
    tensor.index_put_(
        {zero_indices, torch::tensor(column, tensor.options().dtype(kLong))},
        torch::zeros(num_zeros, tensor.options()));
  }

  // 返回处理后的张量
  return tensor;
}

// 在受控环境中创建均匀分布的张量
Tensor uniform_(Tensor tensor, double low, double high) {
  // 取消梯度追踪
  NoGradGuard guard;
  // 使用均匀分布填充张量
  return tensor.uniform_(low, high);
}

// 使用Kaiming初始化方法创建均匀分布的张量
Tensor kaiming_uniform_(
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 计算Kaiming标准差
  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);

  // 根据标准差计算均匀分布的边界
  const auto bound = std::sqrt(3.0) * std;

  // 使用均匀分布填充张量
  return tensor.uniform_(-bound, bound);
}

// 使用Kaiming初始化方法创建正态分布的张量
Tensor kaiming_normal_(
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 计算Kaiming标准差
  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);

  // 使用正态分布填充张量
  return tensor.normal_(0, std);
}

// 使用Xavier初始化方法创建正态分布的张量
Tensor xavier_normal_(Tensor tensor, double gain) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 计算输入输出通道的fan-in和fan-out
  Fan fan(tensor);

  // 计算Xavier初始化的标准差
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));

  // 使用正态分布填充张量
  return tensor.normal_(0, std);
}

// 使用Xavier初始化方法创建均匀分布的张量
Tensor xavier_uniform_(Tensor tensor, double gain) {
  // 取消梯度追踪
  NoGradGuard guard;

  // 计算输入输出通道的fan-in和fan-out
  Fan fan(tensor);

  // 计算Xavier初始化的标准差
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));

  // 根据标准差计算均匀分布的边界
  const auto a = std::sqrt(3.0) * std;

  // 使用均匀分布填充张量
  return tensor.uniform_(-a, a);
}

// 在受控环境中将张量置零
Tensor zeros_(Tensor tensor) {
  // 取消梯度追踪
  NoGradGuard guard;
  // 将张量所有元素置零
  return tensor.zero_();
}

// 计算张量的输入输出通道数
std::tuple<int64_t, int64_t> _calculate_fan_in_and_fan_out(
    // 计算输入张量的“输入特征数”和“输出特征数”
    std::tuple<int64_t, int64_t> compute_in_and_out_features(
        // 传入的张量对象
        const Tensor& tensor) {
      // 获取张量的维度
      const auto dimensions = tensor.dim();
      // 检查张量维度是否至少为2，否则抛出错误信息
      TORCH_CHECK(
          dimensions >= 2,
          "Fan in and fan out can not be computed "
          "for tensor with fewer than 2 dimensions")
    
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t fan_in, fan_out;
      // 如果张量维度为2，表示是线性层的情况
      if (dimensions == 2) { // Linear
        // 计算线性层的“输入特征数”和“输出特征数”
        fan_in = tensor.size(1);
        fan_out = tensor.size(0);
      } else {
        // 对于非线性层（如卷积层等），计算其“输入特征数”和“输出特征数”
        const auto num_input_fmaps = tensor.size(1);
        const auto num_output_fmaps = tensor.size(0);
        auto receptive_field_size = 1;
        // 如果张量维度大于2，则计算感受野大小
        if (tensor.dim() > 2) {
          receptive_field_size = tensor[0][0].numel();
        }
        // 计算“输入特征数”和“输出特征数”
        fan_in = num_input_fmaps * receptive_field_size;
        fan_out = num_output_fmaps * receptive_field_size;
      }
      // 返回计算得到的“输入特征数”和“输出特征数”的元组
      return std::tie(fan_in, fan_out);
    }
}
} // namespace init
} // namespace nn
} // namespace torch
```