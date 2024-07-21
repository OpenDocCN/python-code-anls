# `.\pytorch\torch\csrc\api\src\nn\modules\adaptive.cpp`

```
// 包含C++标准库和Torch库的头文件，用于导入必要的类和函数
#include <c10/util/irange.h>
#include <torch/nn/modules/adaptive.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>

// 命名空间别名，简化torch::nn::functional的引用为F
namespace F = torch::nn::functional;

// 使用指定的命名空间，使得torch::indexing的内容可以直接使用
using namespace torch::indexing;

// 定义torch命名空间内的nn命名空间
namespace torch {
namespace nn {

// 构造函数实现，初始化ASMoutput对象
ASMoutput::ASMoutput(Tensor output_, double loss_)
    : output(std::move(output_)), loss(loss_) {}

// AdaptiveLogSoftmaxWithLossImpl类的构造函数实现
AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl(
    AdaptiveLogSoftmaxWithLossOptions options_)
    : options(std::move(options_)),
      shortlist_size(0),
      n_clusters(0),
      head_size(0) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用reset函数进行初始化设置
  reset();
}

// 重置函数，初始化和检查配置
void AdaptiveLogSoftmaxWithLossImpl::reset() {
  // 检查和验证cutoffs参数的合法性
  TORCH_CHECK(
      options.cutoffs().size() > 0,
      "cutoffs should be a sequence of length larger than 0");
  TORCH_CHECK(
      std::is_sorted(options.cutoffs().begin(), options.cutoffs().end()) &&
          *std::min_element(
              options.cutoffs().begin(), options.cutoffs().end()) > 0 &&
          *std::max_element(
              options.cutoffs().begin(), options.cutoffs().end()) <=
              (options.n_classes() - 1) &&
          std::set<int64_t>(options.cutoffs().begin(), options.cutoffs().end())
                  .size() == options.cutoffs().size(),
      "cutoffs should be a sequence of unique, positive integers sorted in an increasing order, ",
      "where each value is between 1 and n_classes-1");
  TORCH_CHECK(options.div_value() != 0, "div_value should not be equal to 0");

  // 将cutoffs赋值给成员变量cutoffs，并加入一个额外的元素作为上界
  cutoffs = options.cutoffs();
  cutoffs.push_back(options.n_classes());

  // 计算shortlist_size、n_clusters和head_size的值
  shortlist_size = cutoffs[0];
  n_clusters = cutoffs.size() - 1;
  head_size = shortlist_size + n_clusters;

  // 创建并注册head模块，配置输入和输出的线性层
  head = this->register_module(
      "head",
      Linear(LinearOptions(options.in_features(), head_size)
                 .bias(options.head_bias())));
  
  // 创建并注册tail模块，使用ModuleList保存多个Sequential模块
  tail = this->register_module("tail", ModuleList());

  // 遍历n_clusters范围内的每个索引，创建并添加projection序列模块到tail
  for (const auto i : c10::irange(n_clusters)) {
    // 计算当前投影的输入和输出大小
    int64_t hsz = static_cast<int64_t>(std::floor(
        options.in_features() / std::pow(options.div_value(), (i + 1))));
    int64_t osz = cutoffs[i + 1] - cutoffs[i];

    // 创建并添加当前投影的Sequential模块，包含两个线性层
    Sequential projection(
        Linear(LinearOptions(options.in_features(), hsz).bias(false)),
        Linear(LinearOptions(hsz, osz).bias(false)));
    tail->push_back(projection);
  }
}

// 重置模型参数函数，用于重置head和tail的线性层参数
void AdaptiveLogSoftmaxWithLossImpl::reset_parameters() {
  // 重置head模块的参数
  head->reset_parameters();
  
  // 遍历tail中的每个模块，重置其子模块i2h和h2o的线性层参数
  for (const auto i : c10::irange(tail->size())) {
    auto i2h = tail[i]->children()[0]->as<Linear>();
    auto h2o = tail[i]->children()[1]->as<Linear>();
    i2h->reset_parameters();
    h2o->reset_parameters();
  }
}

// 前向传播函数，计算输入的softmax损失和输出
ASMoutput AdaptiveLogSoftmaxWithLossImpl::forward(
    const Tensor& input_,
    const Tensor& target_) {
  auto targ_dim = target_.dim();

  // 检查目标张量的维度是否符合预期
  TORCH_CHECK(
      targ_dim == 1 || targ_dim == 0,
      "0D or 1D target tensor expected, multi-target not supported");

  // 如果目标张量是一维的
  if (targ_dim == 1) {
    // 检查输入张量的维度是否符合预期，给出相应的错误信息
    TORCH_CHECK(
        input_.dim() == 2,
        "1D target tensor expects 2D input tensors, but found inputs with sizes ",
        input_.sizes(),
        ".");
  } else {
    // 检查输入张量的维度是否符合预期，给出相应的错误信息
    TORCH_CHECK(
        input_.dim() == 1,
        "0D target tensor expects 1D input tensors, but found inputs with sizes ",
        input_.sizes(),
        ".");
  }

  // 确定是否为批处理输入
  bool is_batched = (targ_dim > 0);
  // 根据是否为批处理选择输入张量的处理方式
  Tensor input = is_batched ? input_ : input_.unsqueeze(0);
  // 根据是否为批处理选择目标张量的处理方式
  Tensor target = is_batched ? target_ : target_.unsqueeze(0);

  // 初始化已使用的行数为 0，获取批处理大小
  int64_t used_rows = 0;
  const int64_t batch_size = target.size(0);

  // 初始化输出张量为与输入张量相同类型的全零张量
  Tensor output = input.new_zeros(batch_size);
  // 初始化收集索引张量为与目标张量相同类型的空张量
  Tensor gather_inds = target.new_empty(batch_size);

  // 复制截断值，并在开头插入 0
  auto cutoff_values = cutoffs;
  cutoff_values.insert(cutoff_values.begin(), 0);

  // 遍历截断值列表，进行相应的处理
  for (const auto i : c10::irange(cutoff_values.size() - 1)) {
    // 获取当前截断值区间的下限和上限
    int64_t low_idx = cutoff_values[i];
    int64_t high_idx = cutoff_values[i + 1];

    // 根据当前截断值区间生成目标掩码张量，用于选择目标张量中符合条件的索引
    const Tensor target_mask = (target >= low_idx) * (target < high_idx);
    // 获取目标掩码张量中非零元素的索引，并压缩成一维张量
    const Tensor row_indices = target_mask.nonzero().squeeze();

    // 如果当前区间没有符合条件的索引，则继续下一轮循环
    if (row_indices.numel() == 0) {
      continue;
    }

    // 根据当前区间是否为第一个区间，选择不同的操作
    if (i == 0) {
      // 如果是第一个区间，则直接复制目标张量中符合条件的元素到收集索引张量
      gather_inds.index_copy_(0, row_indices, target.index({target_mask}));
    } else {
      // 如果不是第一个区间，则需进一步处理目标相对值和输入子集
      Tensor relative_target = target.index({target_mask}) - low_idx;
      Tensor input_subset = input.index_select(0, row_indices);

      // 对输入子集进行处理，计算聚类输出
      const Tensor cluster_output =
          tail[i - 1]->as<Sequential>()->forward(input_subset);
      int64_t cluster_index = shortlist_size + i - 1;

      // 将聚类索引填充到收集索引张量对应位置
      gather_inds.index_fill_(0, row_indices, cluster_index);

      // 计算聚类输出的对数概率，并获取局部对数概率
      const Tensor cluster_logprob = F::log_softmax(cluster_output, 1);
      const Tensor local_logprob =
          cluster_logprob.gather(1, relative_target.unsqueeze(1));
      // 将局部对数概率复制到输出张量对应位置
      output.index_copy_(0, row_indices, local_logprob.squeeze(1));
    }

    // 更新已使用的行数计数
    used_rows += row_indices.numel();
  }

  // 检查使用的行数是否与批处理大小相等，否则输出相应的错误信息
  TORCH_CHECK(
      used_rows == batch_size,
      "Target values should be in [0, ",
      options.n_classes() - 1,
      "], "
      "but values in range [",
      target.min().item().toDouble(),
      ", ",
      target.max().item().toDouble(),
      "] "
      "were found. ");

  // 对输入应用头部处理，计算头部输出的对数概率
  const Tensor head_output = head(input);
  const Tensor head_logprob = F::log_softmax(head_output, 1);
  // 将头部对数概率与收集索引张量对应位置进行加和，并压缩成一维张量
  output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze();
  // 计算输出张量的负均值作为损失值
  const double loss = (-output).mean().item().toDouble();

  // 如果不是批处理，则将输出张量压缩成一维张量
  if (!is_batched) {
    output = output.squeeze(0);
  }

  // 返回最终的输出结果，包括输出张量和损失值
  return ASMoutput(output, loss);
}

// 返回完整的对数概率
Tensor AdaptiveLogSoftmaxWithLossImpl::_get_full_log_prob(
    const Tensor& input,
    const Tensor& head_output) {
  // 创建一个与输入相同形状的空张量，用于存储输出
  Tensor out = input.new_empty({head_output.size(0), options.n_classes()});
  // 计算头部输出的对数概率
  const Tensor head_logprob = F::log_softmax(head_output, 1);

  // 将头部输出的对数概率放入输出张量的指定位置
  out.index_put_(
      {Slice(), Slice(None, shortlist_size)},
      head_logprob.index({Slice(), Slice(None, shortlist_size)}));

  // 遍历截断点数组
  for (const auto i : c10::irange(cutoffs.size() - 1)) {
    // 获取当前截断点和下一个截断点的索引
    int64_t start_idx = cutoffs[i];
    int64_t stop_idx = cutoffs[i + 1];
    // 计算当前簇的输出和对数概率
    const Tensor cluster_output = tail[i]->as<Sequential>()->forward(input);
    const Tensor cluster_logprob = F::log_softmax(cluster_output, 1);
    // 计算输出的对数概率，考虑到短列表和当前簇的贡献
    auto output_logprob = cluster_logprob +
        head_logprob.index({Slice(), static_cast<int64_t>(shortlist_size + i)})
            .unsqueeze(1);

    // 将计算得到的对数概率放入输出张量的指定位置
    out.index_put_({Slice(), Slice(start_idx, stop_idx)}, output_logprob);
  }
  return out; // 返回填充完整的对数概率张量
}

// 计算输入的对数概率
Tensor AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl::log_prob(
    const Tensor& input) {
  // 计算头部输出
  const Tensor head_output = head(input);
  // 调用_get_full_log_prob计算完整的对数概率并返回
  return _get_full_log_prob(input, head_output);
}

// 预测函数
Tensor AdaptiveLogSoftmaxWithLossImpl::predict(const Tensor& input) {
  // 计算头部输出
  const Tensor head_output = head(input);
  // 使用头部输出进行预测
  Tensor output = torch::argmax(head_output, 1);
  auto not_in_shortlist = (output >= shortlist_size);
  auto all_in_shortlist = bitwise_not(not_in_shortlist.any());

  if (all_in_shortlist.item().toBool()) {
    return output; // 如果所有预测都在短列表中，则直接返回输出
  } else if (not_in_shortlist.all().item().toBool()) {
    // 如果所有预测都不在短列表中，则计算完整的对数概率并返回预测
    const Tensor log_prob = _get_full_log_prob(input, head_output);
    return torch::argmax(log_prob, 1);
  } else {
    // 否则，计算非短列表中的输入的对数概率，并更新输出
    const Tensor log_prob = _get_full_log_prob(
        input.index({not_in_shortlist}), head_output.index({not_in_shortlist}));
    output.index_put_({not_in_shortlist}, torch::argmax(log_prob, 1));
    return output;
  }
}

// 打印函数，输出模型信息
void AdaptiveLogSoftmaxWithLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AdaptiveLogSoftmaxWithLoss";
}

} // namespace nn
} // namespace torch
```