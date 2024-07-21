# `.\pytorch\torch\csrc\api\include\torch\optim\lbfgs.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/nn/module.h>
// 包含 Torch 的神经网络模块头文件

#include <torch/optim/optimizer.h>
// 包含 Torch 的优化器头文件

#include <torch/optim/serialize.h>
// 包含 Torch 的优化器序列化相关头文件

#include <torch/serialize/archive.h>
// 包含 Torch 的序列化存档头文件

#include <deque>
// 包含双端队列头文件

#include <functional>
// 包含函数对象头文件

#include <memory>
// 包含智能指针头文件

#include <vector>
// 包含向量头文件

namespace torch {
namespace optim {

struct TORCH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  LBFGSOptions(double lr = 1);
  // LBFGSOptions 结构体的构造函数，默认学习率为 1
  TORCH_ARG(double, lr) = 1;
  // 定义 TORCH_ARG 宏，用于声明 lr 属性并初始化为 1
  TORCH_ARG(int64_t, max_iter) = 20;
  // 定义 TORCH_ARG 宏，声明 max_iter 属性并初始化为 20
  TORCH_ARG(std::optional<int64_t>, max_eval) = c10::nullopt;
  // 定义 TORCH_ARG 宏，声明 max_eval 属性并初始化为 c10::nullopt（空）
  TORCH_ARG(double, tolerance_grad) = 1e-7;
  // 定义 TORCH_ARG 宏，声明 tolerance_grad 属性并初始化为 1e-7
  TORCH_ARG(double, tolerance_change) = 1e-9;
  // 定义 TORCH_ARG 宏，声明 tolerance_change 属性并初始化为 1e-9
  TORCH_ARG(int64_t, history_size) = 100;
  // 定义 TORCH_ARG 宏，声明 history_size 属性并初始化为 100
  TORCH_ARG(std::optional<std::string>, line_search_fn) = c10::nullopt;
  // 定义 TORCH_ARG 宏，声明 line_search_fn 属性并初始化为 c10::nullopt（空）

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  // 实现序列化方法，接受输入存档对象 archive
  void serialize(torch::serialize::OutputArchive& archive) const override;
  // 实现序列化方法，接受输出存档对象 archive
  TORCH_API friend bool operator==(
      const LBFGSOptions& lhs,
      const LBFGSOptions& rhs);
  // 声明友元函数 operator==，用于比较两个 LBFGSOptions 对象是否相等
  double get_lr() const override;
  // 获取当前学习率的方法
  void set_lr(const double lr) override;
  // 设置学习率的方法
};

struct TORCH_API LBFGSParamState
    : public OptimizerCloneableParamState<LBFGSParamState> {
  TORCH_ARG(int64_t, func_evals) = 0;
  // 定义 TORCH_ARG 宏，声明 func_evals 属性并初始化为 0
  TORCH_ARG(int64_t, n_iter) = 0;
  // 定义 TORCH_ARG 宏，声明 n_iter 属性并初始化为 0
  TORCH_ARG(double, t) = 0;
  // 定义 TORCH_ARG 宏，声明 t 属性并初始化为 0
  TORCH_ARG(double, prev_loss) = 0;
  // 定义 TORCH_ARG 宏，声明 prev_loss 属性并初始化为 0
  TORCH_ARG(Tensor, d) = {};
  // 定义 TORCH_ARG 宏，声明 d 属性并初始化为空 Tensor
  TORCH_ARG(Tensor, H_diag) = {};
  // 定义 TORCH_ARG 宏，声明 H_diag 属性并初始化为空 Tensor
  TORCH_ARG(Tensor, prev_flat_grad) = {};
  // 定义 TORCH_ARG 宏，声明 prev_flat_grad 属性并初始化为空 Tensor
  TORCH_ARG(std::deque<Tensor>, old_dirs);
  // 定义 TORCH_ARG 宏，声明 old_dirs 属性为双端队列
  TORCH_ARG(std::deque<Tensor>, old_stps);
  // 定义 TORCH_ARG 宏，声明 old_stps 属性为双端队列
  TORCH_ARG(std::deque<Tensor>, ro);
  // 定义 TORCH_ARG 宏，声明 ro 属性为双端队列
  TORCH_ARG(std::optional<std::vector<Tensor>>, al) = c10::nullopt;
  // 定义 TORCH_ARG 宏，声明 al 属性为可选的 Tensor 向量，初始化为空

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  // 实现序列化方法，接受输入存档对象 archive
  void serialize(torch::serialize::OutputArchive& archive) const override;
  // 实现序列化方法，接受输出存档对象 archive
  TORCH_API friend bool operator==(
      const LBFGSParamState& lhs,
      const LBFGSParamState& rhs);
  // 声明友元函数 operator==，用于比较两个 LBFGSParamState 对象是否相等
};

class TORCH_API LBFGS : public Optimizer {
 public:
  explicit LBFGS(
      std::vector<OptimizerParamGroup> param_groups,
      LBFGSOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<LBFGSOptions>(defaults)) {
    // 显式声明构造函数，接受参数组列表和 LBFGSOptions 默认值对象
    TORCH_CHECK(
        param_groups_.size() == 1,
        "LBFGS doesn't support per-parameter options (parameter groups)");
    // 检查参数组的大小是否为 1，LBFGS 不支持每个参数的选项（参数组）
    if (defaults.max_eval() == c10::nullopt) {
      // 如果默认的 max_eval 未设置
      auto max_eval_val = (defaults.max_iter() * 5) / 4;
      // 计算 max_eval_val 为 max_iter 的 5/4 倍
      static_cast<LBFGSOptions&>(param_groups_[0].options())
          .max_eval(max_eval_val);
      // 设置第一个参数组的选项中的 max_eval 为 max_eval_val
      static_cast<LBFGSOptions&>(*defaults_.get()).max_eval(max_eval_val);
      // 设置默认选项中的 max_eval 为 max_eval_val
    }
    // 结束条件
    # 初始化_numel_cache为c10::nullopt
    _numel_cache = c10::nullopt;
    
    
    
    # LBFGS类的显式构造函数，接受参数列表params和LBFGSOptions默认参数
    explicit LBFGS(std::vector<Tensor> params, LBFGSOptions defaults = {})
        : LBFGS({OptimizerParamGroup(std::move(params))}, defaults) {}
    
    
    
    # LBFGS类的step方法，实现自Optimizer类的纯虚函数，执行优化步骤
    Tensor step(LossClosure closure) override;
    
    
    
    # LBFGS类的save方法，实现自serialize::OutputArchive的纯虚函数，保存对象状态到archive
    void save(serialize::OutputArchive& archive) const override;
    
    
    
    # LBFGS类的load方法，实现自serialize::InputArchive的纯虚函数，从archive加载对象状态
    void load(serialize::InputArchive& archive) override;
    
    
    
    # LBFGS类的私有成员变量，用于缓存参数张量的元素数量
    std::optional<int64_t> _numel_cache;
    
    
    
    # LBFGS类的私有方法，计算参数张量的元素数量
    int64_t _numel();
    
    
    
    # LBFGS类的私有方法，收集参数张量的梯度并展平成一维张量
    Tensor _gather_flat_grad();
    
    
    
    # LBFGS类的私有方法，根据给定的步长和更新张量，添加梯度更新
    void _add_grad(const double step_size, const Tensor& update);
    
    
    
    # LBFGS类的私有方法，根据给定的闭包函数和方向向量，评估方向导数
    std::tuple<double, Tensor> _directional_evaluate(
        const LossClosure& closure,
        const std::vector<Tensor>& x,
        double t,
        const Tensor& d);
    
    
    
    # LBFGS类的私有方法，设置参数张量的数值
    void _set_param(const std::vector<Tensor>& params_data);
    
    
    
    # LBFGS类的私有方法，克隆参数张量的副本
    std::vector<Tensor> _clone_param();
    
    
    
    # LBFGS类的静态模板方法，用于序列化对象self到archive
    template <typename Self, typename Archive>
    static void serialize(Self& self, Archive& archive) {
      _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(LBFGS);
    }
};
} // namespace optim
} // namespace torch
```