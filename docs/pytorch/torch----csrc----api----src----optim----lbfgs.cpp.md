# `.\pytorch\torch\csrc\api\src\optim\lbfgs.cpp`

```
// 包含 LBFGS 优化器的头文件
#include <torch/optim/lbfgs.h>

// 包含自动求导相关的头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>

// 包含序列化相关的头文件
#include <torch/serialize/archive.h>
#include <torch/utils.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
#include <c10/util/irange.h>

// 包含标准库的头文件
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// torch 命名空间下的 optim 命名空间
namespace torch {
namespace optim {

// LBFGSOptions 类的构造函数，初始化学习率 lr_
LBFGSOptions::LBFGSOptions(double lr) : lr_(lr) {}

// 比较两个 LBFGSOptions 对象是否相等的运算符重载函数
bool operator==(const LBFGSOptions& lhs, const LBFGSOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.max_iter() == rhs.max_iter()) &&
      (lhs.max_eval() == rhs.max_eval()) &&
      (lhs.tolerance_grad() == rhs.tolerance_grad()) &&
      (lhs.tolerance_change() == rhs.tolerance_change() &&
       (lhs.history_size() == rhs.history_size())) &&
      (lhs.line_search_fn() == rhs.line_search_fn());
}

// 将 LBFGSOptions 对象序列化为输出存档的函数
void LBFGSOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_iter);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_eval);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_grad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_change);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(history_size);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(line_search_fn);
}

// 将 LBFGSOptions 对象从输入存档中反序列化的函数
void LBFGSOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, max_iter);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(int64_t, max_eval);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_change);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, history_size);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(std::string, line_search_fn);
}

// 获取学习率 lr_ 的函数
double LBFGSOptions::get_lr() const {
  return lr();
}

// 设置学习率 lr_ 的函数
void LBFGSOptions::set_lr(const double lr) {
  this->lr(lr);
}

// 比较两个 LBFGSParamState 对象是否相等的运算符重载函数
bool operator==(const LBFGSParamState& lhs, const LBFGSParamState& rhs) {
  // 检查是否为 nullptr 的 Lambda 函数
  auto isNull = [](const std::optional<std::vector<Tensor>>& val) {
    return val == c10::nullopt;
  };
  return (lhs.func_evals() == rhs.func_evals()) &&
      (lhs.n_iter() == rhs.n_iter()) && (lhs.t() == rhs.t()) &&
      (lhs.prev_loss() == rhs.prev_loss()) &&
      torch::equal_if_defined(lhs.d(), rhs.d()) &&
      torch::equal_if_defined(lhs.H_diag(), rhs.H_diag()) &&
      torch::equal_if_defined(lhs.prev_flat_grad(), rhs.prev_flat_grad()) &&
      // 比较容器 old_dirs() 是否相等的辅助函数
      if_container_equal(lhs.old_dirs(), rhs.old_dirs()) &&
      // 比较容器 old_stps() 是否相等的辅助函数
      if_container_equal(lhs.old_stps(), rhs.old_stps()) &&
      // 比较容器 ro() 是否相等的辅助函数
      if_container_equal(lhs.ro(), rhs.ro()) &&
      // 比较可选的容器 al() 是否相等的辅助函数
      ((isNull(lhs.al()) && isNull(rhs.al())) ||
       (!isNull(lhs.al()) && !isNull(rhs.al()) &&
        if_container_equal(*lhs.al(), *rhs.al())));
}
// 将 LBFGSParamState 对象的数据序列化到输出存档中
void LBFGSParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  // 序列化 func_evals 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(func_evals);
  // 序列化 n_iter 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(n_iter);
  // 序列化 t 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(t);
  // 序列化 prev_loss 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_loss);
  // 序列化 d 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(d);
  // 序列化 H_diag 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(H_diag);
  // 序列化 prev_flat_grad 变量
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_flat_grad);
  // 序列化 old_dirs 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(old_dirs);
  // 序列化 old_stps 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(old_stps);
  // 序列化 ro 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(ro);
  // 仅在 al 变量非空时序列化（Python 版本中只有显式定义的状态变量才会被序列化）
  if (al() != c10::nullopt) {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(al);
  }
}

// 从输入存档中反序列化 LBFGSParamState 对象的数据
void LBFGSParamState::serialize(torch::serialize::InputArchive& archive) {
  // 反序列化 func_evals 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, func_evals);
  // 反序列化 n_iter 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, n_iter);
  // 反序列化 t 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, t);
  // 反序列化 prev_loss 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, prev_loss);
  // 反序列化 d 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, d);
  // 反序列化 H_diag 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, H_diag);
  // 反序列化 prev_flat_grad 变量
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev_flat_grad);
  // 反序列化 old_dirs 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, old_dirs);
  // 反序列化 old_stps 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, old_stps);
  // 反序列化 ro 变量（类型为 deque<Tensor>）
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, ro);
  // 反序列化 al 可选变量（类型为 optional<vector<Tensor>>）
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(std::vector<Tensor>, al);
}

// 收集所有参数的梯度，并将其展平为一个张量
Tensor LBFGS::_gather_flat_grad() {
  // 存储每个参数视图的集合
  std::vector<Tensor> views;
  // 遍历第一个参数组中的所有参数
  for (const auto& p : param_groups_.at(0).params()) {
    // 如果参数梯度未定义，则创建一个形状为 (参数元素数目,) 的张量并填充为零
    if (!p.grad().defined()) {
      views.emplace_back(p.new_empty({p.numel()}).zero_());
    } else if (p.grad().is_sparse()) {  // 如果参数梯度是稀疏的
      // 将稀疏梯度转换为稠密张量后展平为一维张量
      views.emplace_back(p.grad().to_dense().view(-1));
    } else {
      // 直接展平参数梯度为一维张量
      views.emplace_back(p.grad().view(-1));
    }
  }
  // 拼接所有视图，形成一个展平的张量
  return torch::cat(views, 0);
}

// 计算所有参数总元素数
int64_t LBFGS::_numel() {
  // 如果缓存中没有元素数目，则计算并缓存
  if (_numel_cache == c10::nullopt) {
    auto res = 0;
    // 遍历第一个参数组中的所有参数，累加元素数目
    for (const auto& p : param_groups_.at(0).params()) {
      res += p.numel();
    }
    // 将计算结果存入缓存
    _numel_cache = res;
  }
  // 返回缓存中的元素数目
  return *_numel_cache;
}

// 添加梯度到参数中，使用给定的步长
void LBFGS::_add_grad(const double step_size, const Tensor& update) {
  auto offset = 0;
  // 遍历第一个参数组中的所有参数
  for (auto& p : param_groups_.at(0).params()) {
    auto numel = p.numel();
    // 使用偏移量和参数元素数目对梯度更新进行切片，并视图成参数的形状后添加到参数中
    p.add_(
        update.index({at::indexing::Slice(offset, offset + numel)}).view_as(p),
        step_size);
    // 更新偏移量到下一个参数的起始位置
    offset += numel;
  }
  // 断言偏移量等于所有参数的总元素数目
  TORCH_INTERNAL_ASSERT(offset == _numel());
}

// 设置参数为给定的参数数据
void LBFGS::_set_param(const std::vector<Tensor>& params_data) {
  auto& _params = param_groups_.at(0).params();
  // 断言给定参数数据的大小与模型参数的数量相同
  TORCH_INTERNAL_ASSERT(params_data.size() == _params.size());
  // 将给定的参数数据复制到模型参数中
  for (const auto i : c10::irange(_params.size())) {
    _params.at(i).copy_(params_data.at(i));
  }
}

// 克隆参数，并返回克隆后的结果
std::vector<Tensor> LBFGS::_clone_param() {
  std::vector<Tensor> result;
  // 遍历第一个参数组中的所有参数，克隆每一个参数并添加到结果集合中
  for (const auto& p : param_groups_.at(0).params()) {
    result.emplace_back(p.clone(at::MemoryFormat::Contiguous));
  }
  // 返回包含所有克隆参数的集合
  return result;
}
// 计算梯度并添加到当前方向上
_add_grad(t, d);

// 启用自动求导模式，并计算闭包函数的损失值
double loss;
{
  torch::AutoGradMode enable_grad(true);
  loss = closure().item<double>();
}

// 收集平铺后的梯度
auto flat_grad = _gather_flat_grad();

// 设置参数为给定的向量 x
_set_param(x);

// 返回损失值和平铺后的梯度作为元组
return std::make_tuple(loss, flat_grad);



// 三次插值函数，用于计算给定点 x1 和 x2 之间的插值
// 基于 https://github.com/torch/optim/blob/master/polyinterp.lua 移植
double _cubic_interpolate(
    double x1,
    double f1,
    double g1,
    double x2,
    double f2,
    double g2,
    std::optional<std::tuple<double, double>> bounds = c10::nullopt) {

  // 计算插值区域的边界
  double xmin_bound, xmax_bound;
  if (bounds != c10::nullopt) {
    std::tie(xmin_bound, xmax_bound) = *bounds;
  } else {
    std::tie(xmin_bound, xmax_bound) =
        (x1 <= x2) ? std::make_tuple(x1, x2) : std::make_tuple(x2, x1);
  }

  // 计算常见情况下的三次插值
  auto d1 = (g1 + g2) - (3 * (f1 - f2) / (x1 - x2));
  auto d2_square = std::pow(d1, 2);

  double d2;
  if (d2_square >= g1 * g2) {
    d2 = std::sqrt(d2_square - g1 * g2);

    double min_pos;
    if (x1 <= x2) {
      min_pos = x2 - ((x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)));
    } else {
      min_pos = x1 - ((x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)));
    }

    // 返回插值结果，确保在边界内
    return std::min(std::max(min_pos, xmin_bound), xmax_bound);
  } else {
    // 当无法进行三次插值时，返回边界中点
    return (xmin_bound + xmax_bound) / 2;
  }
}



// 强 Wolfe 条件满足判断函数
// 使用 obj_func 评估目标函数的值和梯度，以及给定的参数 t 和 d
std::tuple<double, Tensor, double, int64_t> _strong_wolfe(
    const Function& obj_func,
    const std::vector<Tensor>& x,
    double t,
    const Tensor& d,
    double f,
    Tensor g,
    const Tensor& gtd,
    double c1 = 1e-4,
    double c2 = 0.9, // Wolfe 条件的默认参数
    double tolerance_change = 1e-9,
    ```
    double max_ls = 25) { // NOLINT(cppcoreguidelines-avoid-magic-numbers)


    // 设置最大的线搜索步数，用于控制迭代次数
    // NOLINT(cppcoreguidelines-avoid-magic-numbers): 避免使用魔术数字的Lint规则
    // max_ls 为最大的线搜索步数，指定为25，以控制迭代次数
    double max_ls = 25;

  auto val = [](const Tensor& t) { return t.item<double>(); };

  auto d_norm = val(d.abs().max());
  g = g.clone(at::MemoryFormat::Contiguous);
  // evaluate objective and gradient using initial step
  auto [f_new, g_new] = obj_func(x, t, d);
  int64_t ls_func_evals = 1;
  auto gtd_new = g_new.dot(d);

  // bracket an interval containing a point satisfying the Wolfe criteria
  double t_prev = 0;
  auto f_prev = f;
  auto g_prev = g;
  auto gtd_prev = gtd;
  bool done = false;
  auto ls_iter = 0;
  std::vector<double> bracket, bracket_f;
  std::vector<Tensor> bracket_g, bracket_gtd;

  while (ls_iter < max_ls) {
    // check conditions
    if ((f_new > (f + c1 * t * val(gtd))) ||
        (ls_iter > 1 && (f_new >= f_prev))) {
      bracket = {t_prev, t};
      bracket_f = {f_prev, f_new};
      bracket_g = {g_prev, g_new.clone(at::MemoryFormat::Contiguous)};
      bracket_gtd = {gtd_prev, gtd_new};
      break;
    }
    if (std::abs(val(gtd_new)) <= (-c2 * val(gtd))) {
      bracket = {t, t};
      bracket_f = {f_new, f_new};
      bracket_g = {g_new, g_new};
      done = true;
      break;
    }
    if (val(gtd_new) >= 0) {
      bracket = {t_prev, t};
      bracket_f = {f_prev, f_new};
      bracket_g = {g_prev, g_new.clone(at::MemoryFormat::Contiguous)};
      bracket_gtd = {gtd_prev, gtd_new};
      break;
    }
    // interpolate
    auto min_step = t +
        0.01 * (t - t_prev); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    auto max_step = t * 10; // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    auto tmp = t;
    t = _cubic_interpolate(
        t_prev,
        f_prev,
        val(gtd_prev),
        t,
        f_new,
        val(gtd_new),
        std::make_tuple(min_step, max_step));
    // next step
    t_prev = tmp;
    f_prev = f_new;
    g_prev = g_new.clone(at::MemoryFormat::Contiguous);
    gtd_prev = gtd_new;
    std::tie(f_new, g_new) = obj_func(x, t, d);
    ls_func_evals += 1;
    gtd_new = g_new.dot(d);
    ls_iter += 1;
  }
  // reached max number of iterations?
  if (ls_iter == max_ls) {
    bracket = {0, t};
    bracket_f = {f, f_new};
    bracket_g = {g, g_new};
  }

  // zoom phase: we now have a point satisfying the criteria, or
  // a bracket around it. We refine the bracket until we find the
  // exact point satisfying the criteria
  bool insuf_progress = false;
  // find high and low points in bracket
  auto [low_pos, high_pos] = bracket_f[0] <= bracket_f[1]
      ? std::make_tuple(0, 1)
      : std::make_tuple(1, 0);
  while (!done && (ls_iter < max_ls)) {
    // compute new trial value
    t = _cubic_interpolate(
        bracket[0],
        bracket_f[0],
        val(bracket_gtd[0]),
        bracket[1],
        bracket_f[1],
        val(bracket_gtd[1]));

    // test that we are making sufficient progress:
    // in case `t` is so close to boundary, we mark that we are making
    // insufficient progress, and if
    // 计算 bracket 数组中的最大值和最小值
    double bracket_max = std::max(bracket[0], bracket[1]);
    auto bracket_min = std::min(bracket[0], bracket[1]);
    // 计算 eps，为 bracket 最大值和最小值的差的 0.1 倍
    auto eps = 0.1 * (bracket_max - bracket_min); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    
    // 如果 t 距离 bracket 的边界点的距离小于 eps，则进行插值调整
    if (std::min(bracket_max - t, t - bracket_min) < eps) {
      // 如果上一步进展不足或者 t 处于边界上，则将 t 调整到离最近边界点 eps 距离的位置
      if (insuf_progress || (t >= bracket_max) || (t <= bracket_min)) {
        t = (std::abs(t - bracket_max) < std::abs(t - bracket_min))
            ? bracket_max - eps
            : bracket_min + eps;
        insuf_progress = false;
      } else {
        insuf_progress = true;
      }
    } else {
      insuf_progress = false;
    }

    // 计算新点的函数值和梯度值
    std::tie(f_new, g_new) = obj_func(x, t, d);
    ls_func_evals += 1;
    gtd_new = g_new.dot(d);
    ls_iter += 1;

    // 判断 Armijo 条件和 Wolfe 条件是否满足，更新 bracket 数组和相关信息
    if ((f_new > (f + c1 * t * val(gtd))) || (f_new >= bracket_f[low_pos])) {
      // Armijo 条件未满足或者不低于最低点
      bracket[high_pos] = t;
      bracket_f[high_pos] = f_new;
      bracket_g[high_pos] = g_new.clone(at::MemoryFormat::Contiguous);
      bracket_gtd[high_pos] = gtd_new;
      std::tie(low_pos, high_pos) = bracket_f[0] <= bracket_f[1]
          ? std::make_tuple(0, 1)
          : std::make_tuple(1, 0);
    } else {
      if (val(at::abs(gtd_new)) <= (-c2 * val(gtd))) {
        // Wolfe 条件满足
        done = true;
      } else if ((val(gtd_new) * (bracket[high_pos] - bracket[low_pos])) >= 0) {
        // 将原高点设为新低点
        bracket[high_pos] = bracket[low_pos];
        bracket_f[high_pos] = bracket_f[low_pos];
        bracket_g[high_pos] = bracket_g[low_pos];
        bracket_gtd[high_pos] = bracket_gtd[low_pos];
      }

      // 将新点设为新低点
      bracket[low_pos] = t;
      bracket_f[low_pos] = f_new;
      bracket_g[low_pos] = g_new.clone(at::MemoryFormat::Contiguous);
      bracket_gtd[low_pos] = gtd_new;
    }

    // 如果 bracket 数组的范围很小，则结束搜索
    if ((std::abs(bracket[1] - bracket[0]) * d_norm) < tolerance_change)
      break;
  }

  // 返回结果
  t = bracket[low_pos];
  f_new = bracket_f[low_pos];
  g_new = bracket_g[low_pos];
  return std::make_tuple(f_new, g_new, t, ls_func_evals);
  // 结束 LBFGS::step 方法定义
  Tensor LBFGS::step(LossClosure closure) {
    // 进入无梯度计算环境
    NoGradGuard no_grad;
    // 检查闭包函数是否存在
    TORCH_CHECK(closure != nullptr, "LBFGS requires a closure function");
    // 内部断言，验证参数组的大小为1
    TORCH_INTERNAL_ASSERT(param_groups_.size() == 1);
    // 定义一个函数 val，用于返回 Tensor 的 double 值
    auto val = [](const Tensor& t) { return t.item<double>(); };

    // 获取参数组中的第一个参数组
    auto& group = param_groups_.at(0);
    // 获取参数列表
    auto& _params = group.params();
    // 获取 LBFGSOptions 对象的引用
    const auto& options = static_cast<const LBFGSOptions&>(group.options());
    // 获取优化器学习率 lr
    auto lr = options.lr();
    // 获取最大迭代次数 max_iter
    auto max_iter = options.max_iter();
    // 获取最大评估次数 max_eval
    auto max_eval = options.max_eval();
    // 获取梯度容差 tolerance_grad
    auto tolerance_grad = options.tolerance_grad();
    // 获取变化容差 tolerance_change
    auto tolerance_change = options.tolerance_change();
    // 获取线性搜索函数 line_search_fn
    auto line_search_fn = options.line_search_fn();
    // 获取历史大小 history_size
    auto history_size = options.history_size();

    // 注意：LBFGS 只有全局状态，但我们将其注册为第一个参数的状态，
    // 这有助于在 load_state_dict 中进行类型转换
    // 查找第一个参数的状态
    auto param_state = state_.find(_params.at(0).unsafeGetTensorImpl());
    // 如果状态不存在，则创建一个新的 LBFGSParamState 对象
    if (param_state == state_.end()) {
      state_[_params.at(0).unsafeGetTensorImpl()] =
          std::make_unique<LBFGSParamState>();
    }
    // 获取参数状态的引用
    auto& state = static_cast<LBFGSParamState&>(
        *state_[_params.at(0).unsafeGetTensorImpl()]);

    // 计算初始的 f(x) 和 df/dx
    Tensor orig_loss;
    {
      // 开启自动梯度计算模式
      torch::AutoGradMode enable_grad(true);
      // 计算闭包函数的值
      orig_loss = closure();
    }

    // 计算初始损失值
    auto loss = val(orig_loss);
    // 当前评估次数加一
    auto current_evals = 1;
    // 更新函数评估次数
    state.func_evals(state.func_evals() + 1);
    // 获取扁平化梯度
    auto flat_grad = _gather_flat_grad();
    // 计算优化条件
    auto opt_cond = (val(flat_grad.abs().max()) <= tolerance_grad);

    // 如果满足优化条件，则直接返回初始损失值
    if (opt_cond) {
      return orig_loss;
    }

    // 以下是在状态中缓存的张量（用于追踪）
    auto& d = state.d();
    auto& t = state.t();
    auto& old_dirs = state.old_dirs();
    auto& old_stps = state.old_stps();
    auto& ro = state.ro();
    auto& H_diag = state.H_diag();
    auto& prev_flat_grad = state.prev_flat_grad();
    auto& prev_loss = state.prev_loss();

    // 迭代最大 max_iter 次数进行优化
    while (n_iter < max_iter) {
      // 迭代次数加一
      n_iter += 1;
      // 更新迭代次数到状态
      state.n_iter(state.n_iter() + 1);

      // 计算梯度下降方向
      if (state.n_iter() == 1) {
        // 初次迭代，梯度下降方向为负梯度方向
        d = flat_grad.neg();
        // Hessian 对角线元素初始化为1
        H_diag = torch::tensor(1);
        // 清空历史方向和步长
        old_dirs = {};
        old_stps = {};
        ro = {};
    } else {
      // 执行 LBFGS 更新（更新内存）
      auto y = flat_grad.sub(prev_flat_grad);  // 计算梯度的差异 y = grad - prev_grad
      auto s = d.mul(t);  // 计算方向的缩放 s = d * t
      auto ys = y.dot(s); // 计算 y 和 s 的点积 ys = y * s
      if (val(ys) > 1e-10) { // 如果 ys 大于阈值 1e-10
        // 更新内存
        if (static_cast<int64_t>(old_dirs.size()) == history_size) {
          // 将历史记录向前移动一位（有限内存）
          old_dirs.pop_front();  // 移除最早的方向
          old_stps.pop_front();  // 移除最早的步长
          ro.pop_front();  // 移除最早的 ro
        }
        // 存储新的方向/步长
        old_dirs.emplace_back(y);  // 将新的 y 添加到 old_dirs
        old_stps.emplace_back(s);  // 将新的 s 添加到 old_stps
        ro.emplace_back(1. / ys);  // 将新的 1/ys 添加到 ro

        // 更新初始 Hessian 近似的缩放
        H_diag = ys / y.dot(y); // 更新 H_diag = ys / (y * y)
      }

      // 计算近似的 L-BFGS 逆 Hessian 乘以梯度
      int64_t num_old = static_cast<int64_t>(old_dirs.size());

      if (state.al() == c10::nullopt) {
        state.al(std::vector<Tensor>(history_size));
      }
      auto& al = state.al();

      // 将 L-BFGS 循环中的迭代折叠为使用一个缓冲区
      auto q = flat_grad.neg();  // 取梯度的负值作为初始 q
      for (int64_t i = num_old - 1; i > -1; i--) {
        (*al).at(i) = old_stps.at(i).dot(q) * ro.at(i);  // 更新 al[i] = old_stps[i] * q * ro[i]
        q.add_(old_dirs.at(i), -val((*al).at(i)));  // 更新 q = q - old_dirs[i] * al[i]
      }

      // 乘以初始 Hessian
      // r/d 是最终的方向
      auto r = torch::mul(q, H_diag);  // 计算 r = q * H_diag
      d = r;  // 更新 d = r
      for (const auto i : c10::irange(num_old)) {
        auto be_i = old_dirs.at(i).dot(r) * ro.at(i);  // 计算 be_i = old_dirs[i] * r * ro[i]
        r.add_(old_stps.at(i), val((*al).at(i) - be_i));  // 更新 r = r + old_stps[i] * (al[i] - be_i)
      }
    }

    if (!prev_flat_grad.defined()) {
      prev_flat_grad = flat_grad.clone(at::MemoryFormat::Contiguous);  // 如果 prev_flat_grad 未定义，则克隆 flat_grad
    } else {
      prev_flat_grad.copy_(flat_grad);  // 否则将 flat_grad 复制给 prev_flat_grad
    }
    prev_loss = loss;  // 更新 prev_loss

    // ############################################################
    // # 计算步长
    // ############################################################
    // 重置步长的初始猜测
    if (state.n_iter() == 1) {
      t = std::min(1., 1. / val(flat_grad.abs().sum())) * lr;  // 根据梯度绝对值的总和计算步长 t
    } else {
      t = lr;  // 否则步长为 lr
    }

    // 计算方向导数
    auto gtd = flat_grad.dot(d); // 计算梯度与方向的点积 gtd = flat_grad * d

    // 方向导数低于容差
    if (val(gtd) > -tolerance_change)
      break;  // 如果 gtd 大于 -tolerance_change，跳出循环

    // 可选的线搜索：用户函数
    auto ls_func_evals = 0;  // 初始化线搜索函数的评估次数为 0
    if (line_search_fn != c10::nullopt) {
      TORCH_CHECK(
          *line_search_fn == "strong_wolfe",
          "only 'strong_wolfe' is supported");  // 检查线搜索函数是否为 strong_wolfe
      auto x_init = _clone_param();  // 克隆参数作为初始值
      auto obj_func =
          [&](const std::vector<Tensor>& x, double t, const Tensor& d) {
            return _directional_evaluate(closure, x, t, d);  // 定义目标函数
          };
      std::tie(loss, flat_grad, t, ls_func_evals) =
          _strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd);  // 进行 strong_wolfe 线搜索
      _add_grad(t, d);  // 添加梯度
      opt_cond = (val(flat_grad.abs().max()) <= tolerance_grad);  // 更新优化条件


注释完成。
    } else {
      // 如果没有线搜索，则简单地使用固定步长移动
      _add_grad(t, d);  // 添加梯度乘以步长到当前参数中
      if (n_iter != max_iter) {
        // 如果不是在最后一次迭代中，则重新评估函数
        // 这么做的原因是在随机设置中，在这里重新评估函数没有意义
        {
          torch::AutoGradMode enable_grad(true);  // 启用自动梯度计算模式
          loss = val(closure());  // 计算损失函数的值
        }
        flat_grad = _gather_flat_grad();  // 收集平铺的梯度
        opt_cond = val(torch::max(flat_grad.abs())) <= tolerance_grad;  // 判断是否满足梯度容差条件
        ls_func_evals = 1;  // 线搜索函数评估次数设为1
      }
    }
    // 更新函数评估次数
    current_evals += ls_func_evals;
    state.func_evals(state.func_evals() + ls_func_evals);

    // ############################################################
    // # 检查终止条件
    // ############################################################
    if (n_iter == max_iter)
      break;  // 达到最大迭代次数时跳出循环

    if (current_evals >= *max_eval)
      break;  // 达到最大评估次数时跳出循环

    // 达到最优条件时跳出循环
    if (opt_cond)
      break;

    // 检查是否缺乏进展
    if (val(d.mul(t).abs().max()) <= tolerance_change)
      break;  // 如果参数变化的绝对值小于等于变化容差，则跳出循环

    if (std::abs(loss - prev_loss) < tolerance_change)
      break;  // 如果损失函数变化小于变化容差，则跳出循环
  }

  return orig_loss;  // 返回初始损失值
}

void LBFGS::save(serialize::OutputArchive& archive) const {
  // 将当前 LBFGS 对象序列化到输出存档中
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  // 创建一个空的 IValue 对象用于存储 pytorch_version
  IValue pytorch_version;
  // 尝试从存档中读取 "pytorch_version"，如果成功则执行下一步序列化操作
  if (archive.try_read("pytorch_version", pytorch_version)) {
    // 序列化当前 LBFGS 对象到存档中
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    // 输出警告信息，指示正在使用旧的序列化格式加载 LBFGS 优化器
    TORCH_WARN(
        "Your serialized LBFGS optimizer is still using the old serialization format. "
        "The func_evals and n_iter value in state will be set to 0, ro will be set to an empty deque "
        "and al will be set to c10::nullopt because the old LBFGS optimizer didn't save these values."
        "You should re-save your LBFGS optimizer to use the new serialization format.");
    // 定义需要加载的 Tensor 变量
    Tensor d, t, H_diag, prev_flat_grad, prev_loss;
    std::deque<Tensor> old_dirs, old_stps;
    // 从存档中加载各个 Tensor 变量的数据
    archive("d", d, /*is_buffer=*/true);
    archive("t", t, /*is_buffer=*/true);
    archive("H_diag", H_diag, /*is_buffer=*/true);
    archive("prev_flat_grad", prev_flat_grad, /*is_buffer=*/true);
    archive("prev_loss", prev_loss, /*is_buffer=*/true);
    // 加载 deque 类型的数据 old_dirs 和 old_stps
    torch::optim::serialize(archive, "old_dirs", old_dirs);
    torch::optim::serialize(archive, "old_stps", old_stps);

    // 注意：LBFGS 只有一个全局状态，但我们将其注册为第一个参数的状态，这有助于在 load_state_dict 中进行类型转换
    auto state = std::make_unique<LBFGSParamState>();
    // 设置 state 对象的各个成员变量值
    state->d(d);
    state->t(t.item<double>());
    state->H_diag(H_diag);
    state->prev_flat_grad(prev_flat_grad);
    state->prev_loss(prev_loss.item<double>());
    state->old_dirs(old_dirs);
    state->old_stps(old_stps);
    // 将 state 对象移动到 state_ 中的相应参数位置
    state_[param_groups_.at(0).params().at(0).unsafeGetTensorImpl()] =
        std::move(state);
  }
}
} // namespace optim
} // namespace torch
```