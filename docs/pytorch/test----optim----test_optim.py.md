# `.\pytorch\test\optim\test_optim.py`

```
# Owner(s): ["module: optimizer"]

# 导入 PyTorch 库
import torch
# 从 torch.optim 中导入各种优化器类
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
)
# 从 torch.testing._internal.common_utils 中导入测试相关的工具函数和类
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    skipIfTorchDynamo,
    TestCase,
)

# load_tests 函数来自 common_utils 用于在 sandcastle 上自动筛选测试，这行代码用于消除 flake 警告
load_tests = load_tests

# 定义一个函数 _diff_fn，用于计算梯度检查函数的差分
def _diff_fn(p, grad, opt_differentiable_state, opt_class, kwargs, *ignored):
    # 深拷贝参数 p
    p = p.clone()
    # 设置参数 p 的梯度
    p.grad = grad
    # 深拷贝优化器状态中的张量，以便在函数输入中正确跟踪状态张量
    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }
    # 创建优化器对象 opt
    opt = opt_class([p], **kwargs)
    # 更新优化器状态中参数 p 的值
    opt.state[p].update(opt_differentiable_state)
    # 执行优化器的一步优化过程
    opt.step()
    # 返回更新后的参数 p 和其梯度相关的状态张量
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )

# 装饰器，用于在 Torch Dynamo 模式下跳过不支持的可微分优化器测试
@skipIfTorchDynamo("Differentiable optimizers not supported")
class TestDifferentiableOptimizer(TestCase):
    # 测试 SGD 优化器的函数
    def test_sgd(self):
        # 创建随机张量 p，要求计算其梯度，数据类型为 float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建随机梯度张量 grad，数据类型为 float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建随机张量 mbuff，要求计算其梯度，数据类型为 float64
        mbuff = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 定义状态字典，包含动量缓冲器 mbuff
        state = {"momentum_buffer": mbuff}
        # 调用 gradcheck 函数进行梯度检查
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                SGD,  # 使用 SGD 优化器类
                {"lr": 0.9, "differentiable": True},  # 优化器的参数设置
                *state.values(),  # 状态字典的值作为额外参数传入
            ),
        )

    # 测试 Adam 优化器的函数
    def test_adam(self):
        # 定义空的状态字典
        state = {}
        # 创建随机张量 p，要求计算其梯度，数据类型为 float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建随机梯度张量 grad，数据类型为 float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 定义状态字典中的各种张量，要求计算其梯度，数据类型为 float64
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)  # 步数
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)  # 指数平均值
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)  # 指数平均平方
        state["max_exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)  # 最大指数平均平方

        # 调用 gradcheck 函数进行梯度检查
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adam,  # 使用 Adam 优化器类
                {"lr": 0.9, "differentiable": True, "amsgrad": True},  # 优化器的参数设置
                *state.values(),  # 状态字典的值作为额外参数传入
            ),
        )
    def test_rmsprop(self):
        state = {}
        # 创建空字典用于存储优化器状态信息
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 p，要求其梯度信息，并使用 float64 数据类型
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 grad，要求其梯度信息，并使用 float64 数据类型
        state["step"] = torch.zeros((), dtype=torch.float64)
        # 在状态字典中添加名为 "step" 的张量，其值为零，数据类型为 float64
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 在状态字典中添加名为 "square_avg" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型
        state["momentum_buffer"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        # 在状态字典中添加名为 "momentum_buffer" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型
        # 可能会因 sqrt 操作导致大值和 NaN 问题
        state["grad_avg"] = 1e-2 * torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        # 在状态字典中添加名为 "grad_avg" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型，值为 0.01 乘以随机张量
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RMSprop,
                {
                    "lr": 0.9,
                    "maximize": True,
                    "momentum": 0.9,
                    "differentiable": True,
                    "centered": True,
                    "weight_decay": 0.1,
                },
                *state.values(),
            ),
        )

    def test_adadelta(self):
        state = {}
        # 创建空字典用于存储优化器状态信息
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 p，要求其梯度信息，并使用 float64 数据类型
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 grad，要求其梯度信息，并使用 float64 数据类型
        # `step` 不是连续变量（尽管我们定义为浮点数），因此不应该要求梯度。
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 在状态字典中添加名为 "step" 的张量，其值为 10.0，不要求梯度，数据类型为 float64
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 在状态字典中添加名为 "square_avg" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型
        state["acc_delta"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 在状态字典中添加名为 "acc_delta" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adadelta,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adagrad(self):
        state = {}
        # 创建空字典用于存储优化器状态信息
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 p，要求其梯度信息，并使用 float64 数据类型
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个大小为 10 的张量 grad，要求其梯度信息，并使用 float64 数据类型
        # `step` 不是连续变量（尽管我们定义为浮点数），因此不应该要求梯度。
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 在状态字典中添加名为 "step" 的张量，其值为 10.0，不要求梯度，数据类型为 float64
        state["sum"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 在状态字典中添加名为 "sum" 的张量，大小为 10，要求其梯度信息，并使用 float64 数据类型
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adagrad,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )
    def test_adamax(self):
        # 初始化空字典 state
        state = {}
        # 创建一个形状为 (10,) 的张量 p，要求梯度，数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 grad，要求梯度，数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 将步数 `step` 设置为 torch.tensor(10.0)，不要求梯度，数据类型为 torch.float64
        # 这里表示 Adamax 算法中的步数
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 exp_avg，要求梯度，数据类型为 torch.float64
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 exp_inf，要求梯度，数据类型为 torch.float64
        state["exp_inf"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 使用 gradcheck 函数检查梯度更新函数 _diff_fn 的正确性
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adamax,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )

    @skipIfTorchDynamo(
        "The inplace mu update fails with dynamo, "
        "since this is only happening when differentiable is enabled, skipping for now"
    )
    def test_asgd(self):
        # 初始化空字典 state
        state = {}
        # 创建一个形状为 (10,) 的张量 p，要求梯度，数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 grad，要求梯度，数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 将步数 `step` 设置为 torch.tensor(10.0)，不要求梯度，数据类型为 torch.float64
        # eta 和 mu 也设置为不要求梯度的 torch.tensor(0.9) 和 torch.tensor(1.0)
        # 这些变量在 ASGD 算法中不是连续变量，因此不需要梯度
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["eta"] = torch.tensor(0.9, requires_grad=False, dtype=torch.float64)
        state["mu"] = torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 ax，要求梯度，数据类型为 torch.float64
        state["ax"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        # 使用 gradcheck 函数检查梯度更新函数 _diff_fn 的正确性
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                ASGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_rprop(self):
        # 初始化空字典 state
        state = {}
        # 创建一个形状为 (10,) 的张量 p，要求梯度，数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 grad，要求梯度，数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 将步数 `step` 设置为 torch.tensor(10.0)，不要求梯度，数据类型为 torch.float64
        # 在 Rprop 算法中，step 不是连续变量，因此不需要梯度
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 prev，要求梯度，数据类型为 torch.float64
        state["prev"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 step_size，要求梯度，数据类型为 torch.float64
        state["step_size"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        # 使用 gradcheck 函数检查梯度更新函数 _diff_fn 的正确性
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Rprop,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )
    # 定义一个测试函数，用于测试 AdamW 优化器的梯度检查
    def test_adamw(self):
        # 初始化状态字典
        state = {}
        # 创建一个形状为 (10,) 的随机张量 p，要求计算梯度，数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 grad，要求计算梯度，数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` 是一个 float 类型的常量，不需要计算梯度
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 exp_avg，要求计算梯度，数据类型为 torch.float64
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 exp_avg_sq，要求计算梯度，数据类型为 torch.float64
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 max_exp_avg_sq，要求计算梯度，数据类型为 torch.float64
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        # 调用 gradcheck 函数来检查梯度
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                AdamW,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    # 定义一个测试函数，用于测试 NAdam 优化器的梯度检查
    def test_nadam(self):
        # 初始化状态字典
        state = {}
        # 创建一个形状为 (10,) 的随机张量 p，要求计算梯度，数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 grad，要求计算梯度，数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` 是一个 float 类型的常量，不需要计算梯度
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 exp_avg，要求计算梯度，数据类型为 torch.float64
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的随机张量 exp_avg_sq，要求计算梯度，数据类型为 torch.float64
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个常量张量 mu_product，要求计算梯度，数据类型为 torch.float64
        state["mu_product"] = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)

        # 调用 gradcheck 函数来检查梯度，测试第一个配置的参数
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

        # 调用 gradcheck 函数来检查梯度，测试第二个配置的参数
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )
    # 定义一个测试函数 `test_radam`，用于测试某个功能或者类的行为
    def test_radam(self):
        # 初始化一个空的状态字典
        state = {}
        # 创建一个形状为 (10,) 的张量 p，要求其梯度信息，并且数据类型为 torch.float64
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 grad，要求其梯度信息，并且数据类型为 torch.float64
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 定义一个名为 "step" 的张量，并赋值为 10.0，不要求其梯度信息，数据类型为 torch.float64
        # 在优化算法中，step 通常代表优化步骤的计数或者步长，不需要计算梯度
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 exp_avg，要求其梯度信息，并且数据类型为 torch.float64
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # 创建一个形状为 (10,) 的张量 exp_avg_sq，要求其梯度信息，并且数据类型为 torch.float64
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        # 运行一个名为 gradcheck 的函数，用于检查一个函数在给定参数下的梯度
        gradcheck(
            _diff_fn,
            (
                p,  # 输入参数 p
                grad,  # 输入参数 grad
                state,  # 输入参数 state
                RAdam,  # 输入参数 RAdam（一个类或者函数）
                {"lr": 0.9, "differentiable": True},  # 输入参数的字典形式
                *state.values(),  # 扩展 state 字典中的所有值作为额外参数传递给 gradcheck
            ),
        )
        # 再次运行 gradcheck 函数，但是这次提供了更多的参数
        gradcheck(
            _diff_fn,
            (
                p,  # 输入参数 p
                grad,  # 输入参数 grad
                state,  # 输入参数 state
                RAdam,  # 输入参数 RAdam（一个类或者函数）
                {
                    "lr": 0.9,  # 学习率为 0.9
                    "weight_decay": 0.1,  # 权重衰减为 0.1
                    "decoupled_weight_decay": True,  # 使用分离的权重衰减
                    "differentiable": True,  # 设置为可导
                },
                *state.values(),  # 扩展 state 字典中的所有值作为额外参数传递给 gradcheck
            ),
        )
# 如果脚本直接被执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 打印提示信息，建议通过 test/test_optim.py 脚本运行测试
    print("These tests should be run through test/test_optim.py instead")
```